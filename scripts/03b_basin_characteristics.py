"""03b_basin_characteristics.py

Compute and store basin-level physical characteristics for every Indiana
streamflow gauge.  Reads the station inventory (script 01) and the watershed
polygons from S3 (script 03).

Characteristics produced per gauge
-----------------------------------
drain_area_mi2   Drainage area (mi²); from NWIS drain_area_va, falling back to
                 the geodesic area of the NLDI watershed polygon from script 03.
stream_length_mi Length of the upstream main stem (mi); from the NLDI UM
                 navigation endpoint (2000-mi cap).
slope_ft_mi      10-85 main-channel slope (ft/mi); 3DEP elevation queried at
                 the 10th and 85th percentile distance points along the main
                 stem, divided by 75% of the total channel length.
tc_hr            Kirpich (1940) time of concentration (hours):
                     Tc (min) = 0.0078 × L_ft^0.77 × S_ftft^-0.385
                 where L_ft   = stream_length_mi × 5280,
                       S_ftft = slope_ft_mi / 5280.
pct_u            % urban land cover (NLCD 2019 developed low/medium/high
                 intensity, classes 22+23+24, total watershed); from the EPA
                 StreamCat API (api.epa.gov/StreamCAT/metrics) by ComID.
pct_w            % water/wetland cover (NLCD 2019 open water + woody and
                 herbaceous wetlands, classes 11+90+95, total watershed); from
                 the EPA StreamCat API by ComID.

Writes:
    s3://<bucket>/<prefix>watersheds/basin_characteristics.parquet
"""
from __future__ import annotations

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import requests
from pyproj import Geod

from utils import RetryPolicy, load_config, s3_client, with_retries, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("03b_basin_char")

NLDI_BASE      = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
EPQS_URL       = "https://epqs.nationalmap.gov/v1/json"
STREAMCAT_BASE = "https://api.epa.gov/StreamCat/streams/metrics"

# EPA StreamCat metric names (Ws suffix = total upstream watershed).
# Urban (%U): NLCD 2019 developed low (22) / medium (23) / high intensity (24)
_URBAN_METRICS = ["PctUrbLo2019", "PctUrbMd2019", "PctUrbHi2019"]
# Water (%W): NLCD 2019 open water (11) / woody (90) / herbaceous wetlands (95)
_WATER_METRICS = ["PctOw2019", "PctWdWet2019", "PctHbWet2019"]


# ── Distance / elevation helpers ──────────────────────────────────────────────

def _haversine_mi(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin((lon2 - lon1) * math.pi / 360) ** 2
    )
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _elevation_ft(lon: float, lat: float, timeout: int) -> float:
    r = requests.get(
        EPQS_URL,
        params={"x": lon, "y": lat, "units": "Feet", "wkid": 4326, "includeDate": "false"},
        timeout=timeout,
    )
    r.raise_for_status()
    return float(r.json()["value"])


# ── Channel geometry ──────────────────────────────────────────────────────────

def fetch_channel_geometry(
    site_no: str, timeout: int
) -> tuple[Optional[float], Optional[float]]:
    """Return (stream_length_mi, slope_ft_mi) from the NLDI upstream main stem
    and USGS 3DEP elevation at the 10th and 85th percentile distance points.

    Returns (None, None) if the main stem cannot be fetched or is too short.
    Raises requests.RequestException on network failure (caller should retry).
    """
    url = f"{NLDI_BASE}/USGS-{site_no}/navigation/UM/flowlines?distance=2000"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    coords: list[tuple[float, float]] = []
    for feat in r.json().get("features", []):
        coords.extend(feat["geometry"]["coordinates"])
    if len(coords) < 2:
        return None, None

    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + _haversine_mi(*coords[i - 1], *coords[i]))
    total_mi = cum[-1]
    if total_mi <= 0:
        return None, None

    def _coord_at(frac: float) -> tuple[float, float]:
        target = frac * total_mi
        for i in range(1, len(cum)):
            if cum[i] >= target:
                return coords[i]
        return coords[-1]

    p10 = _coord_at(0.10)
    p85 = _coord_at(0.85)
    e10 = _elevation_ft(*p10, timeout)
    e85 = _elevation_ft(*p85, timeout)

    slope_ft_mi = abs(e85 - e10) / (0.75 * total_mi)
    return total_mi, max(slope_ft_mi, 0.1)


def kirpich_tc_hr(length_mi: float, slope_ft_mi: float) -> float:
    """Kirpich (1940) time of concentration in hours.

    Tc (min) = 0.0078 × L_ft^0.77 × S_ftft^-0.385
    """
    L_ft   = length_mi * 5280.0
    S_ftft = slope_ft_mi / 5280.0
    return 0.0078 * (L_ft ** 0.77) * (S_ftft ** -0.385) / 60.0


# ── Land cover characteristics ────────────────────────────────────────────────

def _get_comid(site_no: str, timeout: int) -> str:
    """Return the NHDPlus ComID for a USGS streamgage via NLDI.

    The /linked-data/nwissite feature endpoint does not consistently expose a
    comid property, so we resolve it from the nearest upstream flowline, which
    carries an nhdpv2_COMID property in all NLDI versions.
    Raises requests.RequestException on network failure (caller should retry).
    Raises ValueError if no flowlines are found or the ComID is absent.
    """
    url = f"{NLDI_BASE}/USGS-{site_no}/navigation/UM/flowlines?distance=10"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    features = r.json().get("features", [])
    if not features:
        raise ValueError(f"no upstream flowlines found for site {site_no}")
    props = features[0].get("properties", {})
    comid = props.get("nhdplus_comid") or props.get("nhdpv2_COMID") or props.get("comid") or props.get("COMID")
    if not comid:
        raise ValueError(
            f"no ComID in flowline properties for site {site_no}; "
            f"available keys: {list(props.keys())}"
        )
    return str(comid)


def fetch_land_cover(
    comid: str, timeout: int
) -> tuple[Optional[float], Optional[float]]:
    """Fetch total-watershed urban and water/wetland fractions from EPA StreamCat.

    Returns (pct_u, pct_w) as percentages (0–100), or (None, None) if the
    ComID is not found in the StreamCat dataset.
    Raises requests.RequestException on network failure (caller should retry).
    """
    all_metrics = _URBAN_METRICS + _WATER_METRICS
    r = requests.get(
        STREAMCAT_BASE,
        params={
            "name": ",".join(all_metrics),
            "comid": comid,
            "areaOfInterest": "watershed",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    items = r.json().get("items") or r.json().get("Items") or []
    if not items:
        return None, None
    row = items[0]
    # StreamCat returns all field names in lowercase (e.g. "pcturblo2019ws")
    pct_u = sum(row.get(f"{m.lower()}ws") or 0.0 for m in _URBAN_METRICS)
    pct_w = sum(row.get(f"{m.lower()}ws") or 0.0 for m in _WATER_METRICS)
    return pct_u, pct_w


# ── Drainage area ─────────────────────────────────────────────────────────────

_M2_PER_MI2 = 2_589_988.1
_GEOD = Geod(ellps="WGS84")


def area_mi2_from_s3_geojson(site_no: str, bucket: str, prefix: str) -> Optional[float]:
    key = f"{prefix}watersheds/per_gauge/{site_no}.geojson"
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        gdf = gpd.read_file(io.BytesIO(obj["Body"].read()))
        if gdf.empty:
            return None
        total_m2 = sum(
            abs(_GEOD.geometry_area_perimeter(geom)[0])
            for geom in gdf.geometry
            if geom is not None
        )
        return total_m2 / _M2_PER_MI2 if total_m2 > 0 else None
    except Exception as e:
        log.debug("%s: polygon area failed: %s", site_no, e)
        return None


# ── Per-station processor ─────────────────────────────────────────────────────

def process_site(
    site_no: str,
    da_hint: Optional[float],
    bucket: str,
    prefix: str,
    cfg: dict,
) -> dict:
    timeout = cfg["streamstats"]["request_timeout_sec"]
    out: dict = {
        "site_no":          site_no,
        "drain_area_mi2":   None,
        "stream_length_mi": None,
        "slope_ft_mi":      None,
        "tc_hr":            None,
        "pct_u":            None,
        "pct_w":            None,
    }

    # ── Drainage area ──────────────────────────────────────────────────────
    da: Optional[float] = (
        float(da_hint)
        if (da_hint is not None and not pd.isna(da_hint) and float(da_hint) > 0)
        else None
    )
    if da is None:
        da = area_mi2_from_s3_geojson(site_no, bucket, prefix)
        if da is not None:
            log.debug("%s: drain area from S3 polygon = %.2f mi²", site_no, da)
    if da is None:
        log.warning("%s: no drainage area available", site_no)
    out["drain_area_mi2"] = round(da, 4) if da is not None else None

    # ── Channel geometry (length + 10-85 slope) ────────────────────────────
    try:
        length_mi, slope = with_retries(
            lambda: fetch_channel_geometry(site_no, timeout),
            RetryPolicy(max_attempts=3, base_delay=3.0),
            exceptions=(requests.RequestException,),
        )
    except Exception as e:
        log.warning("%s: channel geometry failed (%s)", site_no, e)
        length_mi, slope = None, None

    out["stream_length_mi"] = round(length_mi, 4) if length_mi is not None else None
    out["slope_ft_mi"]      = round(slope, 3)      if slope      is not None else None

    # ── Kirpich Tc ─────────────────────────────────────────────────────────
    if length_mi is not None and slope is not None:
        out["tc_hr"] = round(kirpich_tc_hr(length_mi, slope), 3)

    # ── ComID → needed for StreamCat land cover ────────────────────────────
    comid: Optional[str] = None
    try:
        comid = with_retries(
            lambda: _get_comid(site_no, timeout),
            RetryPolicy(max_attempts=3, base_delay=3.0),
            exceptions=(requests.RequestException,),
        )
    except Exception as e:
        log.warning("%s: comid lookup failed (%s) — land cover will be null", site_no, e)

    # ── Land cover (pct_u, pct_w) ──────────────────────────────────────────
    if comid is not None:
        try:
            pct_u, pct_w = with_retries(
                lambda: fetch_land_cover(comid, timeout),
                RetryPolicy(max_attempts=3, base_delay=3.0),
                exceptions=(requests.RequestException,),
            )
            out["pct_u"] = round(pct_u, 2) if pct_u is not None else None
            out["pct_w"] = round(pct_w, 2) if pct_w is not None else None
        except Exception as e:
            log.warning("%s: land cover fetch failed (%s)", site_no, e)

    log.debug(
        "%s: da=%.2f mi² len=%.2f mi slope=%.2f ft/mi tc=%.2f hr pct_u=%s pct_w=%s",
        site_no,
        da or 0.0, length_mi or 0.0, slope or 0.0, out["tc_hr"] or 0.0,
        out["pct_u"], out["pct_w"],
    )
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    out_key = f"{prefix}watersheds/basin_characteristics.parquet"

    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    inv = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    inv["site_no"] = inv["site_no"].astype(str)
    log.info("Loaded %d stations from inventory", len(inv))

    # Skip stations that already have all fields populated
    complete_sites: set[str] = set()
    existing = pd.DataFrame()
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=out_key)
        existing = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
        existing["site_no"] = existing["site_no"].astype(str)
        required = ["drain_area_mi2", "stream_length_mi", "slope_ft_mi", "tc_hr", "pct_u", "pct_w"]
        complete_sites = set(existing.dropna(subset=required)["site_no"])
        log.info(
            "Existing output: %d rows, %d complete — skipping those",
            len(existing), len(complete_sites),
        )
    except Exception:
        log.info("No existing output — running fresh")

    targets = inv[~inv["site_no"].isin(complete_sites)].copy()
    log.info("Stations to process: %d", len(targets))
    if targets.empty:
        log.info("Nothing to do.")
        return

    max_workers = cfg["streamstats"].get("max_concurrent", 8)
    results: list[dict] = []
    n = len(targets)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                process_site,
                str(row["site_no"]),
                row.get("drain_area_va"),
                bucket,
                prefix,
                cfg,
            ): str(row["site_no"])
            for _, row in targets.iterrows()
        }
        done = 0
        for fut in as_completed(futs):
            site = futs[fut]
            done += 1
            try:
                results.append(fut.result())
            except Exception as e:
                log.error("%s: unexpected error: %s", site, e)
            if done % 25 == 0 or done == n:
                log.info("[%d/%d]", done, n)

    parts: list[pd.DataFrame] = []
    if not existing.empty and complete_sites:
        parts.append(existing[existing["site_no"].isin(complete_sites)])
    if results:
        parts.append(pd.DataFrame(results))
    if not parts:
        log.error("No results produced.")
        return

    out_df = (
        pd.concat(parts, ignore_index=True)
        .sort_values("site_no")
        .reset_index(drop=True)
    )
    write_parquet_to_s3(out_df, bucket, out_key)
    n_tc  = out_df["tc_hr"].notna().sum()
    n_lc  = out_df["pct_u"].notna().sum()
    log.info(
        "Wrote basin_characteristics.parquet: %d stations, %d with Tc, %d with land cover",
        len(out_df), n_tc, n_lc,
    )


if __name__ == "__main__":
    main()
