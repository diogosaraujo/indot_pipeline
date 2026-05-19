"""13_download_usgs_precip.py

Download instantaneous precipitation (parameter code 00045, in) from the
USGS Water Data API for gauges within the watershed union.

USGS IV precipitation data has approximately 120 days of online retention;
historical IV precipitation is not archived for most sites.  Each run
downloads the window [max(existing_record, now-120d), now] per station.

Station discovery:
    Builds the watershed union from S3 GeoJSONs (Option A), derives its
    bounding box, and queries the USGS NWIS site service for parameterCd=00045
    with a single bBox request that automatically covers all states in the
    watershed (IN, IL, OH, MI, KY, NY, PA, VA, WV, NC, etc.).  Results are
    then filtered to the exact watershed polygon.  Falls back to the Indiana
    bounding box if no watershed GeoJSONs are found in S3.

Data download:
    Uses dataretrieval.waterdata.get_continuous() — the same Water Data v3
    API endpoint used by scripts 01 and 02.

Gap-filling:
    If the parquet already exists in S3, only rows with datetime_utc >
    max(existing) per station are appended.

Reads:
    s3://<bucket>/<prefix>watersheds/per_gauge/*.geojson   (polygon filter)
    s3://<bucket>/<prefix>precip/usgs/precip_iv.parquet    (optional, gap-fill)

Writes:
    s3://<bucket>/<prefix>precip/usgs/stations.parquet
    s3://<bucket>/<prefix>precip/usgs/stations.geojson
    s3://<bucket>/<prefix>precip/usgs/precip_iv.parquet

Schema (precip_iv.parquet):
    site_no         str
    station_nm      str
    latitude        float64
    longitude       float64
    datetime_utc    datetime64[ns, UTC]
    precip_in       float64
"""
from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import botocore.exceptions
import pandas as pd
import pyarrow.parquet as pq
import requests
from dataretrieval import waterdata

from utils import (
    RetryPolicy,
    build_watershed_union,
    filter_by_polygon,
    load_config,
    s3_client,
    with_retries,
    write_bytes_to_s3,
    write_parquet_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("13_usgs_precip")

PRECIP_PARAM    = "00045"
# USGS IV archive retention window
IV_RETENTION_DAYS = 120
# Indiana bounding box fallback (minlat, minlon, maxlat, maxlon)
INDIANA_BBOX    = (37.77, -88.10, 41.76, -84.78)

NWIS_SITE_URL   = "https://waterservices.usgs.gov/nwis/site/"


# ── Station inventory ─────────────────────────────────────────────────────────

def fetch_usgs_precip_stations(
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Query NWIS site service for IV precipitation gauges within a bounding box.

    bbox = (west_lon, south_lat, east_lon, north_lat) — matches shapely polygon.bounds.
    A single bBox request automatically covers all states whose territory falls
    inside the watershed union (IN, IL, OH, MI, KY, NY, PA, VA, WV, NC, etc.)
    without requiring a hardcoded state list.
    """
    west, south, east, north = bbox
    params = {
        "format":           "rdb",
        "bBox":             f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}",
        "parameterCd":      PRECIP_PARAM,
        "hasDataTypeCd":    "iv",
        "outputDataTypeCd": "iv",
    }
    try:
        r = requests.get(NWIS_SITE_URL, params=params, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("NWIS bBox site query failed: %s", e)
        return pd.DataFrame(columns=["site_no", "station_nm", "latitude", "longitude"])

    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    # lines[0] = header, lines[1] = type row, lines[2+] = data
    if len(lines) < 3:
        log.info("No precipitation sites found in bbox.")
        return pd.DataFrame(columns=["site_no", "station_nm", "latitude", "longitude"])

    header = lines[0].split("\t")
    records: list[dict] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        records.append(dict(zip(header, parts)))

    if not records:
        return pd.DataFrame(columns=["site_no", "station_nm", "latitude", "longitude"])

    df = pd.DataFrame(records)
    # NWIS RDB columns include dec_lat_va, dec_long_va, station_nm, site_no
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "long" in c.lower() or "lon" in c.lower()), None)
    nm_col  = next((c for c in df.columns if "station_nm" in c.lower()), None)

    if lat_col is None or lon_col is None:
        log.warning("Could not locate lat/lon columns in NWIS RDB output: %s", list(df.columns)[:10])
        return pd.DataFrame(columns=["site_no", "station_nm", "latitude", "longitude"])

    out = pd.DataFrame({
        "site_no":    df["site_no"].astype(str).str.strip(),
        "station_nm": df[nm_col].astype(str).str.strip() if nm_col else "",
        "latitude":   pd.to_numeric(df[lat_col],  errors="coerce"),
        "longitude":  pd.to_numeric(df[lon_col], errors="coerce"),
    }).dropna(subset=["latitude", "longitude"])
    out = out.drop_duplicates(subset=["site_no"]).reset_index(drop=True)
    log.info(
        "NWIS precip inventory: %d unique sites in bbox "
        "(S=%.3f W=%.3f → N=%.3f E=%.3f)",
        len(out), south, west, north, east,
    )
    return out


# ── GeoJSON helpers ───────────────────────────────────────────────────────────

def stations_to_geojson(df: pd.DataFrame) -> bytes:
    """Serialize a stations DataFrame to a GeoJSON FeatureCollection (UTF-8 bytes)."""
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "site_no":    row.get("site_no", ""),
                "station_nm": row.get("station_nm", ""),
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc).encode("utf-8")


# ── Existing-parquet helpers ──────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def get_max_dates(df: Optional[pd.DataFrame], id_col: str = "site_no") -> dict:
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.groupby(id_col)["datetime_utc"].max().to_dict()


# ── Per-station data download ─────────────────────────────────────────────────

def _patched_session(pat: Optional[str]):
    """Monkey-patch requests.Session to inject the USGS API token if set."""
    if not pat:
        return
    import requests as _req
    _orig = _req.Session.send

    def _send(self, r, **kw):
        r.headers.setdefault("Authorization", f"Bearer {pat}")
        kw.setdefault("timeout", 120)
        return _orig(self, r, **kw)

    _req.Session.send = _send


def fetch_precip_iv(
    site_no: str, start_str: str, end_str: str
) -> Optional[pd.DataFrame]:
    monitoring_location_id = f"USGS-{site_no}" if not site_no.startswith("USGS-") else site_no
    time_string = f"{start_str}/{end_str}"

    def _call():
        df, _ = waterdata.get_continuous(
            monitoring_location_id=monitoring_location_id,
            parameter_code=PRECIP_PARAM,
            time=time_string,
        )
        return df

    df = with_retries(_call, RetryPolicy(max_attempts=4, base_delay=3.0))
    if df is None or df.empty:
        return None
    if "time" not in df.columns or "value" not in df.columns:
        return None

    out = pd.DataFrame({
        "site_no":      site_no,
        "datetime_utc": pd.to_datetime(df["time"], utc=True, errors="coerce"),
        "precip_in":    pd.to_numeric(df["value"], errors="coerce").astype("float64"),
    })
    return out.dropna(subset=["datetime_utc"]).reset_index(drop=True)


def process_station(
    row: pd.Series,
    max_dates: dict,
    now_utc: pd.Timestamp,
    retention_start: pd.Timestamp,
) -> tuple[str, Optional[pd.DataFrame]]:
    site_no = str(row["site_no"])
    last_known = max_dates.get(site_no)

    if last_known is not None:
        eff_start = pd.Timestamp(last_known) + pd.Timedelta(hours=1)
    else:
        eff_start = retention_start

    if eff_start >= now_utc:
        return site_no, None  # already up-to-date

    start_str = eff_start.strftime("%Y-%m-%dT%H:%M+00:00")
    end_str   = now_utc.strftime("%Y-%m-%dT%H:%M+00:00")

    df = fetch_precip_iv(site_no, start_str, end_str)
    return site_no, df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    bucket      = cfg["aws"]["output_bucket"]
    prefix      = cfg["aws"]["output_prefix"]
    max_workers = cfg.get("execution", {}).get("max_workers_io", 8)

    now_utc          = pd.Timestamp(datetime.utcnow(), tz="UTC")
    retention_start  = now_utc - pd.Timedelta(days=IV_RETENTION_DAYS)

    pat = os.getenv("API_USGS_PAT")
    _patched_session(pat)
    if pat:
        log.info("Using USGS API token from API_USGS_PAT")
    else:
        log.warning("API_USGS_PAT not set; requests may be more rate-limited")

    # ── Watershed polygon ──────────────────────────────────────────────────
    log.info("Building watershed union from S3...")
    polygon = build_watershed_union(bucket, prefix)
    if polygon is not None:
        log.info("Watershed union loaded.")
    else:
        log.info("No watershed files — using Indiana bounding box.")

    # ── Station inventory ──────────────────────────────────────────────────
    log.info("Fetching USGS precipitation site inventory...")
    if polygon is not None:
        # Derive bbox from the actual watershed union — automatically covers all
        # states in the polygon (IN, IL, OH, MI, KY, NY, PA, VA, WV, NC, etc.)
        west, south, east, north = polygon.bounds
        query_bbox = (west, south, east, north)
    else:
        # Indiana fallback: convert (minlat, minlon, maxlat, maxlon) → (W, S, E, N)
        query_bbox = (INDIANA_BBOX[1], INDIANA_BBOX[0], INDIANA_BBOX[3], INDIANA_BBOX[2])
    all_stations = fetch_usgs_precip_stations(query_bbox)
    stations = filter_by_polygon(all_stations, polygon, fallback_bbox=INDIANA_BBOX)
    log.info("Precipitation sites in watershed: %d / %d", len(stations), len(all_stations))

    if stations.empty:
        log.warning("No USGS precipitation sites found — exiting.")
        return

    write_parquet_to_s3(stations, bucket, f"{prefix}precip/usgs/stations.parquet")
    log.info("Wrote station inventory: s3://%s/%sprecip/usgs/stations.parquet", bucket, prefix)

    geojson_key = f"{prefix}precip/usgs/stations.geojson"
    write_bytes_to_s3(stations_to_geojson(stations), bucket, geojson_key)
    log.info("Wrote station GeoJSON:   s3://%s/%s", bucket, geojson_key)

    # ── Data download (concurrent) ─────────────────────────────────────────
    existing = _read_parquet_s3(bucket, f"{prefix}precip/usgs/precip_iv.parquet")
    max_dates = get_max_dates(existing)

    if existing is not None:
        log.info(
            "Existing parquet: %d rows, %d stations. Gap-filling from per-station max.",
            len(existing), existing["site_no"].nunique(),
        )
    else:
        log.info(
            "No existing parquet — downloading last %d days for all stations.",
            IV_RETENTION_DAYS,
        )

    new_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_station, row, max_dates, now_utc, retention_start): row["site_no"]
            for _, row in stations.iterrows()
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            site_no = futures[fut]
            try:
                sid, df = fut.result()
                if df is not None and not df.empty:
                    new_frames.append(df)
                    log.info("[%d/%d] %s: %d new rows", i, total, sid, len(df))
                else:
                    log.info("[%d/%d] %s: no new data", i, total, sid)
            except Exception as e:
                log.warning("[%d/%d] %s: %s", i, total, site_no, e)

    if not new_frames:
        log.info("No new rows — parquet unchanged.")
        return

    new_df = pd.concat(new_frames, ignore_index=True)

    # Attach station metadata (name/lat/lon) to new rows
    meta_cols = ["site_no", "station_nm", "latitude", "longitude"]
    meta_cols = [c for c in meta_cols if c in stations.columns]
    new_df = new_df.merge(stations[meta_cols], on="site_no", how="left")

    # Merge with existing and dedup
    parts = [p for p in [existing, new_df] if p is not None and not p.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    combined = (
        combined
        .drop_duplicates(subset=["site_no", "datetime_utc"], keep="last")
        .sort_values(["site_no", "datetime_utc"])
        .reset_index(drop=True)
    )

    write_parquet_to_s3(combined, bucket, f"{prefix}precip/usgs/precip_iv.parquet")
    log.info(
        "Wrote precip_iv.parquet: %d rows for %d stations.",
        len(combined), combined["site_no"].nunique(),
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
