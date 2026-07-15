"""visualize_lanesville_hydrograph.py

Hydrograph / hyetograph for the Lanesville IN flash flood — June 9, 2026.
Point: 38.23955°N, 85.98265°W.

Six panels (2×3 grid) — one per MRMS rolling accumulation window (3/6/12/24/48/72 h).
  Left y / bottom x : NWM analysis_assim streamflow (cfs) — solid blue line
                       + Q10 / Q50 / Q100 dashed hlines (Rao 2005 regression)
  Right y / top x   : MRMS QPE rolling accumulation (in) — inverted bars
                       + Atlas 14 P10/P50/P100/P1000 dotted hlines

Writes:
    s3://indot-bridge-pipeline/v1/events/lanesville_06_09_2026/
        lanesville_20260609_hydrograph.png
"""
from __future__ import annotations

import ast
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests
import s3fs
from pyproj import Geod
from shapely.geometry import shape as shapely_shape

sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_units, canonicalize_mrms_grid, decompress_gz, open_mrms_grib

# ── Configuration ─────────────────────────────────────────────────────────────
POINT_LAT     = 38.23955
POINT_LON     = -85.98265
EVENT_DATE    = date(2026, 6, 9)
DISPLAY_HOURS = list(range(6, 23))          # 06–22 UTC on event day
DURATIONS     = [3, 6, 12, 24]             # MRMS rolling windows (h)

MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_1H_FOLDER  = "MultiSensor_QPE_01H_Pass2_00.00"
NWM_BUCKET      = "noaa-nwm-pds"
PIPELINE_BUCKET = "indot-bridge-pipeline"
S3_EVENT_FOLDER = f"v1/events/lanesville_{EVENT_DATE:%m_%d_%Y}/"
PLOT_FNAME      = f"lanesville_{EVENT_DATE:%Y%m%d}_hydrograph.png"

OUT_DIR  = "results"
OUT_PLOT = os.path.join(OUT_DIR, PLOT_FNAME)
os.makedirs(OUT_DIR, exist_ok=True)

M3S_TO_CFS = 35.3147
KM2_TO_MI2 = 0.386102
EDT        = timedelta(hours=-4)

# Return-period visual styles (colour, label)
_RP_STYLE = {
    10:   ("green",      "10-yr"),
    50:   ("darkorange", "50-yr"),
    100:  ("red",        "100-yr"),
    1000: ("darkviolet", "1000-yr"),
}

# ── Rao (2005) regression coefficients ───────────────────────────────────────
_RC: dict[int, dict[int, tuple]] = {
    1: {10: (47.8,  0.802, 0.535, None), 50: (61.4,  0.805, 0.573, None), 100: (67.5,  0.805, 0.585, None)},
    2: {10: (69.6,  0.798, 0.473, None), 50: (133.1, 0.762, 0.417, None), 100: (169.5, 0.748, 0.394, None)},
    3: {10: (74.6,  0.889, 0.416, None), 50: (104.5, 0.894, 0.430, None), 100: (116.8, 0.898, 0.434, None)},
    4: {10: (31.1,  0.820, 0.681, 0.080), 50: (42.9, 0.819, 0.707, 0.077), 100: (48.4, 0.816, 0.712, 0.075)},
    5: {10: (35.8,  0.776, 0.368, None), 50: (53.1,  0.756, 0.347, None), 100: (60.8,  0.748, 0.338, None)},
    6: {10: (22.4,  0.732, 0.776, None), 50: (31.5,  0.696, 0.917, None), 100: (34.6,  0.687, 0.974, None)},
    7: {10: (65.0,  0.873, 0.372, -0.795), 50: (108.4, 0.849, 0.354, -0.803), 100: (129.3, 0.839, 0.347, -0.803)},
    8: {10: (106.0, 0.835, -0.733, None), 50: (126.5, 0.842, -0.707, None), 100: (134.2, 0.843, -0.695, None)},
}


def _assign_region(lat: float, lon: float) -> int:
    if lat > 41.0 and lon < -86.7: return 8
    if lat > 41.0:                  return 7
    if lat > 40.2 and lon > -85.8: return 1
    if lat > 39.7 and lon > -85.5: return 2
    if lon < -87.0 and lat > 38.5: return 3
    if lat > 39.5:                  return 4
    if lon < -86.2 and lat > 38.0: return 5
    return 6


def _apply_regression(region: int, rp: int, da: float, slope: float,
                      pct_u: float = 0.0, pct_w: float = 0.0) -> Optional[float]:
    c = _RC.get(region, {}).get(rp)
    if c is None:
        return None
    C, a1, a2, a3 = c
    if   region == 8: q = C * da**a1 * (pct_w + 1)**a2
    elif region == 7: q = C * da**a1 * slope**a2 * (pct_w + 1)**a3
    elif region == 4: q = C * da**a1 * slope**a2 * (pct_u + 1)**a3
    else:             q = C * da**a1 * slope**a2
    return round(q, 1) if q > 0 else None


# ── Basin characteristics for an arbitrary NHD COMID ─────────────────────────

_GEOD       = Geod(ellps="WGS84")
_M2_TO_MI2  = 2_589_988.1


def _get_nldi_basin_area_mi2(comid: str, timeout: int = 30) -> Optional[float]:
    """Compute watershed area (mi²) from the NLDI basin polygon for a COMID."""
    r = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/basin",
        timeout=timeout,
    )
    r.raise_for_status()
    total_m2 = 0.0
    for feat in r.json().get("features", []):
        geom = shapely_shape(feat["geometry"])
        total_m2 += abs(_GEOD.geometry_area_perimeter(geom)[0])
    return round(total_m2 / _M2_TO_MI2, 4) if total_m2 > 0 else None


def get_point_comid(lat: float, lon: float) -> str:
    r = requests.get(
        "https://api.water.usgs.gov/nldi/linked-data/comid/position",
        params={"coords": f"POINT({lon} {lat})"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["features"][0]["properties"]["identifier"]


def _haversine_mi(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((lat2 - lat1) * math.pi / 360) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin((lon2 - lon1) * math.pi / 360) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _elevation_ft(lon: float, lat: float) -> float:
    r = requests.get(
        "https://epqs.nationalmap.gov/v1/json",
        params={"x": lon, "y": lat, "units": "Feet", "wkid": 4326, "includeDate": "false"},
        timeout=20,
    )
    r.raise_for_status()
    return float(r.json()["value"])


def get_basin_chars(comid: str) -> dict:
    """Drainage area (mi²), 10-85 slope (ft/mi), pct_u, pct_w for a COMID."""
    out = {"drain_area_mi2": None, "slope_ft_mi": 1.0, "pct_u": 0.0, "pct_w": 0.0}

    # Drainage area: NLDI basin polygon (geodesic area, same method as 03b fallback)
    try:
        out["drain_area_mi2"] = _get_nldi_basin_area_mi2(comid)
        print(f"  DA from NLDI basin: {out['drain_area_mi2']:.2f} mi²")
    except Exception as e:
        print(f"  NLDI basin area failed: {e}")

    # StreamCat: land cover only (same field names / Ws-suffix logic as 03b)
    _URBAN = ["PctUrbLo2019", "PctUrbMd2019", "PctUrbHi2019"]
    _WATER = ["PctOw2019", "PctWdWet2019", "PctHbWet2019"]
    try:
        r = requests.get(
            "https://api.epa.gov/StreamCat/streams/metrics",
            params={
                "name": ",".join(_URBAN + _WATER),
                "comid": comid,
                "areaOfInterest": "watershed",
            },
            timeout=30,
        )
        r.raise_for_status()
        items = r.json().get("items") or r.json().get("Items") or []
        if items:
            row = items[0]
            out["pct_u"] = sum(row.get(f"{m.lower()}ws", 0.0) or 0.0 for m in _URBAN)
            out["pct_w"] = sum(row.get(f"{m.lower()}ws", 0.0) or 0.0 for m in _WATER)
    except Exception as e:
        print(f"  StreamCat land cover failed: {e}")

    # Channel geometry: NLDI upstream main stem + 3DEP elevation (same method as 03b)
    try:
        r = requests.get(
            f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
            "/navigation/UM/flowlines?distance=2000",
            timeout=30,
        )
        r.raise_for_status()
        coords: list = []
        for feat in r.json().get("features", []):
            coords.extend(feat["geometry"]["coordinates"])
        if len(coords) >= 2:
            cum = [0.0]
            for i in range(1, len(coords)):
                cum.append(cum[-1] + _haversine_mi(*coords[i - 1], *coords[i]))
            total_mi = cum[-1]
            if total_mi > 0:
                def _coord_at(frac: float):
                    t = frac * total_mi
                    for i in range(1, len(cum)):
                        if cum[i] >= t:
                            return coords[i]
                    return coords[-1]
                p10 = _coord_at(0.10)
                p85 = _coord_at(0.85)
                e10 = _elevation_ft(*p10)
                e85 = _elevation_ft(*p85)
                slope = abs(e85 - e10) / (0.75 * total_mi)
                out["slope_ft_mi"] = max(slope, 0.1)
                print(f"  Channel: {total_mi:.1f} mi, slope {out['slope_ft_mi']:.2f} ft/mi")
    except Exception as e:
        print(f"  Channel geometry failed: {e} — using 1.0 ft/mi fallback")

    return out


def compute_regression_q(lat: float, lon: float, basin: dict) -> dict[int, Optional[float]]:
    region = _assign_region(lat, lon)
    da    = basin.get("drain_area_mi2") or 0.0
    slope = basin.get("slope_ft_mi") or 1.0
    pct_u = basin.get("pct_u") or 0.0
    pct_w = basin.get("pct_w") or 0.0
    print(f"  Region {region}, DA={da:.1f} mi², slope={slope:.2f} ft/mi, "
          f"pct_u={pct_u:.1f}%, pct_w={pct_w:.1f}%")
    if da <= 0:
        print("  WARNING: no drainage area — regression Q will be None")
    return {rp: (_apply_regression(region, rp, da, slope, pct_u, pct_w) if da > 0 else None)
            for rp in [10, 50, 100]}


# ── Nearest gauged station (preferred Q source) ───────────────────────────────

def _get_gauge_comid(site_no: str, timeout: int = 30) -> Optional[int]:
    """Return NHD COMID for a USGS gauge via NLDI upstream flowlines (same as 03b)."""
    r = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/nwissite"
        f"/USGS-{site_no}/navigation/UM/flowlines?distance=10",
        timeout=timeout,
    )
    r.raise_for_status()
    features = r.json().get("features", [])
    if not features:
        return None
    props = features[0].get("properties", {})
    comid = (props.get("nhdplus_comid") or props.get("nhdpv2_COMID")
             or props.get("comid") or props.get("COMID"))
    return int(comid) if comid else None


def get_nearest_gauge_q(lat: float, lon: float) -> dict:
    """Find the nearest Indiana gauge with Q10/Q50/Q100 in the pipeline S3 parquet.

    Returns dict: {site_no, dist_mi, comid, Q10, Q50, Q100} — Q values in cfs,
    comid is the NHD COMID to use for NWM.  All values are None on failure.
    """
    import io as _io
    import pandas as _pd
    import pyarrow.parquet as _pq

    out: dict = {"site_no": None, "dist_mi": None, "comid": None,
                 "Q10": None, "Q50": None, "Q100": None}
    try:
        s3c = boto3.client("s3", region_name="us-east-1")

        inv = _pq.read_table(
            _io.BytesIO(
                s3c.get_object(
                    Bucket=PIPELINE_BUCKET,
                    Key="v1/stations/indiana_streamflow_sites.parquet",
                )["Body"].read()
            )
        ).to_pandas()
        inv["site_no"] = inv["site_no"].astype(str)

        fstats = _pq.read_table(
            _io.BytesIO(
                s3c.get_object(
                    Bucket=PIPELINE_BUCKET,
                    Key="v1/flow_stats/per_gauge_flow_stats.parquet",
                )["Body"].read()
            )
        ).to_pandas()
        fstats["site_no"] = fstats["site_no"].astype(str)

        merged = (
            inv[["site_no", "dec_lat_va", "dec_long_va"]]
            .merge(fstats[["site_no", "Q10", "Q50", "Q100"]], on="site_no", how="inner")
            .dropna(subset=["Q10", "Q100", "dec_lat_va", "dec_long_va"])
        )
        if merged.empty:
            print("  WARNING: no gauges with Q10/Q100 found in pipeline S3")
            return out

        merged = merged.copy()
        merged["_dist"] = merged.apply(
            lambda r: _haversine_mi(
                lon, lat, float(r["dec_long_va"]), float(r["dec_lat_va"])
            ),
            axis=1,
        )
        best = merged.loc[merged["_dist"].idxmin()]

        site_no        = str(best["site_no"])
        out["site_no"] = site_no
        out["dist_mi"] = float(best["_dist"])
        out["Q10"]     = float(best["Q10"])
        out["Q50"]     = float(best["Q50"]) if _pd.notna(best.get("Q50")) else None
        out["Q100"]    = float(best["Q100"])

        q50_str = f"{out['Q50']:,.0f}" if out["Q50"] is not None else "None"
        print(f"  Nearest gauge: USGS {site_no} ({out['dist_mi']:.1f} mi away)")
        print(f"  Q10={out['Q10']:,.0f} cfs  Q50={q50_str} cfs  Q100={out['Q100']:,.0f} cfs")

        out["comid"] = _get_gauge_comid(site_no)
        print(f"  NHD COMID for NWM: {out['comid']}")

    except Exception as e:
        print(f"  WARNING: gauge Q lookup failed: {e}")

    return out


# ── Atlas 14 ──────────────────────────────────────────────────────────────────

def _extract_js_array(text: str, name: str) -> list:
    """Parse a `name=[...]` JavaScript array literal using bracket counting.

    The lazy-regex approach fails on nested arrays (matches the first inner `]`
    instead of the outer one). Bracket counting is unambiguous.
    """
    m = re.search(rf'{re.escape(name)}\s*=\s*\[', text)
    if not m:
        raise ValueError(f"'{name}' not found in response")
    start = m.end() - 1        # index of the opening '['
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return ast.literal_eval(text[start:i + 1])
    raise ValueError(f"Unmatched bracket for '{name}'")


def fetch_atlas14(lat: float, lon: float) -> dict[int, dict[int, Optional[float]]]:
    """Return {duration_h: {rp: depth_in}} from NOAA Atlas 14 PDS series.

    Durations: 3, 6, 12, 24, 48, 72 h  |  RPs: 10, 50, 100, 1000 yr
    Returns empty (None values) on any failure so the rest of the plot renders.
    """
    _empty = {d: {rp: None for rp in [10, 50, 100, 1000]} for d in DURATIONS}

    try:
        r = requests.get(
            "https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py",
            params={"type": "pf", "units": "us", "series": "pds",
                    "lat": lat, "lon": lon},
            timeout=30,
        )
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"  WARNING: Atlas 14 HTTP request failed: {e}")
        return _empty

    try:
        quantiles = _extract_js_array(text, "quantiles")
    except ValueError:
        print(f"  WARNING: Atlas 14 parse failed — response preview:\n  {text[:400]!r}")
        return _empty

    # PDS duration column indices (0-based):
    # 5m=0, 10m=1, 15m=2, 30m=3, 60m=4, 2h=5, 3h=6, 6h=7, 12h=8,
    # 24h=9, 2d=10, 3d=11, 4d=12, 7d=13, …
    dur_col = {3: 6, 6: 7, 12: 8, 24: 9, 48: 10, 72: 11}
    # RP row indices (0-based): 1yr=0, 2yr=1, 5yr=2, 10yr=3, 25yr=4,
    # 50yr=5, 100yr=6, 200yr=7, 500yr=8, 1000yr=9
    rp_row  = {10: 3, 50: 5, 100: 6, 1000: 9}

    result: dict[int, dict[int, Optional[float]]] = {}
    for dur_h, ci in dur_col.items():
        result[dur_h] = {}
        for rp, ri in rp_row.items():
            try:
                val = float(quantiles[ri][ci])
                result[dur_h][rp] = val if val > 0 else None
            except (IndexError, TypeError, ValueError):
                result[dur_h][rp] = None
    return result


# ── MRMS point time series ────────────────────────────────────────────────────

_MRMS_IDX: Optional[tuple[int, int]] = None   # (lat_idx, lon_idx) cached after first hit


def load_mrms_1h_at_point(fs: s3fs.S3FileSystem,
                           dt: datetime) -> Optional[float]:
    global _MRMS_IDX
    ymd   = dt.strftime("%Y%m%d")
    ts    = dt.strftime("%Y%m%d-%H0000")
    fname = f"MRMS_{MRMS_1H_FOLDER}_{ts}.grib2.gz"
    key   = f"{MRMS_BUCKET}/CONUS/{MRMS_1H_FOLDER}/{ymd}/{fname}"
    try:
        with fs.open(key, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"\n  MRMS {dt:%Y-%m-%d %H}z: fetch error: {e}")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(decompress_gz(raw))
            tmp_path = tmp.name
        ds  = open_mrms_grib(tmp_path)
        var = list(ds.data_vars)[0]
        arr, lats, lons = canonicalize_mrms_grid(ds[var])
        arr = apply_units(arr, kind="qpe", units="in")
        os.unlink(tmp_path)
        if _MRMS_IDX is None:
            lat_i = int(np.argmin(np.abs(lats - POINT_LAT)))
            lon_i = int(np.argmin(np.abs(lons - POINT_LON)))
            _MRMS_IDX = (lat_i, lon_i)
            print(f"\n  Grid snap: lat={lats[lat_i]:.4f}, lon={lons[lon_i]:.4f}")
        lat_i, lon_i = _MRMS_IDX
        val = float(arr[lat_i, lon_i])
        return max(0.0, val) if not np.isnan(val) else 0.0
    except Exception as e:
        print(f"\n  MRMS {dt:%Y-%m-%d %H}z: parse error: {e}")
        return None


def fetch_mrms_timeseries(fs: s3fs.S3FileSystem,
                          all_dts: list[datetime]) -> dict[datetime, float]:
    hourly: dict[datetime, float] = {}

    def _one(dt: datetime):
        return dt, load_mrms_1h_at_point(fs, dt)

    print(f"  Fetching {len(all_dts)} MRMS files "
          f"({all_dts[0]:%Y-%m-%d %H}z → {all_dts[-1]:%Y-%m-%d %H}z)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_one, dt): dt for dt in all_dts}
        done = 0
        for fut in as_completed(futs):
            dt, val = fut.result()
            hourly[dt] = val if val is not None else 0.0
            done += 1
            if done % 20 == 0 or done == len(all_dts):
                print(f"  [{done}/{len(all_dts)}]", end="\r")
    print(f"\n  Non-zero hours: {sum(v > 0 for v in hourly.values())}")
    return hourly


def compute_rolling(hourly: dict[datetime, float],
                    display_dts: list[datetime]) -> dict[int, list[float]]:
    """Sum the N hours ending at each display datetime (inclusive)."""
    result: dict[int, list[float]] = {d: [] for d in DURATIONS}
    for dt in display_dts:
        for dur_h in DURATIONS:
            total = sum(hourly.get(dt - timedelta(hours=i), 0.0)
                        for i in range(dur_h))
            result[dur_h].append(total)
    return result


# ── NWM single-COMID time series ──────────────────────────────────────────────

_NWM_IDX: Optional[int] = None   # cached index of the COMID in feature_id


def load_nwm_at_comid(fs: s3fs.S3FileSystem,
                       dt: datetime,
                       comid: int) -> Optional[float]:
    global _NWM_IDX
    key = (f"{NWM_BUCKET}/nwm.{dt:%Y%m%d}/analysis_assim/"
           f"nwm.t{dt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc")
    try:
        with fs.open(key, "rb") as fobj:
            with h5py.File(fobj, "r") as hf:
                if _NWM_IDX is None:
                    all_ids = hf["feature_id"][:].astype(np.int64)
                    matches = np.where(all_ids == comid)[0]
                    if len(matches) == 0:
                        print(f"\n  COMID {comid} not found in NWM feature_id array")
                        return None
                    _NWM_IDX = int(matches[0])
                    print(f"\n  NWM: COMID {comid} at array index {_NWM_IDX}")
                # NWM channel_rt streamflow is a packed int (scale_factor ~0.01);
                # h5py returns raw ints, so unpack by hand (xarray would auto-apply)
                # or the flow comes out ~100x too big. attrs are 1-element arrays.
                dset = hf["streamflow"]
                sf = float(np.asarray(dset.attrs.get("scale_factor", 1.0)).ravel()[0])
                ao = float(np.asarray(dset.attrs.get("add_offset", 0.0)).ravel()[0])
                q = float(dset[_NWM_IDX]) * sf + ao
                return q if q >= 0 else None   # m³/s — matches regression units after conversion
    except Exception as e:
        print(f"\n  NWM {dt:%H}z: {e}")
        return None


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(
    display_dts: list[datetime],
    q_m3s: list[Optional[float]],
    precip_rolling: dict[int, list[float]],
    atlas14: dict[int, dict[int, Optional[float]]],
) -> None:
    edt_dts = [dt + EDT for dt in display_dts]
    bar_w   = timedelta(minutes=40)

    # Dense x-coords for Atlas 14 marker-X lines (needed so markevery spaces evenly)
    n_pts   = 40
    span    = edt_dts[-1] - edt_dts[0]
    mark_xs = [edt_dts[0] + span * i / (n_pts - 1) for i in range(n_pts)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Lanesville, IN Flash Flood — {EVENT_DATE:%B %-d, %Y}\n"
        f"NWM streamflow vs MRMS precipitation  |  "
        f"{POINT_LAT:.4f}°N, {abs(POINT_LON):.4f}°W",
        fontsize=12, fontweight="bold", y=0.99,
    )

    q_vals_plot = [v if v is not None else np.nan for v in q_m3s]
    q_max = np.nanmax(q_vals_plot) if not all(np.isnan(q_vals_plot)) else 1.0

    for panel_idx, (ax1, dur_h) in enumerate(zip(axes.flat, DURATIONS)):
        ax2 = ax1.twinx()
        col = panel_idx % 2
        row = panel_idx // 2

        # ── Streamflow (left y / bottom x) ────────────────────────────────
        ax1.plot(edt_dts, q_vals_plot, color="royalblue", lw=2.2,
                 zorder=3, label="NWM Q (m³/s)")

        # ── Precipitation (right y / top x, inverted) ──────────────────────
        prec_vals = precip_rolling[dur_h]
        ax2.bar(edt_dts, prec_vals, width=bar_w,
                color="steelblue", alpha=0.45, zorder=1)

        for rp in [10, 50, 100, 1000]:
            p_ref = atlas14.get(dur_h, {}).get(rp)
            if p_ref is not None:
                color = _RP_STYLE[rp][0]
                # --x--x-- style: dense line + evenly-spaced X markers
                ax2.plot(mark_xs, [p_ref] * n_pts,
                         color=color, lw=1.0, ls="--",
                         marker="x", markevery=6, markersize=6, zorder=2,
                         label=f"P{rp} = {p_ref:.2f} in")

        ax2.invert_yaxis()

        # ── Top x-axis (precipitation time labels) ──────────────────────
        ax_top = ax1.secondary_xaxis("top")
        ax_top.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
        plt.setp(ax_top.get_xticklabels(), fontsize=7, rotation=45, ha="left")

        # ── Bottom x-axis (streamflow time labels) ────────────────────────
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
        plt.setp(ax1.get_xticklabels(), fontsize=7, rotation=45, ha="right")

        # ── Axis labels / title ────────────────────────────────────────────
        ax1.set_title(f"{dur_h}h accumulation", fontsize=10)
        ax1.set_xlim(edt_dts[0] - timedelta(minutes=30),
                     edt_dts[-1] + timedelta(minutes=30))

        if col == 0:
            ax1.set_ylabel("Streamflow (m³/s)", color="royalblue", fontsize=9)
        if col == 1:
            ax2.set_ylabel("Precipitation (in)", color="steelblue", fontsize=9)
        if row == 1:
            ax1.set_xlabel("Time (EDT)", fontsize=8)

        ax1.tick_params(axis="y", colors="royalblue", labelsize=8)
        ax2.tick_params(axis="y", colors="steelblue", labelsize=8)

        ax1.set_ylim(bottom=0, top=max(q_max * 1.2, 0.1))

        p_max = max((v for v in prec_vals if v is not None), default=0.01)
        p_ref_max = max(
            (atlas14.get(dur_h, {}).get(rp) or 0.0 for rp in [10, 50, 100, 1000]),
            default=0.0,
        )
        ax2.set_ylim(top=0, bottom=max(p_max, p_ref_max) * 1.3)

        # ── Legend ────────────────────────────────────────────────────────
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left",
                   framealpha=0.75, ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved locally: {OUT_PLOT}")


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str) -> str:
    key = S3_EVENT_FOLDER + PLOT_FNAME
    s3  = boto3.client("s3", region_name="us-east-1")
    print(f"Uploading to s3://{PIPELINE_BUCKET}/{key} ...")
    s3.upload_file(local_path, PIPELINE_BUCKET, key,
                   ExtraArgs={"ContentType": "image/png"})
    uri = f"s3://{PIPELINE_BUCKET}/{key}"
    print(f"Saved: {uri}")
    return uri


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nPoint: {POINT_LAT}°N, {POINT_LON}°E")

    # 1. COMID
    print("\n── Finding COMID ────────────────────────────────────────────────────")
    comid_str = get_point_comid(POINT_LAT, POINT_LON)
    comid_int = int(comid_str)
    print(f"  COMID: {comid_int}")

    # 2. Atlas 14
    print("\n── Atlas 14 ─────────────────────────────────────────────────────────")
    atlas14 = fetch_atlas14(POINT_LAT, POINT_LON)
    for dur_h in DURATIONS:
        row = atlas14.get(dur_h, {})
        parts = ", ".join(f"P{rp}={v:.2f}\"" for rp, v in row.items() if v)
        print(f"  {dur_h:2d}h: {parts}")

    # 4. MRMS wide time-series (need up to 72h before first display hour)
    print("\n── MRMS 1h QPE ──────────────────────────────────────────────────────")
    fs = s3fs.S3FileSystem(anon=True)
    start_dt = (datetime(EVENT_DATE.year, EVENT_DATE.month, EVENT_DATE.day,
                         DISPLAY_HOURS[0])
                - timedelta(hours=max(DURATIONS)))
    end_dt   = datetime(EVENT_DATE.year, EVENT_DATE.month, EVENT_DATE.day,
                        DISPLAY_HOURS[-1])
    all_dts: list[datetime] = []
    cur = start_dt
    while cur <= end_dt:
        all_dts.append(cur)
        cur += timedelta(hours=1)

    hourly_precip = fetch_mrms_timeseries(fs, all_dts)

    # 5. Rolling accumulations at display hours
    display_dts = [datetime(EVENT_DATE.year, EVENT_DATE.month, EVENT_DATE.day, h)
                   for h in DISPLAY_HOURS]
    precip_rolling = compute_rolling(hourly_precip, display_dts)

    # 6. NWM streamflow (m³/s — no cfs conversion; regression Q converted at plot time)
    print("\n── NWM streamflow ───────────────────────────────────────────────────")
    q_m3s: list[Optional[float]] = []
    for dt in display_dts:
        print(f"  {dt:%H}z...", end="\r")
        q_m3s.append(load_nwm_at_comid(fs, dt, comid_int))
    n_ok = sum(q is not None for q in q_m3s)
    print(f"\n  {n_ok}/{len(display_dts)} hours loaded, "
          f"peak={max((q for q in q_m3s if q), default=0):.1f} m³/s")

    # 7. Plot + upload
    print("\n── Creating figure ──────────────────────────────────────────────────")
    make_figure(display_dts, q_m3s, precip_rolling, atlas14)
    upload_to_s3(OUT_PLOT)
    print("\nDone.")


if __name__ == "__main__":
    main()
