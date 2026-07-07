"""visualize_lanesville_event.py

3-panel animated GIF for the Lanesville, IN flash flood — June 9, 2026.

  Panel 1: MRMS QPE 1-hour (Pass2) — hourly rainfall depth (nearest precip station starred)
  Panel 2: Hyetograph of the nearest USGS precip station to the map centre, with a
           marker that advances hour-by-hour in sync with panels 1 and 3, plus the
           Atlas-14 return period (ARI) of the station's peak accumulation for the event
  Panel 3: NWM analysis streamflow on NHD flowlines

Note: the nearest-station hyetograph uses NOAA GHCNh hourly precip
(precip/noaa/ghcnh_hourly.parquet, already downloaded by script 12).  GHCNh is in
MILLIMETRES and its sub-hourly reports are running 1-hour totals, so the hourly value
is the per-hour MAX ÷ 25.4 (the pipeline convention, cf. 08d).

Before running (if not already installed):
    pip install contextily pillow

Reads (public S3, no credentials needed):
    noaa-mrms-pds  —  MultiSensor_QPE_01H_Pass2_00.00
    noaa-nwm-pds   —  analysis_assim channel_rt

Writes:
    results/lanesville_20260609.gif
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import boto3
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import requests
import s3fs
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata

# Pipeline utilities (script lives alongside utils.py)
sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_units, canonicalize_mrms_grid, decompress_gz, open_mrms_grib

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("WARNING: contextily not installed — basemap tiles will be omitted.")
    print("         Install with:  pip install contextily")

# ── Configuration ──────────────────────────────────────────────────────────────
EVENT_DATE   = date(2026, 6, 9)
LON_MIN, LON_MAX = -86.06, -85.88
LAT_MIN, LAT_MAX = 38.15, 38.31
HOURS        = list(range(6, 23))   # 06–22 UTC (EDT = UTC-4, so 02:00–18:00 local)

MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_1H_FOLDER  = "MultiSensor_QPE_01H_Pass2_00.00"
NWM_BUCKET      = "noaa-nwm-pds"

# Pipeline S3 bucket (output destination)
PIPELINE_BUCKET  = "indot-bridge-pipeline"
PIPELINE_PREFIX  = "v1/"
S3_EVENT_FOLDER  = f"v1/events/lanesville_{EVENT_DATE:%m_%d_%Y}/"
GIF_FNAME        = f"lanesville_{EVENT_DATE:%Y%m%d}.gif"

OUT_DIR      = "results"
OUT_GIF      = os.path.join(OUT_DIR, GIF_FNAME)
FRAME_MS     = 600      # ms per frame in the GIF
ALPHA_PRECIP = 0.55     # transparency of the MRMS overlay

# Precipitation colour scale (inches)
QPE_1H_VMAX  = 2.0
QPE_3H_VMAX  = 5.0

# NWM streamflow colour scale (m³/s, log)
NWM_Q_VMIN   = 0.5
NWM_Q_VMAX   = 500.0

# ── Precip station hyetograph (panel 2) + event ARI ─────────────────────────────
CENTER_LON = (LON_MIN + LON_MAX) / 2.0
CENTER_LAT = (LAT_MIN + LAT_MAX) / 2.0
GHCNH_KEY         = "precip/noaa/ghcnh_hourly.parquet"  # NOAA GHCNh hourly precip (millimetres)
GHCNH_MM_TO_IN    = 25.4                                # GHCNh precip is stored in mm
ATLAS14_KEY       = "atlas14/precipitation_frequency.parquet"
INV_KEY           = "stations/indiana_streamflow_sites.parquet"
ARI_DURATIONS     = [1, 3, 6, 12, 24]                   # hours checked for the event ARI
MAX_HOURLY_IN     = 12.0                                # drop implausible hourly totals / sentinels
HYETO_COLOR       = "#1f78b4"

os.makedirs(OUT_DIR, exist_ok=True)


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _anon_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True)


def upload_gif_to_s3(local_path: str) -> str:
    """Upload the rendered GIF to the pipeline S3 bucket; return the s3:// URI."""
    key = S3_EVENT_FOLDER + GIF_FNAME
    s3 = boto3.client("s3", region_name="us-east-1")
    print(f"Uploading to s3://{PIPELINE_BUCKET}/{key} ...")
    s3.upload_file(local_path, PIPELINE_BUCKET, key,
                   ExtraArgs={"ContentType": "image/gif"})
    uri = f"s3://{PIPELINE_BUCKET}/{key}"
    print(f"Saved to {uri}")
    return uri


def _read_pipeline_parquet(key: str) -> pd.DataFrame:
    """Read a parquet from the pipeline bucket (uses IAM role credentials)."""
    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket=PIPELINE_BUCKET, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def _read_pipeline_filtered(key: str, columns=None, filters=None) -> pd.DataFrame:
    """Predicate-pushdown read from the pipeline bucket (avoids pulling the whole file)."""
    path = f"{PIPELINE_BUCKET}/{PIPELINE_PREFIX}{key}"
    return pq.read_table(path, filesystem=pafs.S3FileSystem(),
                         columns=columns, filters=filters).to_pandas()


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ── Nearest USGS precip station → hyetograph (panel 2) ──────────────────────────

def load_nearest_precip_station() -> tuple[Optional[dict], Optional[pd.Series]]:
    """Nearest GHCNh precip station to the map centre and its HOURLY hyetograph (in)
    for the event day.  GHCNh sub-hourly reports are running 1-hour totals in
    MILLIMETRES → hourly value = per-hour MAX ÷ 25.4 (pipeline convention, cf. 08d);
    hours with no report are treated as dry.  Returns (info, hourly) or (None, None)
    if no GHCNh station near the centre covers the event day."""
    print(f"\n── Nearest GHCNh precip station to map centre ({CENTER_LAT:.3f}, {CENTER_LON:.3f}) ──")
    day0 = pd.Timestamp(EVENT_DATE, tz="UTC")
    day1 = day0 + pd.Timedelta(days=1)
    try:
        df = _read_pipeline_filtered(
            GHCNH_KEY,
            columns=["station_id", "datetime_utc", "precip_in", "name", "latitude", "longitude"],
            filters=[("datetime_utc", ">=", day0.to_pydatetime()),
                     ("datetime_utc", "<",  day1.to_pydatetime()),
                     ("latitude",  ">=", CENTER_LAT - 1.0), ("latitude",  "<=", CENTER_LAT + 1.0),
                     ("longitude", ">=", CENTER_LON - 1.0), ("longitude", "<=", CENTER_LON + 1.0)],
        )
    except Exception as e:
        print(f"  GHCNh read failed: {e}")
        return None, None
    if df.empty:
        print(f"  No GHCNh precip near the map centre on {EVENT_DATE}.")
        return None, None

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["station_id"]   = df["station_id"].astype(str)
    df["precip_in"]    = pd.to_numeric(df["precip_in"], errors="coerce") / GHCNH_MM_TO_IN   # mm → in
    meta = df.groupby("station_id").agg(lat=("latitude", "first"),
                                        lon=("longitude", "first"),
                                        nm=("name", "first")).reset_index()
    meta["dist_km"] = [_haversine_km(CENTER_LAT, CENTER_LON, r.lat, r.lon) for r in meta.itertuples()]
    nearest = meta.sort_values("dist_km").iloc[0]

    s = df.loc[df["station_id"] == nearest.station_id].set_index("datetime_utc")["precip_in"].astype(float)
    s = s[(s >= 0) & (s <= MAX_HOURLY_IN)].sort_index()
    # Per-hour MAX (running-1h-total convention); label = bin END to match MRMS QPE_01H.
    hourly = s.resample("1h", label="right", closed="right").max().fillna(0.0)
    info = {"site_no": nearest.station_id, "name": str(nearest.nm),
            "lat": float(nearest.lat), "lon": float(nearest.lon), "dist_km": float(nearest.dist_km)}
    print(f"  Nearest: {info['site_no']} {info['name']} ({info['dist_km']:.1f} km); "
          f"{int((hourly > 0).sum())} wet hours on {EVENT_DATE}.")
    return info, hourly


# ── Event ARI from Atlas-14 interpolated to the station ─────────────────────────

def atlas14_at_point(lat: float, lon: float) -> dict[tuple[int, int], float]:
    """Interpolate Atlas-14 depth (in) to a point → {(return_period_yr, duration_hr): depth}."""
    a14 = _read_pipeline_filtered(ATLAS14_KEY)
    a14["site_no"] = a14["site_no"].astype(str)
    inv = _read_pipeline_filtered(INV_KEY, columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    a14 = a14.merge(inv, on="site_no", how="left").dropna(subset=["dec_lat_va", "dec_long_va", "depth_in"])
    out: dict[tuple[int, int], float] = {}
    for (rp, dur), g in a14.groupby(["return_period_yr", "duration_hr"]):
        if int(dur) not in ARI_DURATIONS or len(g) < 3:
            continue
        src = g[["dec_lat_va", "dec_long_va"]].to_numpy(float)
        val = g["depth_in"].to_numpy(float)
        d = griddata(src, val, [[lat, lon]], method="linear")[0]
        if not np.isfinite(d):
            d = griddata(src, val, [[lat, lon]], method="nearest")[0]
        if np.isfinite(d):
            out[(int(rp), int(dur))] = float(d)
    return out


def event_ari(hourly: pd.Series, a14_pt: dict[tuple[int, int], float]) -> Optional[dict]:
    """Peak rolling accumulation over ARI_DURATIONS and its (log-log interpolated)
    Atlas-14 return period.  Returns the duration giving the most extreme ARI."""
    if hourly is None or hourly.empty or not a14_pt:
        return None
    per_dur: dict[int, list] = {}
    for (rp, dur), depth in a14_pt.items():
        per_dur.setdefault(dur, []).append((depth, rp))
    best: Optional[dict] = None
    for dur in ARI_DURATIONS:
        pts = sorted((d, rp) for d, rp in per_dur.get(dur, []) if d > 0)
        if len(pts) < 2:
            continue
        obs = float(hourly.rolling(dur, min_periods=1).sum().max())
        depths = np.array([d for d, _ in pts]); rps = np.array([rp for _, rp in pts])
        ari = float(np.exp(np.interp(np.log(max(obs, 1e-6)), np.log(depths), np.log(rps))))
        if obs < depths[0]:                       # below the smallest published RP
            ari = min(ari, rps[0])
        if best is None or ari > best["ari"]:
            best = {"ari": ari, "dur": dur, "depth": obs, "capped": obs >= depths[-1]}
    return best



# ── Step 1: MRMS ───────────────────────────────────────────────────────────────

def _mrms_key(hour: int) -> str:
    ts = f"{EVENT_DATE:%Y%m%d}-{hour:02d}0000"
    fname = f"MRMS_{MRMS_1H_FOLDER}_{ts}.grib2.gz"
    return f"{MRMS_BUCKET}/CONUS/{MRMS_1H_FOLDER}/{EVENT_DATE:%Y%m%d}/{fname}"


def load_mrms_frame(fs: s3fs.S3FileSystem, hour: int) -> Optional[np.ndarray]:
    """Download one 1-h MRMS GRIB, decompress, return cropped 2-D array (inches)."""
    key = _mrms_key(hour)
    try:
        with fs.open(key, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"  [MRMS h{hour:02d}] file not found: {key}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(decompress_gz(raw))
        tmp_path = tmp.name

    try:
        ds  = open_mrms_grib(tmp_path)
        var = list(ds.data_vars)[0]
        arr, lats, lons = canonicalize_mrms_grid(ds[var])
        arr = apply_units(arr, kind="qpe", units="in")
    finally:
        os.unlink(tmp_path)

    # Crop to event bounding box
    lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
    lon_idx = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
    cropped = arr[np.ix_(lat_idx, lon_idx)]
    return np.where(np.isnan(cropped), 0.0, cropped)


def load_all_mrms(fs: s3fs.S3FileSystem) -> tuple[list, np.ndarray, np.ndarray]:
    """Return (frames_1h, lats_crop, lons_crop).

    frames_1h is a list of 24 arrays (None where file is missing).
    Also returns the lat/lon vectors for the first successful frame so the
    caller can build the plot extent.
    """
    print(f"\n── Loading MRMS QPE_01H_Pass2 for {EVENT_DATE} ──────────────────────────")

    # Derive crop lat/lon from the first available file to get exact grid coords
    lats_crop = lons_crop = None
    key0 = _mrms_key(0)
    try:
        with fs.open(key0, "rb") as f:
            raw = f.read()
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(decompress_gz(raw))
            tmp_path = tmp.name
        ds   = open_mrms_grib(tmp_path)
        var  = list(ds.data_vars)[0]
        _, lats_full, lons_full = canonicalize_mrms_grid(ds[var])
        os.unlink(tmp_path)
        lats_crop = lats_full[(lats_full >= LAT_MIN) & (lats_full <= LAT_MAX)]
        lons_crop = lons_full[(lons_full >= LON_MIN) & (lons_full <= LON_MAX)]
    except Exception as e:
        print(f"  WARNING: could not probe grid from hour-00 file: {e}")

    frames: list[Optional[np.ndarray]] = []
    for h in HOURS:
        print(f"  Loading hour {h:02d} UTC...", end="\r")
        frames.append(load_mrms_frame(fs, h))
    print(f"  Loaded {sum(f is not None for f in frames)}/24 MRMS frames.      ")

    return frames, lats_crop, lons_crop


def compute_3h_rolling(frames_1h: list) -> list:
    """Return 3-h rolling accumulation: sum of frames[h-2], [h-1], [h]."""
    result = []
    for h in range(len(frames_1h)):
        stack = [frames_1h[i] for i in range(max(0, h - 2), h + 1)
                 if frames_1h[i] is not None]
        result.append(np.sum(stack, axis=0) if stack else None)
    return result


# ── Step 2: NHD flowlines ──────────────────────────────────────────────────────

def _parse_comid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalise the COMID column name and cast to Int64."""
    for candidate in ("comid", "COMID", "nhdplus_comid", "nhdplusid", "NHDPlusID"):
        if candidate in gdf.columns:
            if candidate != "comid":
                gdf = gdf.rename(columns={candidate: "comid"})
            print(f"  Using '{candidate}' as COMID — {gdf['comid'].notna().sum()} valid.")
            break
    if "comid" not in gdf.columns:
        gdf["comid"] = np.nan
    gdf["comid"] = gdf["comid"].astype("Int64")
    return gdf.to_crs(epsg=4326)


def _fetch_via_epa_rest() -> gpd.GeoDataFrame:
    """EPA WATERS NHDPlus v2 ArcGIS REST — different domain, not blocked by AWS."""
    url = ("https://watersgeo.epa.gov/arcgis/rest/services"
           "/NHDPlus_NP21/NHDSnapshot_NP21/MapServer/0/query")
    r = requests.get(url, params={
        "geometry":       f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}",
        "geometryType":   "esriGeometryEnvelope",
        "inSR":           "4326",
        "outSR":          "4326",
        "spatialRel":     "esriSpatialRelIntersects",
        "outFields":      "COMID,GNIS_Name,StreamOrde",
        "returnGeometry": "true",
        "f":              "geojson",
    }, timeout=60)
    r.raise_for_status()
    gdf = gpd.read_file(io.StringIO(r.text))
    if gdf.empty:
        raise RuntimeError("EPA REST returned no features.")
    return gdf


def _fetch_via_nldi_grid_sample() -> gpd.GeoDataFrame:
    """Capture all NHD flowlines inside the bbox by sampling a 3×3 grid of
    points, discovering every unique COMID, then navigating upstream
    tributaries (UT) from each one.

    This handles bboxes that span multiple sub-watersheds — navigating from
    a single centre COMID would miss streams draining to other outlets.
    Uses api.water.usgs.gov/nldi which the pipeline already calls from AWS.
    """
    import math
    from shapely.geometry import box as shapely_box

    bbox_geom  = shapely_box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    centre_lat = (LAT_MIN + LAT_MAX) / 2
    diag_km = math.sqrt(
        ((LON_MAX - LON_MIN) * 111 * math.cos(math.radians(centre_lat))) ** 2
        + ((LAT_MAX - LAT_MIN) * 111) ** 2
    )
    nav_dist = max(50, int(diag_km * 3))   # km upstream to search from each seed

    # Step 1: find unique COMIDs at 9 interior sample points
    seed_comids: set[str] = set()
    for xf in (0.2, 0.5, 0.8):
        for yf in (0.2, 0.5, 0.8):
            lon = LON_MIN + (LON_MAX - LON_MIN) * xf
            lat = LAT_MIN + (LAT_MAX - LAT_MIN) * yf
            try:
                r = requests.get(
                    "https://api.water.usgs.gov/nldi/linked-data/comid/position",
                    params={"coords": f"POINT({lon} {lat})"},
                    timeout=20,
                )
                if r.ok:
                    cid = r.json()["features"][0]["properties"]["identifier"]
                    seed_comids.add(cid)
            except Exception:
                pass
    print(f"  Grid sampling: {len(seed_comids)} unique seed COMIDs")

    # Step 2: navigate UT from each seed, clip to bbox
    all_gdfs: list[gpd.GeoDataFrame] = []
    for comid in seed_comids:
        try:
            r = requests.get(
                f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
                f"/navigation/UT/flowlines",
                params={"distance": nav_dist},
                timeout=90,
            )
            if not r.ok:
                continue
            gdf = gpd.read_file(io.StringIO(r.text))
            gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
            if not gdf.empty:
                all_gdfs.append(gdf)
        except Exception as e:
            print(f"  UT nav from {comid} failed: {e}")

    if not all_gdfs:
        raise RuntimeError("No flowlines found via grid sampling.")

    # Step 3: union and deduplicate by COMID
    combined = pd.concat(all_gdfs, ignore_index=True)
    comid_col = next(
        (c for c in ("nhdplus_comid", "comid", "COMID") if c in combined.columns),
        None,
    )
    if comid_col:
        combined = combined.drop_duplicates(subset=[comid_col])
    print(f"  {len(combined)} unique flowlines in bbox after union.")
    return combined


def fetch_nhd_flowlines() -> gpd.GeoDataFrame:
    """Return NHDPlus v2 flowlines with numeric COMIDs for the event bbox.

    Attempts three sources in order:
      1. EPA WATERS ArcGIS REST  (NHDPlus_NP21 — not blocked from AWS)
      2. NLDI upstream-tributaries navigation from gauges in the bbox
      3. Raises RuntimeError (caller falls back to grey lines)
    """
    print("\n── Fetching NHD flowlines ───────────────────────────────────────────────")
    for attempt, fetcher in [
        ("EPA WATERS REST",       _fetch_via_epa_rest),
        ("NLDI grid-sample+UT",  _fetch_via_nldi_grid_sample),
    ]:
        try:
            print(f"  Trying {attempt}...")
            gdf = fetcher()
            gdf = _parse_comid_gdf(gdf)
            print(f"  {len(gdf)} flowline features retrieved via {attempt}.")
            return gdf
        except Exception as e:
            print(f"  {attempt} failed: {e}")

    raise RuntimeError("All NHD sources failed.")


def _geom_segments(geom) -> list:
    """Extract [[x1,y1],[x2,y2]] pairs from a LineString or MultiLineString."""
    segs = []
    if geom is None:
        return segs
    if geom.geom_type == "LineString":
        coords = np.array(geom.coords)
        segs = [coords[i : i + 2] for i in range(len(coords) - 1)]
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            coords = np.array(part.coords)
            segs.extend([coords[i : i + 2] for i in range(len(coords) - 1)])
    return segs


def build_line_collection_data(gdf: gpd.GeoDataFrame):
    """Pre-compute segment arrays and per-segment COMID mapping."""
    import pandas as pd
    all_segs: list = []
    seg_comids: list = []
    row_seg_counts: list = []  # how many segments belong to each GDF row
    for _, row in gdf.iterrows():
        segs = _geom_segments(row.geometry)
        all_segs.extend(segs)
        comid = int(row["comid"]) if pd.notna(row["comid"]) else -1
        seg_comids.extend([comid] * len(segs))
        row_seg_counts.append(len(segs))
    return all_segs, np.array(seg_comids), row_seg_counts


# ── Step 3: NWM streamflow ─────────────────────────────────────────────────────

def _nwm_key(hour: int) -> str:
    d = EVENT_DATE.strftime("%Y%m%d")
    return (f"nwm.{d}/analysis_assim/"
            f"nwm.t{hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc")


def load_nwm_frame(
    fs: s3fs.S3FileSystem,
    hour: int,
    target_comids: np.ndarray,
) -> dict[int, float]:
    """Read streamflow (m³/s) for target COMIDs from one NWM analysis file."""
    key = _nwm_key(hour)
    try:
        with fs.open(f"{NWM_BUCKET}/{key}", "rb") as fobj:
            with h5py.File(fobj, "r") as h:
                all_ids = h["feature_id"][:].astype(np.int64)
                q_arr   = h["streamflow"][:].astype(np.float32)
                # scale factor / fill — NWM stores raw float32, no extra scale needed
                q_arr   = np.where(q_arr < 0, np.nan, q_arr)
                idx = np.where(np.isin(all_ids, target_comids))[0]
                return {int(all_ids[i]): float(q_arr[i]) for i in idx}
    except Exception as e:
        print(f"  [NWM h{hour:02d}] error: {e}")
        return {}


def load_all_nwm(
    fs: s3fs.S3FileSystem,
    target_comids: np.ndarray,
) -> list[dict[int, float]]:
    """Load one streamflow dict per hour; missing hours return {}."""
    print(f"\n── Loading NWM analysis streamflow for {EVENT_DATE} ─────────────────────")
    frames: list[dict[int, float]] = []
    for h in HOURS:
        print(f"  Loading hour {h:02d} UTC...", end="\r")
        frames.append(load_nwm_frame(fs, h, target_comids))
    n_ok = sum(len(f) > 0 for f in frames)
    print(f"  Loaded {n_ok}/{len(HOURS)} NWM frames with data.        ")
    return frames


# ── Step 4: Build animation ────────────────────────────────────────────────────

def make_animation(
    frames_1h: list,
    station_info: Optional[dict],
    hyeto: Optional[pd.Series],
    ari: Optional[dict],
    lats: Optional[np.ndarray],
    lons: Optional[np.ndarray],
    flowlines: gpd.GeoDataFrame,
    nwm_frames: list[dict[int, float]],
    all_segs: list,
    seg_comids: np.ndarray,
) -> None:
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # Precipitation colormap: white → yellow → orange → red → purple
    precip_colors = [
        (1.0, 1.0, 1.0, 0.0),   # transparent for zero
        "#b3d9ff", "#6ab4ff", "#1f78b4",
        "#33a02c", "#b2df8a",
        "#ffff33", "#ff7f00",
        "#e31a1c", "#fb9a99",
        "#6a0dad",
    ]
    precip_cmap = mcolors.LinearSegmentedColormap.from_list(
        "nws_precip", precip_colors, N=256
    )
    precip_cmap.set_under("none")

    norm_1h  = mcolors.Normalize(vmin=0.01, vmax=QPE_1H_VMAX)
    norm_nwm = mcolors.LogNorm(vmin=NWM_Q_VMIN, vmax=NWM_Q_VMAX)
    nwm_cmap = plt.get_cmap("plasma_r")   # dark-purple (low) → yellow (high); always visible

    # ── Figure layout: panels 0 & 2 are maps, panel 1 is the hyetograph ─────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={"wspace": 0.30})
    map_titles = {0: "MRMS QPE — 1-h (in)", 2: "NWM streamflow (m³/s)"}
    for i in (0, 2):
        ax = axes[i]
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Longitude")
        if i == 0:
            ax.set_ylabel("Latitude")
        ax.set_title(map_titles[i], fontsize=11)
        ax.set_aspect("equal")

    # Basemap tiles on the map panels only (added once)
    if HAS_CTX:
        print("\nAdding basemap tiles (requires internet)...")
        for ax in (axes[0], axes[2]):
            try:
                ctx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=12,
                    attribution=False,
                    zorder=0,
                )
            except Exception as e:
                print(f"  Basemap warning: {e}")

    # ── Panel 0: MRMS 1-h QPE (station location starred) ────────────────────────
    blank = np.zeros((2, 2))
    im_1h = axes[0].imshow(
        blank, origin="upper", extent=extent,
        cmap=precip_cmap, norm=norm_1h,
        alpha=ALPHA_PRECIP, zorder=2, interpolation="nearest",
    )
    fig.colorbar(im_1h, ax=axes[0], fraction=0.035, pad=0.08, label="Rainfall (in)")
    if station_info is not None:
        axes[0].scatter([station_info["lon"]], [station_info["lat"]], marker="*",
                        s=190, color="black", edgecolors="white", lw=0.9, zorder=6)

    # ── Panel 1: hyetograph of the nearest precip station (moving marker) ────────
    hour_ts = [pd.Timestamp(EVENT_DATE, tz="UTC") + pd.Timedelta(hours=h) for h in HOURS]
    hy = (np.array([float(hyeto.get(t, 0.0)) for t in hour_ts])
          if (hyeto is not None and not hyeto.empty) else np.zeros(len(HOURS)))
    xh = np.arange(len(HOURS))
    axh = axes[1]
    axh.bar(xh, hy, width=0.8, color=HYETO_COLOR, alpha=0.55, zorder=2)
    axh.set_ylim(0, max(hy.max() * 1.30, 0.2))
    axh.set_xlim(-0.6, len(HOURS) - 0.4)
    axh.set_xticks(xh[::2])
    axh.set_xticklabels([f"{h:02d}" for h in HOURS[::2]])
    axh.set_xlabel("Hour (UTC)")
    axh.set_ylabel("Hourly precip (in)")
    axh.grid(axis="y", ls=":", alpha=0.4)
    if station_info is not None:
        axh.set_title(f"Rain gauge {station_info['site_no']} — {station_info['name'][:32]}\n"
                      f"nearest to map centre ({station_info['dist_km']:.0f} km)", fontsize=10)
    else:
        axh.set_title("Nearest GHCNh precip station — NO DATA near this location/date",
                      fontsize=10, color="firebrick")
    hyeto_line = axh.axvline(xh[0], color="0.35", ls="--", lw=1.2, zorder=4)
    hyeto_dot, = axh.plot([xh[0]], [hy[0]], "o", ms=13, color="#e31a1c",
                          mec="white", mew=1.2, zorder=6)
    if ari is not None:
        cap = " (≥ published max)" if ari.get("capped") else ""
        axh.text(0.03, 0.97,
                 f"Peak {ari['depth']:.2f} in / {ari['dur']} h\n≈ {ari['ari']:.0f}-yr ARI{cap}",
                 transform=axh.transAxes, va="top", ha="left", fontsize=10, fontweight="bold",
                 color="#6a0dad",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="#6a0dad"))

    # ── Panel 2: NWM streamflow on flowlines ────────────────────────────────────
    # LineCollection — attach cmap/norm so set_array() always uses the fixed LogNorm.
    lc = LineCollection(
        all_segs if all_segs else [[[LON_MIN, LAT_MIN], [LON_MAX, LAT_MAX]]],
        cmap=nwm_cmap, norm=norm_nwm, linewidths=4.0, zorder=3,
    )
    lc.set_array(np.full(max(len(all_segs), 1), NWM_Q_VMIN))
    axes[2].add_collection(lc)
    sm_nwm = plt.cm.ScalarMappable(cmap=nwm_cmap, norm=norm_nwm)
    sm_nwm.set_array([])
    fig.colorbar(sm_nwm, ax=axes[2], fraction=0.035, pad=0.08,
                 label="Streamflow (m³/s)")

    # Timestamp text (centred above all panels)
    time_text = fig.text(
        0.5, 0.97,
        "",
        ha="center", va="top", fontsize=13, fontweight="bold",
    )

    # ── Update function ────────────────────────────────────────────────────────
    # FuncAnimation iterates over frame indices (0, 1, …); map to UTC hour via HOURS.
    from datetime import datetime, timezone, timedelta
    _EDT = timedelta(hours=-4)

    def update(frame_idx: int):
        h = HOURS[frame_idx]

        # -- MRMS 1h (fixed colour scale: explicitly re-apply clim each frame) --
        f1 = frames_1h[frame_idx]
        if f1 is not None:
            im_1h.set_data(f1)
            im_1h.set_extent(extent)
        else:
            im_1h.set_data(np.zeros((2, 2)))
        im_1h.set_clim(0.01, QPE_1H_VMAX)

        # -- Hyetograph: advance the current-hour marker + time line --
        hyeto_line.set_xdata([xh[frame_idx], xh[frame_idx]])
        hyeto_dot.set_data([xh[frame_idx]], [hy[frame_idx]])

        # -- NWM streamflow --
        q_map = nwm_frames[frame_idx]
        if q_map and all_segs:
            q_vals = np.array(
                [q_map.get(int(c), NWM_Q_VMIN) if c >= 0 else NWM_Q_VMIN
                 for c in seg_comids],
                dtype=float,
            )
            q_vals = np.clip(q_vals, NWM_Q_VMIN, NWM_Q_VMAX)
            # set_array lets the collection apply its own fixed LogNorm —
            # avoids the colour drift caused by set_colors() bypassing it.
            lc.set_array(q_vals)
            log_range = np.log10(NWM_Q_VMAX) - np.log10(NWM_Q_VMIN)
            lw = 2.0 + 6.0 * (np.log10(q_vals) - np.log10(NWM_Q_VMIN)) / log_range
            lc.set_linewidths(np.clip(lw, 2.0, 8.0))
        else:
            lc.set_array(np.full(max(len(all_segs), 1), NWM_Q_VMIN))
            lc.set_linewidths(2.0)

        # -- Timestamp: UTC + local (EDT = UTC-4) --
        utc_dt  = datetime(EVENT_DATE.year, EVENT_DATE.month, EVENT_DATE.day,
                           h, 0, tzinfo=timezone.utc)
        edt_dt  = utc_dt + _EDT
        time_text.set_text(
            f"{EVENT_DATE:%Y-%m-%d}  {h:02d}:00 UTC  "
            f"({edt_dt:%I:%M %p} EDT)"
        )
        return [im_1h, lc, hyeto_dot, hyeto_line, time_text]

    # ── Render ─────────────────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update,
        frames=range(len(HOURS)),
        interval=FRAME_MS,
        blit=False,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nRendering {len(HOURS)} frames → {OUT_GIF}")
    writer = PillowWriter(fps=int(1000 / FRAME_MS))
    anim.save(OUT_GIF, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved locally: {OUT_GIF}")
    upload_gif_to_s3(OUT_GIF)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fs_anon = _anon_fs()

    # 1. MRMS
    frames_1h, lats, lons = load_all_mrms(fs_anon)

    # 1b. Nearest GHCNh precip station hyetograph + event ARI (panel 2)
    station_info, hyeto = load_nearest_precip_station()
    ari = None
    if station_info is not None and hyeto is not None and not hyeto.empty:
        try:
            a14_pt = atlas14_at_point(station_info["lat"], station_info["lon"])
            ari = event_ari(hyeto, a14_pt)
            if ari:
                print(f"  Event ARI at station: peak {ari['depth']:.2f} in / {ari['dur']} h "
                      f"≈ {ari['ari']:.0f}-yr")
        except Exception as e:
            print(f"  ARI computation failed: {e}")

    # 2. NHD flowlines
    try:
        flowlines = fetch_nhd_flowlines()
    except Exception as e:
        print(f"  WARNING: NHD query failed ({e}) — NWM panel will show grey lines.")
        flowlines = gpd.GeoDataFrame({"comid": [], "geometry": []},
                                      crs="EPSG:4326")

    all_segs, seg_comids, _ = build_line_collection_data(flowlines)
    target_comids = seg_comids[seg_comids >= 0]

    # 3. NWM
    if len(target_comids) > 0:
        nwm_frames = load_all_nwm(fs_anon, target_comids)
    else:
        print("  No COMIDs found — skipping NWM download.")
        nwm_frames = [{} for _ in HOURS]

    # 4. Animate
    make_animation(
        frames_1h, station_info, hyeto, ari, lats, lons,
        flowlines, nwm_frames,
        all_segs, seg_comids,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
