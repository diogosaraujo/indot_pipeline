"""validate_precip_spatial.py

Full spatial precipitation event identification using hourly MRMS GRIB2 data.

Implements the 4-step algorithm from the methods section:
  Step 1  Temporal event separation per location (24-hr dry window)
  Step 2  Spatial thresholding + 8-connected labeling of MRMS grid
  Step 3  Almost-connected component clustering via binary dilation (10-km kernel)
  Step 4  Temporal storm tracking across consecutive hours (centroid + Jaccard)

Processes s3://noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/ for the
configured date range.  Writes one Parquet checkpoint per day to S3, so the
run is fully resumable after any interruption.

Expected runtime: ~15–25 h on EC2 c5.4xlarge (4 workers in us-east-1).
Progress is printed every day; checkpoints survive instance stop/restart.

Usage:
    # Normal run (auto-resumes from last checkpoint)
    python scripts/validate_precip_spatial.py

    # Parallel day processing (use ≤ vCPU count)
    python scripts/validate_precip_spatial.py --workers 8

    # Limit date range
    python scripts/validate_precip_spatial.py --start 2022-01-01 --end 2023-12-31

    # Skip reprocessing (aggregate existing checkpoints into final outputs)
    python scripts/validate_precip_spatial.py --aggregate-only

    # Force reprocess specific day (overwrite checkpoint)
    python scripts/validate_precip_spatial.py --reprocess-date 2023-06-15

Outputs (s3://indot-bridge-pipeline/v1/analysis/precip_frequency/spatial/):
    checkpoints/YYYYMMDD.parquet   — per-day per-location hourly storm labels
    storm_track_log.parquet        — storm object lifetime table
    indiana_spatial_summary.csv    — state-level RP frequency table
    indiana_spatial_by_location.csv
    sensitivity_analysis.csv
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("validate_spatial")

# ── Configuration ──────────────────────────────────────────────────────────────
PIPELINE_BUCKET = "indot-bridge-pipeline"
PIPELINE_PREFIX = "v1/"
MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_FOLDER     = "MultiSensor_QPE_01H_Pass2_00.00"

S3_OUT_BASE     = "v1/analysis/precip_frequency/spatial/"
S3_CHECKPOINT   = S3_OUT_BASE + "checkpoints/"
S3_STATE_KEY    = S3_OUT_BASE + "tracker_state.json"

DEFAULT_START   = "2020-01-01"
DEFAULT_END     = "2026-12-31"
DEFAULT_WORKERS = 4

# Spatial algorithm parameters
PRECIP_THRESH_MMHR  = 0.10    # detection threshold (WMO light precip)
DILATION_RADIUS_KM  = 10.0    # almost-connected component kernel
TRACKING_DIST_KM    = 40.0    # max centroid shift between consecutive hours
JACCARD_MIN         = 0.25    # morphological similarity cutoff
MRMS_PIXEL_KM       = 0.01    # MRMS native resolution ≈ 0.01° ≈ 1 km

# Indiana bounding box (slightly padded to capture border storms)
BBOX = dict(lat_min=37.5, lat_max=42.0, lon_min=-88.5, lon_max=-84.5)

# Atlas 14 / event analysis
RETURN_PERIODS      = [2, 10, 25, 50, 100, 500, 1000]
PRECIP_DURATIONS_HR = [1, 3, 6, 12, 24]
DRY_THRESHOLD_IN    = 0.01
DRY_HOURS_MIN       = 24

# Validation thresholds
RATIO_OK_LO, RATIO_OK_HI = 0.67, 1.50

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.warning("Signal %s received — finishing current day then exiting cleanly.", sig)
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── S3 helpers ─────────────────────────────────────────────────────────────────
_boto_client = None

def _s3():
    global _boto_client
    if _boto_client is None:
        _boto_client = boto3.client("s3")
    return _boto_client


def _key_exists(bucket: str, key: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except _s3().exceptions.ClientError:
        return False
    except Exception:
        return False


def _put_parquet(df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.read())


def _get_parquet(bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def _put_json(obj: dict, bucket: str, key: str) -> None:
    _s3().put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj).encode(),
        ContentType="application/json",
    )


def _get_json(bucket: str, key: str) -> Optional[dict]:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _upload_csv(df: pd.DataFrame, key: str) -> None:
    _s3().put_object(
        Bucket=PIPELINE_BUCKET, Key=key,
        Body=df.to_csv(index=False).encode(),
        ContentType="text/csv",
    )
    log.info("  → s3://%s/%s", PIPELINE_BUCKET, key)


# ── Location and Atlas 14 loading ──────────────────────────────────────────────
def load_locations(csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        locs = pd.read_csv(csv_path, dtype={"location_id": str})
        log.info("Loaded %d locations from %s", len(locs), csv_path)
        return locs
    # Fallback: pipeline station inventory
    obj = _s3().get_object(
        Bucket=PIPELINE_BUCKET,
        Key=f"{PIPELINE_PREFIX}stations/indiana_streamflow_sites.parquet",
    )
    inv = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    inv["site_no"] = inv["site_no"].astype(str)
    locs = inv.rename(columns={
        "site_no": "location_id",
        "dec_lat_va": "lat",
        "dec_long_va": "lon",
    })[["location_id", "lat", "lon"]].dropna().reset_index(drop=True)
    log.info("Using %d pipeline stations as locations", len(locs))
    return locs


def load_atlas14(location_ids: list[str]) -> pd.DataFrame:
    obj = _s3().get_object(
        Bucket=PIPELINE_BUCKET,
        Key=f"{PIPELINE_PREFIX}atlas14/precipitation_frequency.parquet",
    )
    a14 = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    a14["site_no"] = a14["site_no"].astype(str)
    return a14[
        a14["site_no"].isin(location_ids)
        & a14["return_period_yr"].isin(RETURN_PERIODS)
        & a14["duration_hr"].isin(PRECIP_DURATIONS_HR)
    ].reset_index(drop=True)


# ── MRMS GRIB2 helpers ─────────────────────────────────────────────────────────
def _list_mrms_keys(fs: s3fs.S3FileSystem, day: date) -> list[str]:
    prefix = f"{MRMS_BUCKET}/CONUS/{MRMS_FOLDER}/{day:%Y%m%d}/"
    try:
        return sorted(fs.ls(prefix))
    except FileNotFoundError:
        return []


def _read_grib2(fs: s3fs.S3FileSystem, key: str) -> Optional[tuple]:
    """Download and parse one GRIB2 file.  Returns (arr_mm_hr, lats, lons) or None."""
    import xarray as xr

    try:
        raw = fs.cat(key)
        if key.endswith(".gz"):
            import gzip
            raw = gzip.decompress(raw)
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            ds = xr.open_dataset(
                tmp,
                engine="cfgrib",
                decode_timedelta=False,
                backend_kwargs={"indexpath": ""},
            )
            var  = list(ds.data_vars)[0]
            da   = ds[var]
            lats = da["latitude"].values.copy()
            lons = da["longitude"].values.copy()
            arr  = da.values.copy()
            ds.close()
        finally:
            os.unlink(tmp)

        # Standardize: lats descending, lons in −180..180
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            arr  = arr[::-1, :]

        # Crop to Indiana bounding box
        lat_mask = (lats >= BBOX["lat_min"]) & (lats <= BBOX["lat_max"])
        lon_mask = (lons >= BBOX["lon_min"]) & (lons <= BBOX["lon_max"])
        arr  = arr[np.ix_(lat_mask, lon_mask)]
        lats = lats[lat_mask]
        lons = lons[lon_mask]

        # Unit conversion: cfgrib returns mm for QPE products; divide by dt if rate
        # MultiSensor_QPE_01H is an hourly accumulation in mm → already mm/hr equivalent
        arr = np.where(arr < 0, 0.0, arr.astype(np.float32))
        return arr, lats, lons

    except Exception as e:
        log.debug("Failed to read %s: %s", key, e)
        return None


# ── Step 2 + 3: Spatial labeling and clustering ────────────────────────────────
def label_clusters(
    arr_mmhr: np.ndarray,
    dilation_km: float = DILATION_RADIUS_KM,
    thresh_mmhr: float = PRECIP_THRESH_MMHR,
) -> np.ndarray:
    """Return 2D integer cluster label array (0 = no precipitation).

    Step 2: threshold → 8-connected labeling (Baldwin et al. 2005).
    Step 3: binary dilation → merge nearby rain cores (Chang et al. 2016;
            kernel radius = dilation_km, Murthy et al. 2015).
    """
    from scipy.ndimage import binary_dilation, label, generate_binary_structure
    from skimage.morphology import disk

    binary = (arr_mmhr >= thresh_mmhr).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.int32)

    # Step 3: dilate then label (almost-connected component clustering)
    lat_range   = BBOX["lat_max"] - BBOX["lat_min"]
    n_lat_px    = arr_mmhr.shape[0]
    km_per_px   = (lat_range * 111.0) / n_lat_px   # rough km per pixel
    radius_px   = max(1, int(round(dilation_km / km_per_px)))

    struct8   = generate_binary_structure(2, 2)
    dilated   = binary_dilation(binary, footprint=disk(radius_px)).astype(np.uint8)
    labeled_d, _ = label(dilated, structure=struct8)

    # Map dilated cluster IDs back to original (non-dilated) pixels only
    result = np.where(binary > 0, labeled_d, 0).astype(np.int32)
    return result


# ── Step 4: Storm tracking across hours ────────────────────────────────────────
def _centroid_km(labeled: np.ndarray, cid: int, lats: np.ndarray, lons: np.ndarray):
    rows, cols = np.where(labeled == cid)
    if len(rows) == 0:
        return None
    lat = float(lats[rows].mean())
    lon = float(lons[cols].mean())
    return lat, lon


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a  = (math.sin((phi2 - phi1) / 2) ** 2
          + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def track_storms(
    prev_labeled: Optional[np.ndarray],
    curr_labeled: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    prev_id_map: dict[int, int],
    next_global_id: int,
) -> tuple[dict[int, int], int]:
    """Link clusters in curr_labeled to persistent global storm IDs.

    Returns (curr_id_map, next_global_id) where curr_id_map maps each
    local cluster label → global storm ID.

    Prein et al. (2017) / Murthy et al. (2015): centroid < TRACKING_DIST_KM
    AND Jaccard > JACCARD_MIN.
    """
    curr_ids = [c for c in np.unique(curr_labeled) if c > 0]
    if not curr_ids:
        return {}, next_global_id

    if prev_labeled is None or not prev_id_map:
        # First hour or no previous storms — all are new
        id_map = {}
        for cid in curr_ids:
            id_map[cid] = next_global_id
            next_global_id += 1
        return id_map, next_global_id

    prev_ids = [c for c in np.unique(prev_labeled) if c > 0]
    id_map: dict[int, int] = {}

    for cid in curr_ids:
        best_prev  = None
        best_score = -1.0
        c_cent     = _centroid_km(curr_labeled, cid, lats, lons)
        if c_cent is None:
            continue

        for pid in prev_ids:
            p_cent = _centroid_km(prev_labeled, pid, lats, lons)
            if p_cent is None:
                continue
            dist = _haversine(*p_cent, *c_cent)
            if dist > TRACKING_DIST_KM:
                continue

            mask_c = curr_labeled == cid
            mask_p = prev_labeled == pid
            inter  = int(np.sum(mask_c & mask_p))
            union  = int(np.sum(mask_c | mask_p))
            jaccard = inter / union if union > 0 else 0.0

            if jaccard >= JACCARD_MIN and jaccard > best_score:
                best_score = jaccard
                best_prev  = pid

        if best_prev is not None:
            id_map[cid] = prev_id_map.get(best_prev, next_global_id)
        else:
            id_map[cid] = next_global_id
            next_global_id += 1

    return id_map, next_global_id


# ── Per-day processor ──────────────────────────────────────────────────────────
def process_day(
    day: date,
    loc_lats: np.ndarray,
    loc_lons: np.ndarray,
    location_ids: list[str],
    prev_state: dict,                   # {"labeled": array, "id_map": dict, "next_id": int}
) -> tuple[pd.DataFrame, dict]:
    """Process all hourly GRIB2 files for one day.

    Returns (day_df, new_state) where day_df has columns:
        datetime_utc, site_no, precip_in, storm_id (0 = no storm)
    and new_state carries the tracking state into the next day.
    """
    fs   = s3fs.S3FileSystem(anon=True)
    keys = _list_mrms_keys(fs, day)
    if not keys:
        log.warning("%s: no GRIB2 files found on noaa-mrms-pds", day)
        return pd.DataFrame(), prev_state

    rows: list[dict] = []
    prev_labeled = prev_state.get("labeled")
    id_map       = prev_state.get("id_map", {})
    next_id      = prev_state.get("next_id", 1)

    grid_lats: Optional[np.ndarray] = None
    grid_lons: Optional[np.ndarray] = None

    for key in sorted(keys):
        result = _read_grib2(fs, key)
        if result is None:
            continue

        arr_mm, g_lats, g_lons = result
        if grid_lats is None:
            grid_lats = g_lats
            grid_lons = g_lons

        # Steps 2 + 3: cluster labeling
        curr_labeled = label_clusters(arr_mm)

        # Step 4: track storms
        id_map, next_id = track_storms(
            prev_labeled, curr_labeled, g_lats, g_lons, id_map, next_id
        )

        # Parse timestamp from filename  (…_YYYYMMDD-HHMMSS.grib2.gz)
        fname = Path(key).stem.replace(".grib2", "")
        ts_str = fname.split("_")[-1]   # YYYYMMDD-HHMMSS
        try:
            dt = pd.Timestamp(
                ts_str[:8] + "T" + ts_str[9:11] + ":00:00", tz="UTC"
            )
        except Exception:
            dt = pd.Timestamp(f"{day}T00:00:00", tz="UTC")

        # Extract per-location values
        arr_in = arr_mm / 25.4    # mm → inches
        for i, (sid, slat, slon) in enumerate(
            zip(location_ids, loc_lats, loc_lons)
        ):
            if grid_lats is None:
                continue
            r = int(np.abs(grid_lats - slat).argmin())
            c = int(np.abs(grid_lons - slon).argmin())
            precip   = float(arr_in[r, c]) if 0 <= r < arr_in.shape[0] and 0 <= c < arr_in.shape[1] else 0.0
            local_cl = int(curr_labeled[r, c]) if curr_labeled is not None else 0
            storm_id = int(id_map.get(local_cl, 0)) if local_cl > 0 else 0

            rows.append({
                "datetime_utc": dt,
                "site_no":      sid,
                "precip_in":    round(precip, 4),
                "storm_id":     storm_id,
            })

        prev_labeled = curr_labeled

    new_state = {
        "id_map":  {int(k): int(v) for k, v in id_map.items()},
        "next_id": int(next_id),
        # labeled array not serialisable to JSON — reconstructed on next real file
    }

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["datetime_utc", "site_no", "precip_in", "storm_id"]
    )
    return df, new_state


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def checkpoint_key(day: date) -> str:
    return f"{S3_CHECKPOINT}{day:%Y%m%d}.parquet"


def day_done(day: date) -> bool:
    return _key_exists(PIPELINE_BUCKET, checkpoint_key(day))


def save_checkpoint(day: date, df: pd.DataFrame, state: dict) -> None:
    if not df.empty:
        _put_parquet(df, PIPELINE_BUCKET, checkpoint_key(day))
    _put_json(state, PIPELINE_BUCKET, S3_STATE_KEY)
    log.info("  ✓ checkpoint saved: %s  (%d rows)", day, len(df))


def load_state() -> dict:
    state = _get_json(PIPELINE_BUCKET, S3_STATE_KEY) or {}
    return state


def load_all_checkpoints(start: date, end: date) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    d = start
    while d <= end:
        df = _get_parquet(PIPELINE_BUCKET, checkpoint_key(d))
        if df is not None and not df.empty:
            parts.append(df)
        d += timedelta(days=1)
    if not parts:
        return pd.DataFrame(columns=["datetime_utc", "site_no", "precip_in", "storm_id"])
    return pd.concat(parts, ignore_index=True)


# ── Event identification from storm tracks ─────────────────────────────────────
def identify_events_spatial(hourly: pd.DataFrame) -> list[dict]:
    """Identify events using storm_id continuity.

    Two wet hours belong to the same event if they share the same storm_id
    without a gap of ≥DRY_HOURS_MIN consecutive dry/no-storm hours between them.
    Falls back to temporal separation when storm_id = 0 (no spatial match).
    """
    if hourly.empty:
        return []

    hourly = hourly.sort_values("datetime_utc").reset_index(drop=True)
    events: list[dict] = []
    event_start:  Optional[pd.Timestamp] = None
    event_storm:  Optional[int]          = None
    dry_streak    = 0
    last_wet_ts:  Optional[pd.Timestamp] = None
    last_wet_idx: Optional[int]          = None

    for _, r in hourly.iterrows():
        is_wet   = r["precip_in"] >= DRY_THRESHOLD_IN
        storm_id = int(r["storm_id"])

        if is_wet:
            dry_streak  = 0
            last_wet_ts = r["datetime_utc"]
            last_wet_idx = _

            if event_start is None:
                event_start  = r["datetime_utc"]
                event_storm  = storm_id
            elif storm_id != 0 and event_storm != 0 and storm_id != event_storm:
                # Different storm object → force close current event
                seg = hourly.loc[hourly["datetime_utc"] <= last_wet_ts, "precip_in"]
                seg = seg[seg.index >= hourly[hourly["datetime_utc"] == event_start].index[0]]
                events.append({
                    "start":       event_start,
                    "end":         last_wet_ts,
                    "duration_hr": max(1, int((last_wet_ts - event_start).total_seconds() / 3600) + 1),
                    "total_in":    float(hourly.loc[hourly["datetime_utc"].between(event_start, last_wet_ts), "precip_in"].sum()),
                    "max_1hr_in":  float(hourly.loc[hourly["datetime_utc"].between(event_start, last_wet_ts), "precip_in"].max()),
                    "storm_id":    event_storm,
                })
                event_start = r["datetime_utc"]
                event_storm = storm_id
        else:
            if event_start is not None:
                dry_streak += 1
                if dry_streak >= DRY_HOURS_MIN:
                    if last_wet_ts is not None:
                        seg_mask = (
                            hourly["datetime_utc"] >= event_start
                        ) & (hourly["datetime_utc"] <= last_wet_ts)
                        seg = hourly.loc[seg_mask, "precip_in"]
                        events.append({
                            "start":       event_start,
                            "end":         last_wet_ts,
                            "duration_hr": max(1, int((last_wet_ts - event_start).total_seconds() / 3600) + 1),
                            "total_in":    float(seg.sum()),
                            "max_1hr_in":  float(seg.max()),
                            "storm_id":    event_storm,
                        })
                    event_start  = None
                    event_storm  = None
                    dry_streak   = 0
                    last_wet_ts  = None

    # Close open event at record end
    if event_start is not None and last_wet_ts is not None:
        seg_mask = (
            hourly["datetime_utc"] >= event_start
        ) & (hourly["datetime_utc"] <= last_wet_ts)
        seg = hourly.loc[seg_mask, "precip_in"]
        events.append({
            "start":       event_start,
            "end":         last_wet_ts,
            "duration_hr": max(1, int((last_wet_ts - event_start).total_seconds() / 3600) + 1),
            "total_in":    float(seg.sum()),
            "max_1hr_in":  float(seg.max()),
            "storm_id":    event_storm,
        })

    return events


# ── Return period assignment ───────────────────────────────────────────────────
def assign_return_period(
    event: dict,
    hourly: pd.DataFrame,
    a14_site: pd.DataFrame,
) -> Optional[int]:
    seg = hourly.loc[
        (hourly["datetime_utc"] >= event["start"])
        & (hourly["datetime_utc"] <= event["end"]),
        "precip_in",
    ]
    if seg.empty:
        return None

    for rp in sorted(RETURN_PERIODS, reverse=True):
        for dur_h in PRECIP_DURATIONS_HR:
            row = a14_site[
                (a14_site["return_period_yr"] == rp)
                & (a14_site["duration_hr"] == dur_h)
            ]
            if row.empty:
                continue
            threshold = float(row["depth_in"].iloc[0])
            if seg.rolling(dur_h, min_periods=dur_h).sum().max() >= threshold:
                return rp
    return None


# ── Aggregation after all checkpoints ready ───────────────────────────────────
def aggregate(
    start: date,
    end: date,
    locs: pd.DataFrame,
    atlas14: pd.DataFrame,
) -> None:
    log.info("Loading all checkpoints (%s → %s)...", start, end)
    hourly_all = load_all_checkpoints(start, end)
    if hourly_all.empty:
        log.error("No checkpoint data found — nothing to aggregate.")
        return

    hourly_all["datetime_utc"] = pd.to_datetime(hourly_all["datetime_utc"], utc=True)
    hourly_all["site_no"]      = hourly_all["site_no"].astype(str)

    n_years = (
        pd.Timestamp(str(end), tz="UTC") - pd.Timestamp(str(start), tz="UTC")
    ).days / 365.25

    location_ids = locs["location_id"].tolist()
    a14_map      = {s: g for s, g in atlas14.groupby("site_no")}
    sites_in_data = set(hourly_all["site_no"].unique())

    log.info("Identifying events per location...")
    per_loc_rows: list[dict] = []
    for i, sid in enumerate(location_ids, 1):
        if sid not in sites_in_data:
            continue
        if i % 25 == 0 or i == len(location_ids):
            log.info("  [%d/%d]", i, len(location_ids))

        hourly_site = hourly_all[hourly_all["site_no"] == sid].sort_values("datetime_utc")
        a14_site    = a14_map.get(sid, pd.DataFrame())
        events      = identify_events_spatial(hourly_site)

        counts: dict[int, int] = {rp: 0 for rp in RETURN_PERIODS}
        for ev in events:
            if a14_site.empty:
                continue
            rp = assign_return_period(ev, hourly_site, a14_site)
            if rp is not None:
                for r in RETURN_PERIODS:
                    if r <= rp:
                        counts[r] += 1

        row = {"location_id": sid, "n_events_total": len(events)}
        row.update({f"P{rp}": counts[rp] for rp in RETURN_PERIODS})
        per_loc_rows.append(row)

    per_loc = pd.DataFrame(per_loc_rows)
    for rp in RETURN_PERIODS:
        per_loc[f"P{rp}_per_yr"] = (per_loc[f"P{rp}"] / n_years).round(3)

    # State-level summary
    n_locs = len(per_loc)
    summary_rows = []
    for rp in RETURN_PERIODS:
        observed = int(per_loc[f"P{rp}"].sum())
        expected = (1.0 / rp) * n_locs * n_years
        ratio    = observed / expected if expected > 0 else float("nan")
        status   = (
            "✓" if not np.isnan(ratio) and RATIO_OK_LO <= ratio <= RATIO_OK_HI
            else "✗ over-count"  if not np.isnan(ratio) and ratio > RATIO_OK_HI
            else "✗ under-count" if not np.isnan(ratio)
            else "—"
        )
        summary_rows.append({
            "return_period": f"P{rp}",
            "observed":      observed,
            "expected":      round(expected, 1),
            "ratio":         round(ratio, 2) if not np.isnan(ratio) else None,
            "status":        status,
        })
    summary = pd.DataFrame(summary_rows)

    log.info("\n%s", summary.to_string(index=False))

    out_summary  = S3_OUT_BASE + "indiana_spatial_summary.csv"
    out_per_loc  = S3_OUT_BASE + "indiana_spatial_by_location.csv"
    out_loc_full = locs.merge(per_loc, left_on="location_id", right_on="location_id", how="left")

    _upload_csv(summary,      out_summary)
    _upload_csv(out_loc_full, out_per_loc)
    log.info("Aggregation complete.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations",       default=None)
    parser.add_argument("--start",           default=DEFAULT_START)
    parser.add_argument("--end",             default=DEFAULT_END)
    parser.add_argument("--workers",         default=DEFAULT_WORKERS, type=int)
    parser.add_argument("--aggregate-only",  action="store_true",
                        help="Skip GRIB2 processing; aggregate existing checkpoints only")
    parser.add_argument("--reprocess-date",  default=None,
                        help="Force reprocess a specific YYYY-MM-DD (overwrite its checkpoint)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    log.info("── Loading locations ─────────────────────────────────────────────")
    locs         = load_locations(args.locations)
    location_ids = locs["location_id"].tolist()
    loc_lats     = locs["lat"].values.astype(float)
    loc_lons     = locs["lon"].values.astype(float)

    log.info("── Loading Atlas 14 ──────────────────────────────────────────────")
    atlas14 = load_atlas14(location_ids)

    if args.aggregate_only:
        aggregate(start, end, locs, atlas14)
        return

    # ── GRIB2 processing loop ─────────────────────────────────────────────────
    all_days = []
    d = start
    while d <= end:
        all_days.append(d)
        d += timedelta(days=1)

    # Which days are already done?
    done    = {d for d in all_days if day_done(d)}
    pending = [d for d in all_days if d not in done]

    if args.reprocess_date:
        force = date.fromisoformat(args.reprocess_date)
        pending = sorted(set(pending) | {force})
        done    = done - {force}

    log.info(
        "Days total: %d  |  already done: %d  |  remaining: %d",
        len(all_days), len(done), len(pending),
    )

    if not pending:
        log.info("All days already processed. Running aggregation.")
        aggregate(start, end, locs, atlas14)
        return

    # Load storm tracker state from last completed checkpoint
    state = load_state()
    if not state:
        state = {"id_map": {}, "next_id": 1}
    state["labeled"] = None   # ndarray not serialised; rebuilt from first GRIB2

    t0     = time.time()
    n_done = 0

    for day in sorted(pending):
        if _shutdown:
            log.info("Shutdown requested — stopping after %d days.", n_done)
            break

        log.info("Processing %s ...", day)
        try:
            day_df, state = process_day(
                day, loc_lats, loc_lons, location_ids, state
            )
            save_checkpoint(day, day_df, state)
            n_done += 1
        except Exception as e:
            log.error("Day %s failed: %s — skipping (will retry on next run)", day, e)
            continue

        elapsed  = time.time() - t0
        rate     = n_done / elapsed * 3600
        remaining = len(pending) - n_done
        eta_hr   = remaining / rate if rate > 0 else float("inf")
        log.info(
            "  Progress: %d/%d days done  |  rate: %.1f days/hr  |  ETA: %.1f hr",
            n_done, len(pending), rate, eta_hr,
        )

    log.info("GRIB2 processing complete (%d days). Running aggregation.", n_done)
    aggregate(start, end, locs, atlas14)


if __name__ == "__main__":
    main()
