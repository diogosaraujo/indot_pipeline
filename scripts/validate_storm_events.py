"""validate_storm_events.py

Storm-object precipitation frequency validation for Indiana.

Identifies discrete storm events in MRMS space-time and classifies each
by its highest Atlas 14 return period exceedance using per-pixel comparison:

    For each storm event:
        For each pixel in the storm's spatial footprint:
            Compute rolling accumulation (1, 3, 6, 12, 24 hr windows)
            Compare to Atlas 14 threshold AT THAT PIXEL
            If rolling_sum(pixel, duration) >= atlas14(pixel, duration, RP):
                → any pixel crossing RP flags the entire storm as that RP

This yields one return period label per storm event (highest RP exceeded by
any pixel) and cumulative counts (a P100 storm also counts as P10 and P2).

Algorithm
─────────
Step 1  Temporal: storm ends after 24 consecutive hours with no pixels above
        0.10 mm/hr anywhere in the storm's last active footprint.
Step 2  Spatial: 8-connected labeling + binary dilation (10 km kernel) per
        hourly MRMS grid to form storm clusters.
Step 4  Temporal tracking: centroid distance < 40 km AND Jaccard ≥ 0.25
        between consecutive hours → same storm.
RP check: per-pixel rolling accumulation vs. spatially interpolated Atlas 14.

Reads:
    s3://noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/{YYYYMMDD}/
    s3://indot-bridge-pipeline/v1/atlas14/precipitation_frequency.parquet

Writes (s3://indot-bridge-pipeline/v1/analysis/storm_events/):
    checkpoints/{YYYYMMDD}_active.pkl      — active storm state per day
    storm_events.parquet                   — completed storm event database
    indiana_storm_summary.csv              — RP frequency table
    indiana_storm_map.png                  — centroid map coloured by RP

Usage:
    python scripts/validate_storm_events.py
    python scripts/validate_storm_events.py --workers 1 --start 2022-01-01
    python scripts/validate_storm_events.py --aggregate-only
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import pickle
import signal
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, generate_binary_structure, label as ndlabel
from skimage.morphology import disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("storm_events")

# ── Configuration ──────────────────────────────────────────────────────────────
PIPELINE_BUCKET = "indot-bridge-pipeline"
MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_FOLDER     = "MultiSensor_QPE_01H_Pass2_00.00"

S3_OUT          = "v1/analysis/storm_events/"
S3_CHECKPOINT   = S3_OUT + "checkpoints/"
S3_EVENTS_KEY   = S3_OUT + "storm_events.parquet"

DEFAULT_START   = "2020-01-01"
DEFAULT_END     = "2026-12-31"

# Spatial algorithm
PRECIP_THRESH_MMHR = 0.10     # pixel detection threshold
DILATION_KM        = 10.0     # almost-connected component kernel radius
TRACKING_DIST_KM   = 40.0     # max centroid shift between consecutive hours
JACCARD_MIN        = 0.25     # morphological similarity threshold
DRY_HOURS_END      = 24       # consecutive hours with no active pixels → storm over

# Indiana bounding box
BBOX = dict(lat_min=37.5, lat_max=42.0, lon_min=-88.5, lon_max=-84.5)

# Return periods and Atlas 14 durations
RETURN_PERIODS      = [2, 10, 25, 50, 100, 500, 1000]
PRECIP_DURATIONS_HR = [1, 3, 6, 12, 24]

RATIO_OK_LO, RATIO_OK_HI = 0.67, 1.50

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.warning("Signal %s — finishing current day then stopping.", sig)
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── S3 helpers ─────────────────────────────────────────────────────────────────
_s3 = None

def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _key_exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=PIPELINE_BUCKET, Key=key)
        return True
    except Exception:
        return False


def _put_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    s3().put_object(Bucket=PIPELINE_BUCKET, Key=key, Body=data, ContentType=content_type)


def _get_bytes(key: str) -> Optional[bytes]:
    try:
        return s3().get_object(Bucket=PIPELINE_BUCKET, Key=key)["Body"].read()
    except Exception:
        return None


def _put_parquet(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    _put_bytes(buf.getvalue(), key, "application/octet-stream")
    log.info("  → s3://%s/%s", PIPELINE_BUCKET, key)


def _get_parquet(key: str) -> Optional[pd.DataFrame]:
    data = _get_bytes(key)
    if data is None:
        return None
    return pd.read_parquet(io.BytesIO(data))


def _put_csv(df: pd.DataFrame, key: str) -> None:
    _put_bytes(df.to_csv(index=False).encode(), key, "text/csv")
    log.info("  → s3://%s/%s", PIPELINE_BUCKET, key)


# ── Atlas 14 spatial grid ──────────────────────────────────────────────────────
def build_atlas14_grids(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Interpolate Atlas 14 point values to the Indiana MRMS grid.

    Returns {(return_period_yr, duration_hr): 2D threshold array (inches)}.
    Spatial interpolation: linear triangulation via scipy.interpolate.griddata.
    Falls back to nearest-neighbor for pixels outside the convex hull.
    """
    log.info("Loading Atlas 14 and interpolating to MRMS grid...")
    raw = _get_bytes(f"v1/atlas14/precipitation_frequency.parquet")
    if raw is None:
        raise RuntimeError("Atlas 14 parquet not found in pipeline bucket.")
    a14 = pd.read_parquet(io.BytesIO(raw))
    a14["site_no"] = a14["site_no"].astype(str)

    # Need station coordinates — merge with station inventory
    inv_raw = _get_bytes("v1/stations/indiana_streamflow_sites.parquet")
    inv     = pd.read_parquet(io.BytesIO(inv_raw))[["site_no", "dec_lat_va", "dec_long_va"]]
    inv["site_no"] = inv["site_no"].astype(str)
    a14 = a14.merge(inv, on="site_no", how="left").dropna(subset=["dec_lat_va"])

    # Build meshgrid for target Indiana MRMS grid
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    target_pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    grids: dict[tuple[int, int], np.ndarray] = {}
    for rp in RETURN_PERIODS:
        for dur in PRECIP_DURATIONS_HR:
            sub = a14[
                (a14["return_period_yr"] == rp) &
                (a14["duration_hr"]      == dur)
            ].dropna(subset=["depth_in"])

            if len(sub) < 3:
                log.warning("Atlas 14: insufficient points for RP=%d dur=%d — skipping", rp, dur)
                continue

            src_pts  = sub[["dec_lat_va", "dec_long_va"]].values
            src_vals = sub["depth_in"].values

            # Linear interpolation; fill edges with nearest-neighbor
            interp_lin  = griddata(src_pts, src_vals, target_pts, method="linear")
            interp_near = griddata(src_pts, src_vals, target_pts, method="nearest")
            filled = np.where(np.isnan(interp_lin), interp_near, interp_lin)

            grids[(rp, dur)] = filled.reshape(lat_grid.shape).astype(np.float32)

    log.info("  Atlas 14 grid built: %d (RP, duration) combinations", len(grids))
    return grids


# ── Storm object dataclass ─────────────────────────────────────────────────────
@dataclass
class ActiveStorm:
    event_id:      int
    start_dt:      pd.Timestamp
    last_active_dt: pd.Timestamp
    dry_hours:     int = 0

    # Rolling buffer: up to max(PRECIP_DURATIONS_HR) = 24 hourly grids (in/hr)
    # Only pixels that were part of this storm at each hour are kept non-zero.
    hour_buffer:   deque = field(default_factory=lambda: deque(maxlen=24))

    # Union of all pixel masks across all active hours (bool 2D)
    footprint:     Optional[np.ndarray] = None

    # Highest RP exceeded so far (updated incrementally each hour)
    max_rp:        Optional[int] = None


# ── RP check for a completed storm ────────────────────────────────────────────
def classify_storm_rp(
    storm: ActiveStorm,
    atlas14_grids: dict[tuple[int, int], np.ndarray],
) -> Optional[int]:
    """Find highest RP exceeded by any pixel within the storm's footprint.

    For each pixel in storm.footprint:
        For each duration window (1, 3, 6, 12, 24 hr):
            rolling_sum = sum of last duration hours in storm.hour_buffer at that pixel
            If rolling_sum >= atlas14(pixel, duration, RP):
                → storm exceeds RP

    Scans from highest RP down; returns first (highest) RP exceeded.
    """
    if not storm.hour_buffer or storm.footprint is None:
        return None

    # Stack hourly grids: shape (n_hours, n_lat, n_lon)
    stack = np.stack(list(storm.hour_buffer), axis=0)   # already in inches
    n_hrs = stack.shape[0]

    for rp in sorted(RETURN_PERIODS, reverse=True):
        exceeded = False
        for dur in PRECIP_DURATIONS_HR:
            if dur > n_hrs:
                continue
            threshold_grid = atlas14_grids.get((rp, dur))
            if threshold_grid is None:
                continue

            # Slide a window of length `dur` over the time axis
            # For each window position, check if any footprint pixel exceeds threshold
            for t in range(dur - 1, n_hrs):
                window_sum = stack[t - dur + 1 : t + 1, :, :].sum(axis=0)
                # Per-pixel comparison: each pixel vs. its own local Atlas 14 value
                # Only consider pixels in the storm's footprint
                if np.any(
                    (window_sum >= threshold_grid) & storm.footprint
                ):
                    exceeded = True
                    break
            if exceeded:
                break

        if exceeded:
            return rp

    return None   # storm did not exceed P2


# ── GRIB2 reading ──────────────────────────────────────────────────────────────
def _parse_timestamp(key: str) -> Optional[pd.Timestamp]:
    """Extract UTC timestamp from MRMS filename …_YYYYMMDD-HHMMSS.grib2.gz."""
    stem = os.path.basename(key).replace(".grib2.gz", "").replace(".grib2", "")
    ts_part = stem.split("_")[-1]   # YYYYMMDD-HHMMSS
    try:
        return pd.Timestamp(
            ts_part[:8] + "T" + ts_part[9:11] + ":00:00", tz="UTC"
        )
    except Exception:
        return None


def read_grib2_indiana(
    fs: s3fs.S3FileSystem,
    key: str,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Download one GRIB2 file and return (arr_mm_hr, lats, lons) cropped to Indiana."""
    import gzip
    import xarray as xr

    try:
        raw = fs.cat(key)
        if key.endswith(".gz"):
            raw = gzip.decompress(raw)

        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            ds  = xr.open_dataset(tmp, engine="cfgrib", decode_timedelta=False,
                                  backend_kwargs={"indexpath": ""})
            var = list(ds.data_vars)[0]
            da  = ds[var]
            lats = da["latitude"].values.copy()
            lons = da["longitude"].values.copy()
            arr  = da.values.copy().astype(np.float32)
            ds.close()
        finally:
            os.unlink(tmp)

        # Standardize orientation
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            arr  = arr[::-1, :]

        # Crop to Indiana
        lat_m = (lats >= BBOX["lat_min"]) & (lats <= BBOX["lat_max"])
        lon_m = (lons >= BBOX["lon_min"]) & (lons <= BBOX["lon_max"])
        arr   = arr[np.ix_(lat_m, lon_m)]
        lats  = lats[lat_m]
        lons  = lons[lon_m]

        arr = np.where(arr < 0, 0.0, arr)
        return arr, lats, lons

    except Exception as e:
        log.debug("Failed to read %s: %s", key, e)
        return None


# ── Steps 2 + 3: Spatial labeling ─────────────────────────────────────────────
def label_clusters(
    arr_mmhr: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """8-connectivity labeling + dilation-based almost-connected clustering."""
    binary = (arr_mmhr >= PRECIP_THRESH_MMHR).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.int32)

    km_per_px  = ((BBOX["lat_max"] - BBOX["lat_min"]) * 111.0) / len(lats)
    radius_px  = max(1, int(round(DILATION_KM / km_per_px)))
    struct8    = generate_binary_structure(2, 2)
    dilated    = binary_dilation(binary, structure=disk(radius_px)).astype(np.uint8)
    labeled_d, _ = ndlabel(dilated, structure=struct8)
    return np.where(binary > 0, labeled_d, 0).astype(np.int32)


# ── Step 4: Storm tracking ─────────────────────────────────────────────────────
def _centroid(labeled: np.ndarray, cid: int, lats, lons):
    r, c = np.where(labeled == cid)
    if len(r) == 0:
        return None
    return float(lats[r].mean()), float(lons[c].mean())


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a  = (math.sin((phi2 - phi1) / 2) ** 2
          + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def match_clusters(
    prev_labeled: Optional[np.ndarray],
    curr_labeled: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    prev_id_map: dict[int, int],   # local cluster ID → global event ID
    next_event_id: int,
) -> tuple[dict[int, int], int]:
    """Map current cluster labels to persistent event IDs."""
    curr_ids = [c for c in np.unique(curr_labeled) if c > 0]
    if not curr_ids:
        return {}, next_event_id

    if prev_labeled is None or not prev_id_map:
        return {cid: next_event_id + i for i, cid in enumerate(curr_ids)}, \
               next_event_id + len(curr_ids)

    id_map: dict[int, int] = {}
    prev_ids = [c for c in np.unique(prev_labeled) if c > 0]

    for cid in curr_ids:
        c_cent    = _centroid(curr_labeled, cid, lats, lons)
        best_prev = None
        best_j    = -1.0

        for pid in prev_ids:
            p_cent = _centroid(prev_labeled, pid, lats, lons)
            if c_cent is None or p_cent is None:
                continue
            if _haversine(*p_cent, *c_cent) > TRACKING_DIST_KM:
                continue
            m_c = curr_labeled == cid
            m_p = prev_labeled == pid
            inter   = int(np.sum(m_c & m_p))
            union   = int(np.sum(m_c | m_p))
            jaccard = inter / union if union > 0 else 0.0
            if jaccard >= JACCARD_MIN and jaccard > best_j:
                best_j    = jaccard
                best_prev = pid

        if best_prev is not None:
            id_map[cid] = prev_id_map[best_prev]
        else:
            id_map[cid]  = next_event_id
            next_event_id += 1

    return id_map, next_event_id


# ── Per-day processing ─────────────────────────────────────────────────────────
def process_day(
    day: date,
    active_storms: dict[int, ActiveStorm],
    next_event_id: int,
    atlas14_grids: dict,
    completed_events: list[dict],
) -> tuple[dict[int, ActiveStorm], int]:
    """Process one day's GRIB2 files. Mutates active_storms and completed_events."""
    fs   = s3fs.S3FileSystem(anon=True)
    prefix = f"{MRMS_BUCKET}/CONUS/{MRMS_FOLDER}/{day:%Y%m%d}/"
    try:
        keys = sorted(fs.ls(prefix))
    except FileNotFoundError:
        log.warning("%s: no files on noaa-mrms-pds", day)
        return active_storms, next_event_id

    if not keys:
        return active_storms, next_event_id

    prev_labeled: Optional[np.ndarray] = None
    prev_id_map:  dict[int, int]       = {}
    grid_lats: Optional[np.ndarray]    = None
    grid_lons: Optional[np.ndarray]    = None

    for key in keys:
        result = read_grib2_indiana(fs, key)
        if result is None:
            continue

        arr_mm, g_lats, g_lons = result
        if grid_lats is None:
            grid_lats = g_lats
            grid_lons = g_lons

        dt = _parse_timestamp(key)
        if dt is None:
            continue

        arr_in = (arr_mm / 25.4).astype(np.float32)   # mm → inches

        # Steps 2+3: cluster labeling
        curr_labeled = label_clusters(arr_mm, g_lats)

        # Step 4: track storms
        curr_id_map, next_event_id = match_clusters(
            prev_labeled, curr_labeled, g_lats, g_lons,
            prev_id_map, next_event_id,
        )

        # Update active storms
        active_cluster_ids = set(curr_id_map.values())   # global event IDs active this hour

        for event_id in active_cluster_ids:
            # Find which local cluster corresponds to this event_id
            local_ids = [k for k, v in curr_id_map.items() if v == event_id]
            if not local_ids:
                continue

            # Build pixel mask for this storm at this hour
            storm_mask = np.zeros(curr_labeled.shape, dtype=bool)
            for lid in local_ids:
                storm_mask |= (curr_labeled == lid)

            # Zero out pixels not in this storm
            hour_grid = np.where(storm_mask, arr_in, 0.0).astype(np.float32)

            if event_id not in active_storms:
                active_storms[event_id] = ActiveStorm(
                    event_id      = event_id,
                    start_dt      = dt,
                    last_active_dt= dt,
                )

            storm = active_storms[event_id]
            storm.last_active_dt = dt
            storm.dry_hours      = 0
            storm.hour_buffer.append(hour_grid)

            # Update footprint (union of all active pixel masks)
            if storm.footprint is None:
                storm.footprint = storm_mask.copy()
            else:
                storm.footprint |= storm_mask

            # Incremental RP check (optional: skip for speed; full check at storm end)

        # Advance dry_hours for storms not active this hour
        to_close: list[int] = []
        for event_id, storm in active_storms.items():
            if event_id not in active_cluster_ids:
                storm.dry_hours += 1
                if storm.dry_hours >= DRY_HOURS_END:
                    to_close.append(event_id)

        # Close finished storms
        for event_id in to_close:
            storm = active_storms.pop(event_id)
            rp    = classify_storm_rp(storm, atlas14_grids)

            if storm.footprint is not None:
                r_idx, c_idx = np.where(storm.footprint)
                centroid_lat = float(g_lats[r_idx].mean()) if grid_lats is not None and len(r_idx) > 0 else float("nan")
                centroid_lon = float(g_lons[c_idx].mean()) if grid_lons is not None and len(c_idx) > 0 else float("nan")
                n_pixels     = int(storm.footprint.sum())
            else:
                centroid_lat = centroid_lon = float("nan")
                n_pixels     = 0

            completed_events.append({
                "event_id":     event_id,
                "start_dt":     storm.start_dt,
                "end_dt":       storm.last_active_dt,
                "duration_hr":  max(1, int((storm.last_active_dt - storm.start_dt).total_seconds() / 3600) + 1),
                "centroid_lat": round(centroid_lat, 4),
                "centroid_lon": round(centroid_lon, 4),
                "n_pixels_max": n_pixels,
                "assigned_rp":  rp,
            })
            log.debug("Closed storm %d  RP=%s  start=%s", event_id, rp, storm.start_dt)

        prev_labeled = curr_labeled
        prev_id_map  = curr_id_map

    return active_storms, next_event_id


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _ckpt_key(day: date) -> str:
    return f"{S3_CHECKPOINT}{day:%Y%m%d}_done.flag"


def day_done(day: date) -> bool:
    return _key_exists(_ckpt_key(day))


def save_day_checkpoint(
    day: date,
    active_storms: dict[int, ActiveStorm],
    next_event_id: int,
    completed_events: list[dict],
) -> None:
    """Pickle active storm state + flush completed events to S3."""
    # Save active state
    state = {"active_storms": active_storms, "next_event_id": next_event_id}
    _put_bytes(pickle.dumps(state), f"{S3_CHECKPOINT}{day:%Y%m%d}_state.pkl")

    # Append completed events (merge with existing)
    _tmp = _get_parquet(S3_EVENTS_KEY); existing = _tmp if _tmp is not None else pd.DataFrame()
    new_rows  = pd.DataFrame(completed_events)
    if not new_rows.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True) \
                     .drop_duplicates("event_id")
        _put_parquet(combined, S3_EVENTS_KEY)

    # Mark day done
    _put_bytes(b"done", _ckpt_key(day))
    log.info("  ✓ day %s done  |  active storms: %d  |  completed today: %d",
             day, len(active_storms), len(completed_events))


def load_state(day: date) -> Optional[dict]:
    """Load active storm state from the previous day's checkpoint."""
    raw = _get_bytes(f"{S3_CHECKPOINT}{day:%Y%m%d}_state.pkl")
    if raw is None:
        return None
    return pickle.loads(raw)


# ── Summary + map ──────────────────────────────────────────────────────────────
def build_summary(events: pd.DataFrame, n_years: float) -> pd.DataFrame:
    rows = []
    for rp in RETURN_PERIODS:
        # Cumulative count: events classified as this RP or higher
        observed = int((events["assigned_rp"].dropna() >= rp).sum()) \
                   if "assigned_rp" in events.columns else 0
        expected = n_years / rp
        ratio    = observed / expected if expected > 0 else float("nan")
        status   = (
            "✓"           if not np.isnan(ratio) and RATIO_OK_LO <= ratio <= RATIO_OK_HI else
            "✗ over-count" if not np.isnan(ratio) and ratio > RATIO_OK_HI else
            "✗ under-count"
        )
        rows.append({
            "return_period":  f"P{rp}",
            "observed":       observed,
            "expected_per_6yr": round(expected * 6, 1),
            "observed_per_yr": round(observed / n_years, 2),
            "expected_per_yr": round(1 / rp, 4),
            "ratio":          round(ratio, 2) if not np.isnan(ratio) else None,
            "status":         status,
        })
    return pd.DataFrame(rows)


def plot_storm_map(events: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, ax = plt.subplots(figsize=(8, 9))
        cmap   = {2: "lightblue", 10: "steelblue", 25: "green",
                  50: "gold", 100: "orange", 500: "red", 1000: "darkred"}
        for rp in sorted(RETURN_PERIODS):
            sub = events[events["assigned_rp"] == rp].dropna(subset=["centroid_lat"])
            if sub.empty:
                continue
            ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
                       c=cmap.get(rp, "grey"), s=12, alpha=0.6,
                       label=f"P{rp} ({len(sub)} storms)", zorder=3)

        ax.set_xlim(-88.6, -84.6)
        ax.set_ylim(37.6, 42.0)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Indiana Storm Events by Return Period\n(centroid locations)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        _put_bytes(buf.getvalue(), S3_OUT + "indiana_storm_map.png", "image/png")
        log.info("  → s3://%s/%s", PIPELINE_BUCKET, S3_OUT + "indiana_storm_map.png")
    except Exception as e:
        log.warning("Map plot failed: %s", e)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start",          default=DEFAULT_START)
    parser.add_argument("--end",            default=DEFAULT_END)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reprocess-date", default=None)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    n_years = (
        pd.Timestamp(str(end), tz="UTC") - pd.Timestamp(str(start), tz="UTC")
    ).days / 365.25

    if args.aggregate_only:
        _tmp = _get_parquet(S3_EVENTS_KEY); events = _tmp if _tmp is not None else pd.DataFrame()
        if events.empty:
            log.error("No storm_events.parquet found — run processing first.")
            return
        summary = build_summary(events, n_years)
        log.info("\n%s", summary.to_string(index=False))
        _put_csv(summary, S3_OUT + "indiana_storm_summary.csv")
        plot_storm_map(events)
        return

    # ── Build Atlas 14 grid (need MRMS grid coordinates first) ────────────────
    # Read first available GRIB2 file to get grid dimensions
    log.info("Probing MRMS grid for Indiana bounding box dimensions...")
    fs = s3fs.S3FileSystem(anon=True)
    probe_day = start
    grid_lats = grid_lons = None
    while grid_lats is None and probe_day <= end:
        prefix = f"{MRMS_BUCKET}/CONUS/{MRMS_FOLDER}/{probe_day:%Y%m%d}/"
        try:
            keys = sorted(fs.ls(prefix))
            if keys:
                result = read_grib2_indiana(fs, keys[0])
                if result is not None:
                    _, grid_lats, grid_lons = result
        except Exception:
            pass
        probe_day += timedelta(days=1)

    if grid_lats is None:
        log.error("Could not read any MRMS file to determine grid dimensions.")
        sys.exit(1)

    log.info("Indiana MRMS grid: %d lat × %d lon pixels", len(grid_lats), len(grid_lons))
    atlas14_grids = build_atlas14_grids(grid_lats, grid_lons)

    # ── Day loop ──────────────────────────────────────────────────────────────
    all_days = []
    d = start
    while d <= end:
        all_days.append(d)
        d += timedelta(days=1)

    done    = {d for d in all_days if day_done(d)}
    pending = [d for d in all_days if d not in done]

    if args.reprocess_date:
        force   = date.fromisoformat(args.reprocess_date)
        pending = sorted(set(pending) | {force})

    log.info("Days total: %d  |  done: %d  |  remaining: %d",
             len(all_days), len(done), len(pending))

    if not pending:
        log.info("All days processed — running aggregation.")
        events  = _get_parquet(S3_EVENTS_KEY) or pd.DataFrame()
        summary = build_summary(events, n_years)
        log.info("\n%s", summary.to_string(index=False))
        _put_csv(summary, S3_OUT + "indiana_storm_summary.csv")
        plot_storm_map(events)
        return

    # Resume active storm state from last completed day before the first pending day
    active_storms:    dict[int, ActiveStorm] = {}
    next_event_id:    int                    = 1
    completed_events: list[dict]             = []

    first_pending = sorted(pending)[0]
    prev_day      = first_pending - timedelta(days=1)
    state         = load_state(prev_day)
    if state:
        active_storms = state["active_storms"]
        next_event_id = state["next_event_id"]
        log.info("Resumed: %d active storms, next_event_id=%d", len(active_storms), next_event_id)
    else:
        log.info("Starting fresh (no prior state found).")

    t0 = time.time()
    n_processed = 0

    for day in sorted(pending):
        if _shutdown:
            log.info("Shutdown — saving state and exiting.")
            save_day_checkpoint(day - timedelta(days=1), active_storms,
                                next_event_id, completed_events)
            break

        log.info("── %s ─────────────────────────────────────────────────────", day)
        try:
            active_storms, next_event_id = process_day(
                day, active_storms, next_event_id,
                atlas14_grids, completed_events,
            )
            save_day_checkpoint(day, active_storms, next_event_id, completed_events)
            completed_events = []   # flushed to S3; reset for next day
            n_processed += 1
        except Exception as e:
            log.error("Day %s failed: %s — skipping", day, e)
            continue

        elapsed   = time.time() - t0
        rate      = n_processed / elapsed * 3600
        remaining = len(pending) - n_processed
        log.info("  rate: %.1f days/hr  |  ETA: %.1f hr",
                 rate, remaining / rate if rate > 0 else float("inf"))

    # Final aggregation
    log.info("Processing complete — building summary.")
    events  = _get_parquet(S3_EVENTS_KEY) or pd.DataFrame()
    summary = build_summary(events, n_years)
    log.info("\n%s", summary.to_string(index=False))
    _put_csv(summary,  S3_OUT + "indiana_storm_summary.csv")
    _put_csv(events,   S3_OUT + "indiana_storm_events.csv")
    plot_storm_map(events)


if __name__ == "__main__":
    main()
