"""Live MRMS MultiSensor QPE 1-h Pass2 access for the monitor.

Pass2 is the gauge-corrected pass and typically lags real time by ~1 h, so the
poller searches back a few hours for the newest published valid hour.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import timezone

import numpy as np
import pandas as pd

from . import config
from .grib import apply_units, canonicalize_mrms_grid, decompress_gz, open_mrms_grib
from .s3io import anon_fs

log = logging.getLogger("monitor.mrms")


def key_for(ts: pd.Timestamp) -> str:
    d = ts.strftime("%Y%m%d")
    fn = f"MRMS_{config.MRMS_FOLDER}_{d}-{ts.hour:02d}0000.grib2.gz"
    return f"{config.MRMS_BUCKET}/CONUS/{config.MRMS_FOLDER}/{d}/{fn}"


def latest_available_hour(now: pd.Timestamp, search_back: int | None = None) -> pd.Timestamp | None:
    """Newest valid hour (top of hour, UTC) whose file exists on S3."""
    fs = anon_fs()
    sb = config.MRMS_SEARCH_BACK if search_back is None else search_back
    h0 = now.tz_convert("UTC").floor("h")
    for k in range(sb + 1):
        ts = h0 - pd.Timedelta(hours=k)
        if fs.exists(key_for(ts)):
            return ts
    return None


def _nearest_indices(grid_sorted_asc: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index into a sorted-ascending 1-D grid of the nearest cell for each point.

    Uses searchsorted (O(n log m)) instead of a full broadcast, so sampling
    ~17k bridges against the ~3500x7000 CONUS grid stays light on memory.
    """
    pos = np.searchsorted(grid_sorted_asc, points)
    pos = np.clip(pos, 1, len(grid_sorted_asc) - 1)
    left = grid_sorted_asc[pos - 1]
    right = grid_sorted_asc[pos]
    choose_left = (points - left) <= (right - points)
    return np.where(choose_left, pos - 1, pos)


def read_grid(ts: pd.Timestamp):
    """Return (arr_2d, lats_desc, lons_asc) for one valid hour, or None."""
    fs = anon_fs()
    key = key_for(ts)
    try:
        with fs.open(key, "rb") as f:
            raw = f.read()
    except Exception as e:  # noqa: BLE001
        log.warning("MRMS fetch failed for %s: %s", key, e)
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(decompress_gz(raw))
            tmp_path = tmp.name
        ds = open_mrms_grib(tmp_path)
        var = list(ds.data_vars)[0]
        arr, lats, lons = canonicalize_mrms_grid(ds[var])
        arr = apply_units(arr, kind="qpe", units="in")
        return arr, lats, lons
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def subset(grid, bbox) -> tuple:
    """Clip a CONUS grid to a lat/lon box — the state store keeps only Indiana.

    A CONUS grid is ~98 MB; the Indiana window is ~0.5 MB, which is what makes
    keeping an hourly grid history affordable at all.
    """
    arr, glat_desc, glon_asc = grid
    la0, la1 = bbox["lat"]; lo0, lo1 = bbox["lon"]
    r = np.where((glat_desc >= la0) & (glat_desc <= la1))[0]
    c = np.where((glon_asc >= lo0) & (glon_asc <= lo1))[0]
    sub = np.asarray(arr[np.ix_(r, c)], dtype=np.float32)
    return (np.where(np.isfinite(sub), sub, 0.0).astype(np.float32),
            glat_desc[r].astype(np.float32), glon_asc[c].astype(np.float32))


def sample_from_grid(grid, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Nearest-cell values for each point, from a grid already in memory."""
    arr, glat_desc, glon_asc = grid
    # canonicalize returns lats DESCENDING; flip to ascending for searchsorted.
    glat_asc = glat_desc[::-1]
    ilat_asc = _nearest_indices(glat_asc, np.asarray(lats, float))
    ilat = (len(glat_asc) - 1) - ilat_asc               # map back to descending order
    ilon = _nearest_indices(glon_asc, np.asarray(lons, float))
    vals = arr[ilat, ilon]
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return vals.astype(float)


def sample_points(ts: pd.Timestamp, lats: np.ndarray, lons: np.ndarray) -> np.ndarray | None:
    """Hourly MRMS 1-h QPE (inches) at the cell nearest each (lat, lon).

    Missing/negative sentinels become 0.0 (dry), matching the study's precip
    fill convention (08_trigger_analysis: missing precip -> 0).
    """
    got = read_grid(ts)
    return None if got is None else sample_from_grid(got, lats, lons)
