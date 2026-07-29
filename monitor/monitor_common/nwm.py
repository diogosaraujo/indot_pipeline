"""Live NWM operational channel_rt access for the monitor.

Reads streamflow AND velocity for a whole set of COMIDs from a single CONUS
channel_rt file (~12.6 MB) via h5py byte-range over anonymous s3fs — the same
pattern as scripts/10_download_nwm.py::_extract_one_hour, generalized to many
reaches at once.

channel_rt variables are PACKED integers (scale_factor=0.01, add_offset=0);
h5py returns the raw ints (xarray would auto-apply), so we unpack by hand.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .s3io import anon_fs

log = logging.getLogger("monitor.nwm")


def key_for(ts: pd.Timestamp, product: str) -> str:
    return (f"{config.NWM_BUCKET}/nwm.{ts:%Y%m%d}/{product}/"
            f"nwm.t{ts.hour:02d}z.{product}.channel_rt.tm00.conus.nc")


def latest_available_hour(now: pd.Timestamp, product: str,
                          search_back: int | None = None) -> pd.Timestamp | None:
    fs = anon_fs()
    sb = config.NWM_SEARCH_BACK if search_back is None else search_back
    h0 = now.tz_convert("UTC").floor("h")
    for k in range(sb + 1):
        ts = h0 - pd.Timedelta(hours=k)
        if fs.exists(key_for(ts, product)):
            return ts
    return None


def _decode(dset, rows_sorted: np.ndarray) -> np.ndarray:
    sf = float(np.asarray(dset.attrs.get("scale_factor", 1.0)).ravel()[0])
    ao = float(np.asarray(dset.attrs.get("add_offset", 0.0)).ravel()[0])
    return dset[rows_sorted].astype(float) * sf + ao


def read_comids(ts: pd.Timestamp, product: str, comids: np.ndarray) -> pd.DataFrame | None:
    """streamflow (m3/s) + velocity (m/s) for each COMID at valid hour `ts`.

    Returns a DataFrame indexed by comid with columns streamflow_cms, velocity_ms;
    COMIDs absent from the NWM domain get NaN. Returns None if the file is missing.
    """
    import h5py

    fs = anon_fs()
    key = key_for(ts, product)
    comids = np.asarray(comids, dtype=np.int64)
    try:
        with fs.open(key, "rb") as fobj, h5py.File(fobj, "r") as h:
            ids = h["feature_id"][:].astype(np.int64)
            order = np.argsort(ids)
            ids_sorted = ids[order]
            pos = np.searchsorted(ids_sorted, comids)
            pos_clipped = np.clip(pos, 0, len(ids_sorted) - 1)
            rows = order[pos_clipped]
            valid = ids[rows] == comids                      # COMID present in domain

            q = np.full(len(comids), np.nan)
            v = np.full(len(comids), np.nan)
            if valid.any():
                rvalid = rows[valid]
                sort_idx = np.argsort(rvalid)                # h5py fancy-index needs ascending
                rsorted = rvalid[sort_idx]
                qv = _decode(h["streamflow"], rsorted)
                vv = _decode(h["velocity"], rsorted)
                unsort = np.empty_like(sort_idx)
                unsort[sort_idx] = np.arange(len(sort_idx))
                q_valid = qv[unsort]
                v_valid = vv[unsort]
                q_valid[q_valid < 0] = np.nan
                v_valid[v_valid < 0] = np.nan
                q[valid] = q_valid
                v[valid] = v_valid
    except Exception as e:  # noqa: BLE001
        log.warning("NWM read failed for %s: %s", key, e)
        return None

    return pd.DataFrame({"streamflow_cms": q, "velocity_ms": v},
                        index=pd.Index(comids, name="comid"))
