"""11_derive_stage.py

Derive water-surface stage (stage_m) for all NWM products by applying
HAND-based Synthetic Rating Curves (SRC) from HYDRO_TBL_1D.nc.

Reads the three NWM parquets written by script 10, adds a stage_m column
to each, and overwrites them in S3.

The SRC file (HYDRO_TBL_1D.nc) is read anonymously from the public
noaa-nwm-pds bucket.  Its path is configured in config.yaml under
nwm.src_s3_bucket / nwm.src_s3_key.  Run once after script 10 completes.

Reads:
    s3://<bucket>/<prefix>nwm/retrospective.parquet
    s3://<bucket>/<prefix>nwm/analysis_assim.parquet
    s3://<bucket>/<prefix>nwm/open_loop.parquet

Writes (same paths, overwrites in place):
    s3://<bucket>/<prefix>nwm/retrospective.parquet   (adds stage_m)
    s3://<bucket>/<prefix>nwm/analysis_assim.parquet  (adds stage_m)
    s3://<bucket>/<prefix>nwm/open_loop.parquet       (adds stage_m)

Schema after this script (all three products):
    site_no, comid, datetime_utc,
    streamflow_cms, velocity_ms, [nudge_cms,] stage_m
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
import xarray as xr

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("11_stage")


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def load_src_interpolators(comids: list[int], bucket: str, key: str) -> dict:
    """Load HYDRO_TBL_1D.nc from S3 and build per-COMID stage interpolators.

    Returns dict: comid (int) → callable(streamflow_cms) → stage_m.
    COMIDs not in the SRC file are silently omitted.
    """
    fs = s3fs.S3FileSystem(anon=True)
    try:
        with fs.open(f"{bucket}/{key}", "rb") as f:
            data = f.read()
        ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    except Exception as e:
        log.error("Could not load SRC from s3://%s/%s: %s", bucket, key, e)
        return {}

    comid_set   = set(comids)
    feature_ids = ds["feature_id"].values.astype(int)

    # Two variable conventions depending on NWM version stored in the file
    if "Stage_1" in ds:
        n_pts     = sum(1 for v in ds.data_vars if v.startswith("Stage_"))
        stage_arr = np.stack([ds[f"Stage_{i}"].values    for i in range(1, n_pts + 1)], axis=1)
        q_arr     = np.stack([ds[f"Discharge_{i}"].values for i in range(1, n_pts + 1)], axis=1)
    elif "stage_ht_NQ" in ds:
        stage_arr = ds["stage_ht_NQ"].values      # (n_comids, n_pts)
        q_arr     = ds["discharge_ht_NQ"].values
    else:
        log.error("SRC file has unrecognised variable layout")
        return {}

    interpolators: dict = {}
    for idx, fid in enumerate(feature_ids):
        if int(fid) not in comid_set:
            continue
        q_pts = q_arr[idx]
        h_pts = stage_arr[idx]
        valid = np.isfinite(q_pts) & np.isfinite(h_pts)
        if valid.sum() < 2:
            continue
        q_v, h_v = q_pts[valid], h_pts[valid]
        order    = np.argsort(q_v)
        q_s, h_s = q_v[order], h_v[order]
        interpolators[int(fid)] = lambda q, _q=q_s, _h=h_s: float(
            np.interp(q, _q, _h, left=_h[0], right=_h[-1])
        )

    log.info("SRC: built interpolators for %d / %d COMIDs", len(interpolators), len(comids))
    return interpolators


def apply_stage(df: pd.DataFrame, interp: dict) -> pd.DataFrame:
    """Add stage_m column; NaN for COMIDs missing from SRC."""
    df["stage_m"] = [
        interp[c](q) if c in interp else np.nan
        for c, q in zip(df["comid"], df["streamflow_cms"])
    ]
    return df


def main() -> None:
    cfg        = load_config()
    bucket     = cfg["aws"]["output_bucket"]
    prefix     = cfg["aws"]["output_prefix"]
    nwm_cfg    = cfg.get("nwm", {})
    src_bucket = nwm_cfg.get("src_s3_bucket", "noaa-nwm-pds")
    src_key    = nwm_cfg.get("src_s3_key", "")

    if not src_key:
        log.error(
            "nwm.src_s3_key is not set in config.yaml — cannot derive stage.\n"
            "Find the path with:\n"
            "  aws s3 ls s3://noaa-nwm-pds/nwm.20240101/domain/ --no-sign-request\n"
            "Then set nwm.src_s3_key in config.yaml and re-run."
        )
        return

    parquet_keys = {
        "retrospective": f"{prefix}nwm/retrospective.parquet",
        "analysis_assim": f"{prefix}nwm/analysis_assim.parquet",
        "open_loop":      f"{prefix}nwm/open_loop.parquet",
    }

    # Load all available parquets and collect unique COMIDs across all products
    frames: dict[str, pd.DataFrame] = {}
    all_comids: set[int] = set()
    for name, key in parquet_keys.items():
        try:
            df = _read_parquet_s3(bucket, key)
            frames[name] = df
            all_comids.update(df["comid"].astype(int).unique())
            log.info("Loaded %s: %d rows, %d COMIDs", name, len(df), df["comid"].nunique())
        except Exception as e:
            log.warning("Could not read %s: %s — skipping", name, e)

    if not frames:
        log.error("No parquets loaded — aborting.")
        return

    log.info("Loading HAND SRC from s3://%s/%s ...", src_bucket, src_key)
    interp = load_src_interpolators(list(all_comids), src_bucket, src_key)
    if not interp:
        log.error("SRC interpolators empty — stage_m will be NaN for all rows.")

    for name, df in frames.items():
        log.info("Applying stage to %s (%d rows)...", name, len(df))
        df = apply_stage(df, interp)
        write_parquet_to_s3(df, bucket, parquet_keys[name])
        n_valid = df["stage_m"].notna().sum()
        log.info(
            "%s: wrote %d rows (%d with valid stage_m, %.1f%%)",
            name, len(df), n_valid, 100 * n_valid / max(len(df), 1),
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
