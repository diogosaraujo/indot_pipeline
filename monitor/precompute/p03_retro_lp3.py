"""p03 — NWM-Retrospective v3.0 LP3 flow quantiles (Q10/Q50/Q100) per bridge COMID.

Reuses the Bulletin-17C LP3 machinery exactly as scripts/04c_nwm_regression_flows.py
and scripts/visualize_bridge_event.py: extract water-year annual maxima from the
public retrospective Zarr, fit LP3 (fit_lp3 / lp3_quantile from 04b).

COMIDs are processed in batches (one Zarr bulk-load per batch) and checkpointed,
so a run that dies mid-way resumes where it left off. Expect a few hours for all
unique over-water COMIDs — this is a one-time precompute.

Reads   monitor/precompute/bridge_comid_tc.parquet  (for the COMID list + a lat)
Writes  s3://<bucket>/<prefix>monitor/precompute/bridge_comid_lp3.parquet
    comid, Q10_cfs, Q50_cfs, Q100_cfs, n_years, wy_start, wy_end, method
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import config, load_script, pre_key
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p03")

RPS = [10, 50, 100]
IN = pre_key("bridge_comid_tc.parquet")
OUT = pre_key("bridge_comid_lp3.parquet")

m04c = load_script("nwm_lp3_04c", "04c_nwm_regression_flows.py")
b04b = m04c.b04b


def open_retro(nwm10):
    """Open the retrospective Zarr ONCE; return (streamflow_dataarray, available_comids)."""
    import s3fs
    import xarray as xr
    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3fs.S3Map(nwm10.RETRO_ZARR, s3=fs), consolidated=True)
    return ds["streamflow"], set(ds["feature_id"].values.astype(int))


def load_batch_streamflow(sf_da, available: set, comids: list[int],
                          t0: pd.Timestamp, t1: pd.Timestamp) -> dict[int, pd.DataFrame]:
    """Per-COMID hourly streamflow (cms) for a batch — streamflow ONLY (no velocity),
    which halves the Zarr I/O versus extract_retrospective. Same schema annual_max_series wants."""
    ok = [c for c in comids if c in available]
    missing = [c for c in comids if c not in available]
    if missing:
        log.warning("  %d COMIDs not in NWM domain: %s", len(missing), missing[:10])
    if not ok:
        return {}
    sub = sf_da.sel(feature_id=ok).sel(time=slice(t0, t1)).load()
    times = pd.to_datetime(sub["time"].values).tz_localize("UTC")
    out: dict[int, pd.DataFrame] = {}
    for c in ok:
        out[c] = pd.DataFrame({"datetime_utc": times,
                               "streamflow_cms": sub.sel(feature_id=c).values})
    return out


def fit_one(retro_site: pd.DataFrame, lat: float) -> dict | None:
    ann = m04c.annual_max_series(retro_site)         # water-year maxima (cfs), completeness-filtered
    if len(ann) < b04b.MIN_YEARS:
        return None
    params = b04b.fit_lp3(np.log10(ann.values), lat)
    q = {rp: float(b04b.lp3_quantile(rp, params["mean_log"], params["std_log"],
                                     params["skew"])) for rp in RPS}
    return {"Q10_cfs": q[10], "Q50_cfs": q[50], "Q100_cfs": q[100],
            "n_years": int(len(ann)), "wy_start": int(ann.index.min()),
            "wy_end": int(ann.index.max()),
            "method": params.get("fitting_method", "MOM")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=200, help="COMIDs per Zarr bulk-load")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    b, _ = config.bucket_prefix()
    src = read_parquet(b, IN)
    src["comid"] = pd.to_numeric(src["comid"], errors="coerce").astype("Int64")
    src = src.dropna(subset=["comid"])
    lat_by_comid = (src.dropna(subset=["lat"]).groupby("comid")["lat"].first().to_dict())
    comids = sorted({int(c) for c in src["comid"].dropna().unique()})
    if args.limit:
        comids = comids[:args.limit]
    log.info("Unique over-water COMIDs: %d", len(comids))

    done: dict[int, dict] = {}
    try:
        prev = read_parquet(b, OUT)
        done = {int(r["comid"]): r for r in prev.to_dict("records")}
        log.info("Resuming: %d COMIDs already fitted", len(done))
    except Exception:
        pass

    nwm10 = m04c._load_nwm10()
    todo = [c for c in comids if c not in done]
    results = list(done.values())

    log.info("Opening retrospective Zarr once (streamflow only)...")
    sf_da, available = open_retro(nwm10)
    t0 = pd.Timestamp(nwm10.RETRO_START).tz_localize(None)
    t1 = pd.Timestamp(nwm10.RETRO_END).tz_localize(None)

    for bstart in range(0, len(todo), args.batch):
        batch = todo[bstart:bstart + args.batch]
        log.info("Batch %d-%d/%d: loading streamflow for %d COMIDs...",
                 bstart, bstart + len(batch), len(todo), len(batch))
        try:
            series = load_batch_streamflow(sf_da, available, batch, t0, t1)
        except Exception as e:  # noqa: BLE001
            log.error("Batch load failed (%s) — skipping batch", e)
            continue
        for c in batch:
            sub = series.get(c)
            if sub is None or sub.empty:
                continue
            try:
                fit = fit_one(sub, float(lat_by_comid.get(c, 40.0)))
            except Exception as e:  # noqa: BLE001
                log.debug("COMID %d fit failed: %s", c, e)
                fit = None
            if fit:
                results.append({"comid": int(c), **fit})
        write_parquet(_finalize(pd.DataFrame(results)), b, OUT)
        log.info("Checkpoint: %d COMIDs fitted so far", len(results))

    out = _finalize(pd.DataFrame(results))
    write_parquet(out, b, OUT)
    log.info("Wrote %s: %d COMIDs with LP3 Q10/Q50/Q100", OUT, len(out))


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates("comid").copy()
    df["comid"] = df["comid"].astype("int64")
    return df.sort_values("comid").reset_index(drop=True)


if __name__ == "__main__":
    main()
