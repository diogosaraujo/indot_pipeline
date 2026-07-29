"""p04 — assemble the single monitor config the Lambdas read.

Merges p01 (COMID + Tc), p02 (Atlas-14 cell DDF), p03 (retro-LP3 Q) into:

    s3://<bucket>/<prefix>monitor/bridge_monitor_config.parquet
        bridge_id, Asset Name, lat, lon, comid, over_waterway, scour_critical,
        tc_hr, tc_dur_hr, P10, P50, P100, Q10_cfs, Q50_cfs, Q100_cfs

P{rp} = Atlas-14 depth at the bridge's round(Tc)-hour accumulation duration,
by log-log interpolation across the published DDF durations — identical to
scripts/08c_tc_trigger_analysis.py::depth_at_duration.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from common import config, pre_key
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p04")

RPS = config.SEVERITY_RPS       # [10, 50, 100]


def loglog(x, y, xq: float) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    o = np.argsort(x)
    return float(np.exp(np.interp(np.log(xq), np.log(x[o]), np.log(y[o]))))


def main() -> None:
    b, prefix = config.bucket_prefix()
    tc = read_parquet(b, pre_key("bridge_comid_tc.parquet"))
    ddf = read_parquet(b, pre_key("bridge_atlas14.parquet"))
    try:
        lp3 = read_parquet(b, pre_key("bridge_comid_lp3.parquet"))
    except Exception:
        lp3 = pd.DataFrame(columns=["comid", "Q10_cfs", "Q50_cfs", "Q100_cfs"])

    tc["bridge_id"] = tc["bridge_id"].astype(str)
    tc["comid"] = pd.to_numeric(tc["comid"], errors="coerce").astype("Int64")
    tc["tc_dur_hr"] = (pd.to_numeric(tc["tc_hr"], errors="coerce")
                       .round().fillna(1).astype(int).clip(lower=1))
    tc["cell"] = tc["lat"].round(3).astype(str) + "_" + tc["lon"].round(3).astype(str)

    # ── P10/P50/P100 at Tc duration, computed once per (cell, tc_dur) ─────────
    ddf["cell"] = ddf["cell"].astype(str)
    ddf_by_cell = {c: g for c, g in ddf.groupby("cell")}
    p_cache: dict[tuple, dict] = {}
    for cell, dur in tc[["cell", "tc_dur_hr"]].drop_duplicates().itertuples(index=False):
        g = ddf_by_cell.get(cell)
        vals = {}
        if g is not None:
            for rp in RPS:
                rows = g[g["return_period_yr"] == rp].dropna(subset=["depth_in"])
                vals[f"P{rp}"] = loglog(rows["duration_hr"].to_numpy(),
                                        rows["depth_in"].to_numpy(), float(dur))
        else:
            vals = {f"P{rp}": np.nan for rp in RPS}
        p_cache[(cell, dur)] = vals

    for rp in RPS:
        tc[f"P{rp}"] = [p_cache[(c, d)].get(f"P{rp}", np.nan)
                        for c, d in zip(tc["cell"], tc["tc_dur_hr"])]

    # ── Q by COMID ───────────────────────────────────────────────────────────
    lp3["comid"] = pd.to_numeric(lp3["comid"], errors="coerce").astype("Int64")
    qcols = [c for c in ("Q10_cfs", "Q50_cfs", "Q100_cfs") if c in lp3.columns]
    out = tc.merge(lp3[["comid", *qcols]], on="comid", how="left")

    keep = ["bridge_id", config.ASSET_COL, "lat", "lon", "comid",
            config.WATERWAY_COL, config.SCOUR_COL, "tc_hr", "tc_dur_hr",
            *[f"P{rp}" for rp in RPS], *qcols]
    out = out[[c for c in keep if c in out.columns]].drop_duplicates("bridge_id")
    for rp in RPS:
        if f"Q{rp}_cfs" not in out.columns:
            out[f"Q{rp}_cfs"] = np.nan

    dest = config.keys()["config"]
    write_parquet(out.reset_index(drop=True), b, dest)
    log.info("Wrote s3://%s/%s", b, dest)
    log.info("  bridges=%d  scour=%d  with_comid=%d  with_P10=%d  with_Q10=%d",
             len(out), int(out[config.SCOUR_COL].sum()), int(out["comid"].notna().sum()),
             int(out["P10"].notna().sum()), int(out["Q10_cfs"].notna().sum()))


if __name__ == "__main__":
    main()
