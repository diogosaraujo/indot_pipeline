"""p04 — assemble the single monitor config the Lambdas read.

Merges p01 (COMID + Tc), p02 (Atlas-14 cell DDF), p03 (retro-LP3 Q) into:

    s3://<bucket>/<prefix>monitor/bridge_monitor_config.parquet
        bridge_id, Asset Name, lat, lon, comid, over_waterway, scour_critical,
        tc_hr, tc_dur_hr, P10, P50, P100, Q10_cfs, Q50_cfs, Q100_cfs

P{rp} = Atlas-14 depth at the bridge's accumulation duration, by log-log
interpolation across the published DDF durations — identical to
scripts/08c_tc_trigger_analysis.py::depth_at_duration.

Tc CAP (2026-08-26 decision, reversing 2026-08-04): Kirpich Tc runs to 2121 h on
continental main stems, and any bridge whose Tc exceeds the 48 h state window
could never fill its accumulation — 12% of the fleet had a permanently dead
precipitation trigger. Tc is now capped at TC_CAP_HOURS, and the cap applies to
BOTH halves of the comparison: the trailing window AND the Atlas-14 duration the
depth is read at. Clipping only the window would test 24 h of rain against a
multi-day design depth. Native Tc is retained as tc_dur_native_hr, and tc_capped
flags the affected bridges.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import config, pre_key

# Cap on the trailing accumulation window AND the Atlas-14 duration it is
# compared against. 24 h keeps the window inside the 48 h state buffer with
# room for backfill, and matches the flashy-basin reading of a point gauge.
TC_CAP_HOURS = 24
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-cap", type=int, default=TC_CAP_HOURS,
                    help="cap the accumulation duration (and the Atlas-14 duration "
                         "read against it) at this many hours")
    args = ap.parse_args()

    b, prefix = config.bucket_prefix()
    tc = read_parquet(b, pre_key("bridge_comid_tc.parquet"))
    ddf = read_parquet(b, pre_key("bridge_atlas14.parquet"))
    try:
        lp3 = read_parquet(b, pre_key("bridge_comid_lp3.parquet"))
    except Exception:
        lp3 = pd.DataFrame(columns=["comid", "Q10_cfs", "Q50_cfs", "Q100_cfs"])

    tc["bridge_id"] = tc["bridge_id"].astype(str)
    tc["comid"] = pd.to_numeric(tc["comid"], errors="coerce").astype("Int64")
    tc["tc_dur_native_hr"] = (pd.to_numeric(tc["tc_hr"], errors="coerce")
                              .round().fillna(1).astype(int).clip(lower=1))
    tc["tc_dur_hr"] = tc["tc_dur_native_hr"].clip(upper=args.tc_cap)
    tc["tc_capped"] = tc["tc_dur_native_hr"] > args.tc_cap
    n_cap = int(tc["tc_capped"].sum())
    log.info("Tc cap %d h: %d of %d bridges capped (%.1f%%); native max %d h",
             args.tc_cap, n_cap, len(tc), n_cap / max(len(tc), 1) * 100,
             int(tc["tc_dur_native_hr"].max()))
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
            "tc_dur_native_hr", "tc_capped",
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
