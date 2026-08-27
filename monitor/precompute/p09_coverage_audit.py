"""p09 — which bridges cannot be alerted, and why.

A bridge is only monitored to the extent its thresholds exist and are usable.
Silence from the monitor can mean "the river is quiet" or "this bridge could
never fire" — this separates the two, per trigger, so the second case is a
countable list rather than an assumption.

Failure modes checked
---------------------
FLOW
  no_comid        no NHDPlus reach matched, so no NWM streamflow to read
  no_lp3          reach matched but no LP3 fit (short/failed record)
  degenerate_lp3  Q10 == Q50 == Q100 to within DEGENERATE_RATIO — a collapsed
                  fit. Worse than useless: flow >= Q100 passes on equality, so
                  these fire at 100-yr permanently.
  no_floor        the tier this bridge actually fires at is null

PRECIP
  no_atlas14      no Atlas-14 depths for the cell
  window_too_long round(Tc) exceeds the state buffer, so the trailing window can
                  never fill and the completeness guard suppresses it forever.
                  The 24 h Tc cap should drive this to zero.

Writes  monitor/analysis/coverage_audit.parquet   one row per bridge
        plus a printed summary

Usage:
    python monitor/precompute/p09_coverage_audit.py
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import config
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p09")

DEGENERATE_RATIO = 1.05      # Q100/Q10 below this is a collapsed fit
RPS = (10, 50, 100)


def audit(cfg: pd.DataFrame, state_hours: int) -> pd.DataFrame:
    d = cfg.copy()
    scour = d[config.SCOUR_COL].astype(bool)
    d["fire_rp"] = np.where(scour, config.FIRE_RP_SCOUR, config.FIRE_RP_OTHER)
    d["q_floor"] = np.where(scour, d["Q10_cfs"], d["Q50_cfs"])
    d["p_floor"] = np.where(scour, d["P10"], d["P50"])

    qcols = [f"Q{rp}_cfs" for rp in RPS]
    ratio = d["Q100_cfs"] / d["Q10_cfs"].replace(0, np.nan)

    d["flow_no_comid"] = d["comid"].isna()
    d["flow_no_lp3"] = (~d["flow_no_comid"]) & d[qcols].isna().all(axis=1)
    d["flow_degenerate"] = d[qcols].notna().all(axis=1) & (ratio < DEGENERATE_RATIO)
    d["flow_no_floor"] = d["q_floor"].isna() & (~d["flow_no_comid"]) & (~d["flow_no_lp3"])
    d["flow_dead"] = (d["flow_no_comid"] | d["flow_no_lp3"] | d["flow_no_floor"]
                      | d["flow_degenerate"])

    d["precip_no_atlas14"] = d[[f"P{rp}" for rp in RPS]].isna().all(axis=1)
    d["precip_window_too_long"] = d["tc_dur_hr"] > state_hours
    d["precip_no_floor"] = d["p_floor"].isna() & (~d["precip_no_atlas14"])
    d["precip_dead"] = (d["precip_no_atlas14"] | d["precip_window_too_long"]
                        | d["precip_no_floor"])

    d["unmonitored"] = d["flow_dead"] & d["precip_dead"]
    return d


def summarise(d: pd.DataFrame) -> None:
    n = len(d)

    def line(label, mask, note=""):
        c = int(mask.sum())
        sc = int((mask & d[config.SCOUR_COL].astype(bool)).sum())
        log.info("  %-24s %6d  (%4.1f%%)  scour-critical: %-4d %s",
                 label, c, c / n * 100, sc, note)

    log.info("=== %d over-water bridges ===", n)
    log.info("STREAMFLOW trigger")
    line("no COMID", d["flow_no_comid"], "no NWM reach to read")
    line("no LP3 fit", d["flow_no_lp3"])
    line("degenerate LP3", d["flow_degenerate"], "fires at 100-yr permanently")
    line("firing tier null", d["flow_no_floor"])
    line("-> cannot fire on flow", d["flow_dead"])

    log.info("PRECIPITATION trigger")
    line("no Atlas-14", d["precip_no_atlas14"])
    line("window > state buffer", d["precip_window_too_long"], "can never fill")
    line("firing tier null", d["precip_no_floor"])
    line("-> cannot fire on precip", d["precip_dead"])

    log.info("BOTH")
    line("UNMONITORED", d["unmonitored"], "no trigger can ever fire")

    if int(d["flow_degenerate"].sum()):
        log.info("\nDegenerate LP3 reaches (these fire spuriously):")
        g = (d[d["flow_degenerate"]]
             .groupby("comid")
             .agg(bridges=("bridge_id", "size"), q10=("Q10_cfs", "first"),
                  q100=("Q100_cfs", "first")))
        for comid, r in g.iterrows():
            log.info("   COMID %-10d %d bridge(s)  Q10 %.2f  Q100 %.2f",
                     int(comid), int(r.bridges), r.q10, r.q100)
    if int(d["unmonitored"].sum()):
        log.info("\nUnmonitored bridges:")
        for _, r in d[d["unmonitored"]].head(20).iterrows():
            why = []
            for c, lbl in (("flow_no_comid", "no comid"), ("flow_no_lp3", "no LP3"),
                           ("flow_degenerate", "degenerate LP3"),
                           ("precip_no_atlas14", "no Atlas-14"),
                           ("precip_window_too_long", "Tc window too long")):
                if r[c]:
                    why.append(lbl)
            log.info("   %-20s %s", r["bridge_id"], " + ".join(why))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-hours", type=int, default=config.STATE_HOURS)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    k = config.keys()
    cfg = read_parquet(k["bucket"], k["config"])
    d = audit(cfg, args.state_hours)
    summarise(d)

    if not args.no_upload:
        cols = ["bridge_id", config.ASSET_COL, "lat", "lon", "comid",
                config.SCOUR_COL, "tc_dur_hr", "fire_rp",
                *[c for c in d.columns if c.startswith(("flow_", "precip_"))],
                "unmonitored"]
        out = d[[c for c in cols if c in d.columns]]
        dest = f"{k['prefix']}monitor/analysis/coverage_audit.parquet"
        write_parquet(out.reset_index(drop=True), k["bucket"], dest)
        log.info("\nWrote s3://%s/%s", k["bucket"], dest)


if __name__ == "__main__":
    main()
