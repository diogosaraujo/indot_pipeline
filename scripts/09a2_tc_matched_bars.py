"""09a2_tc_matched_bars.py

Matched-ARI version of 09a_tc_summary_figures.py.

Where 09a draws ONE figure per flood target with bars over ALL precipitation ARIs,
this draws a SINGLE figure showing only the MATCHED precip ARI per target:

    P10 → Q10 ,  P50 → Q50 ,  P100 → Q100      (nearest-MRMS source, global pool)

Each flood target shows four bars, with the same definitions as 09a:
    POD, FAR, CSI  → left y-axis   (bounded skill scores)
    FAF            → right y-axis  (false alarms per station-year, log scale)

    POD = TP/(TP+FN)   FAR = FP/(TP+FP)   CSI = TP/(TP+FP+FN)
    FAF = sum(FP) / sum(n_common_hours / 8766)

Reads:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_tc.parquet   (08c)

Writes:
    s3://<bucket>/<prefix>analysis/figures/tc_matched_summary_bars.{png,svg}

Usage:
    python scripts/09a2_tc_matched_bars.py
"""
from __future__ import annotations

import importlib.util
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from utils import load_config, write_bytes_to_s3


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Reuse 09a's loaders, metric constants and FAF formatting verbatim.
m09a = _load("tc_summary_09a", "09a_tc_summary_figures.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09a2_matched")

FLOW_RPS    = [10, 50, 100]
OUTPUT_STEM = "analysis/figures/tc_matched_summary_bars"


def make_figure(rows: list[dict], bucket: str, prefix: str) -> None:
    rps   = [r["rp"] for r in rows]
    x     = np.arange(len(rps))
    width = 0.20
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax2 = ax.twinx()

    # left-axis skill bars: POD, FAR, CSI
    for k, (metric, color) in enumerate(m09a.LEFT_METRICS):
        off  = (k - 1.5) * width
        vals = [r[metric] for r in rows]
        ax.bar(x + off, vals, width, color=color)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=11, rotation=90)

    # right-axis FAF bar (log), drawn from a small floor so bars are visible
    faf = np.array([r["FAF"] for r in rows])
    pos = faf[faf > 0]
    floor = max(pos.min() / 5.0, 1e-3) if pos.size else 1e-3
    ax2.set_yscale("log")
    off_faf = 1.5 * width
    ax2.bar(x + off_faf, np.where(faf > 0, faf, floor) - floor, width, bottom=floor,
            color=m09a.FAF_COLOR)
    for xi, v in zip(x + off_faf, faf):
        if v > 0:
            ax2.annotate(m09a._fmt(v), (xi, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=10, color=m09a.FAF_COLOR, rotation=90)
    ax2.set_ylim(floor, (pos.max() * 4 if pos.size else 1))

    ax.set_ylim(0, 1.18)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_ylabel("Skill score", fontsize=15)
    ax2.set_ylabel("FAF (false alarms / station-yr)", fontsize=15, color=m09a.FAF_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{r['rp']}\n(P{r['rp']})" for r in rows], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13, colors=m09a.FAF_COLOR)
    ax.set_xlabel("Flood target  (matched precipitation ARI)", fontsize=15)
    ax.grid(axis="y", ls=":", alpha=0.4)

    handles = [Patch(color=c, label=mname) for mname, c in m09a.LEFT_METRICS] + \
              [Patch(color=m09a.FAF_COLOR, label="FAF")]
    ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=13, framealpha=0.9)
    n_ev = "  ".join(f"Q{r['rp']}: {r['n_events']} ev" for r in rows)
    ax.set_title(f"Matched precip-ARI trigger (P = Q) — global pool, MRMS nearest\n"
                 f"{rows[0]['n_stations']} stations   |   {n_ev}", fontsize=15)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}.{ext}")
        log.info("Saved s3://%s/%s%s.%s", bucket, prefix, OUTPUT_STEM, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    df = m09a._read_parquet(bucket, f"{prefix}{m09a.INPUT_KEY}",
                            ["site_no", "source", "precip_rp_yr", "flow_rp_yr",
                             "tp", "fp", "fn", "n_flow_events", "n_common_hours"])
    df = df[df["source"] == m09a.SOURCE].copy()
    if df.empty:
        log.error("No %s rows in %s", m09a.SOURCE, m09a.INPUT_KEY)
        return

    rows: list[dict] = []
    for rp in FLOW_RPS:
        sub = df[(df["flow_rp_yr"] == rp) & (df["precip_rp_yr"] == rp)]
        if sub.empty:
            log.warning("No matched rows for P%d/Q%d — skipping", rp, rp)
            continue
        tp = float(sub["tp"].sum()); fp = float(sub["fp"].sum()); fn = float(sub["fn"].sum())
        station_years = float(sub.groupby("site_no")["n_common_hours"].first().sum()) / m09a.HOURS_PER_YEAR
        rows.append({
            "rp":         rp,
            "POD":        tp / (tp + fn + m09a.EPS),
            "FAR":        fp / (tp + fp + m09a.EPS),
            "CSI":        tp / (tp + fp + fn + m09a.EPS),
            "FAF":        fp / (station_years + m09a.EPS),
            "n_events":   int(sub.groupby("site_no")["n_flow_events"].first().sum()),
            "n_stations": int(sub["site_no"].nunique()),
        })
        log.info("Q%d (P%d): POD=%.2f FAR=%.2f CSI=%.2f FAF=%.2f/stn-yr",
                 rp, rp, rows[-1]["POD"], rows[-1]["FAR"], rows[-1]["CSI"], rows[-1]["FAF"])

    if not rows:
        log.error("No matched (P=Q) rows found.")
        return
    make_figure(rows, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
