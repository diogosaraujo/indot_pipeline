"""09a2_tc_matched_bars.py

Matched-ARI version of 09a_tc_summary_figures.py.

09a draws ONE figure per flood target with bars over ALL precipitation ARIs.  This
draws the SAME thing — one figure per flood target — but keeps only the MATCHED
precip ARI:

    Q10 figure → P10 bars ,  Q50 figure → P50 bars ,  Q100 figure → P100 bars
    (nearest-MRMS source, accumulation at the station Kirpich Tc, global pool)

Each figure shows four bars, identical definitions to 09a:
    POD, FAR, CSI  → left y-axis   (bounded skill scores)
    FAF            → right y-axis  (false alarms per station-year, log scale)

    POD = TP/(TP+FN)   FAR = FP/(TP+FP)   CSI = TP/(TP+FP+FN)
    FAF = sum(FP) / sum(n_common_hours / 8766)

Reads:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_tc.parquet   (08c)

Writes (one per flood target):
    s3://<bucket>/<prefix>analysis/figures/tc_matched_bars_Q{10,50,100}.{png,svg}

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
OUTPUT_STEM = "analysis/figures/tc_matched_bars_Q"


def make_figure(rp: int, m: dict, bucket: str, prefix: str) -> None:
    """One figure for one flood target: POD/FAR/CSI (left) + FAF (right, log) for
    the matched precip ARI, same layout as a single 09a bar group."""
    x = np.array([0.0])
    width = 0.20
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax2 = ax.twinx()

    # left-axis skill bars: POD, FAR, CSI
    for k, (metric, color) in enumerate(m09a.LEFT_METRICS):
        off = (k - 1.5) * width
        v = m[metric]
        ax.bar(x + off, [v], width, color=color)
        ax.text(x[0] + off, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, rotation=90)

    # right-axis FAF bar (log), from a small floor so it is visible
    faf = m["FAF"]
    floor = max(faf / 5.0, 1e-3) if faf > 0 else 1e-3
    ax2.set_yscale("log")
    off_faf = 1.5 * width
    ax2.bar(x + off_faf, [max(faf, floor) - floor], width, bottom=floor, color=m09a.FAF_COLOR)
    if faf > 0:
        ax2.annotate(m09a._fmt(faf), (x[0] + off_faf, faf), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=11, color=m09a.FAF_COLOR, rotation=90)
    ax2.set_ylim(floor, max(faf * 4, floor * 10))

    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.6, 0.6)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_ylabel("Skill score", fontsize=15)
    ax2.set_ylabel("FAF (false alarms / station-yr)", fontsize=15, color=m09a.FAF_COLOR)
    ax.set_xticks([0]); ax.set_xticklabels([f"P{rp} @ Tc"], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13, colors=m09a.FAF_COLOR)
    ax.grid(axis="y", ls=":", alpha=0.4)

    handles = [Patch(color=c, label=mn) for mn, c in m09a.LEFT_METRICS] + \
              [Patch(color=m09a.FAF_COLOR, label="FAF")]
    ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_title(f"Q{rp} flood target — matched precip ARI P{rp} @ Tc\n"
                 f"{m['n_events']} events across {m['n_stations']} stations "
                 f"(global pool, MRMS nearest)", fontsize=13)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}{rp}.{ext}")
        log.info("Saved s3://%s/%s%s%d.%s", bucket, prefix, OUTPUT_STEM, rp, ext)
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

    for rp in FLOW_RPS:
        sub = df[(df["flow_rp_yr"] == rp) & (df["precip_rp_yr"] == rp)]
        if sub.empty:
            log.warning("No matched rows for P%d/Q%d — skipping", rp, rp)
            continue
        tp = float(sub["tp"].sum()); fp = float(sub["fp"].sum()); fn = float(sub["fn"].sum())
        station_years = float(sub.groupby("site_no")["n_common_hours"].first().sum()) / m09a.HOURS_PER_YEAR
        m = {
            "POD":        tp / (tp + fn + m09a.EPS),
            "FAR":        fp / (tp + fp + m09a.EPS),
            "CSI":        tp / (tp + fp + fn + m09a.EPS),
            "FAF":        fp / (station_years + m09a.EPS),
            "n_events":   int(sub.groupby("site_no")["n_flow_events"].first().sum()),
            "n_stations": int(sub["site_no"].nunique()),
        }
        log.info("Q%d (P%d): POD=%.2f FAR=%.2f CSI=%.2f FAF=%.2f/stn-yr",
                 rp, rp, m["POD"], m["FAR"], m["CSI"], m["FAF"])
        make_figure(rp, m, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
