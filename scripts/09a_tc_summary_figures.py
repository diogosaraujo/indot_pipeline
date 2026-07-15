"""09a_tc_summary_figures.py

Summary bar charts for the fixed-Tc trigger analysis (08c).

Reads analysis/event_confusion_matrix_tc.parquet, pools TP/FP/FN GLOBALLY (all
stations, no clusters) for the MRMS nearest-pixel source, and draws ONE figure
per flood target (Q10, Q50, Q100).  Each is a grouped bar chart over the
precipitation ARI thresholds; every ARI shows four bars:

    POD, FAR, CSI  → left y-axis   (bounded skill scores)
    FAF            → right y-axis  (False Alarm Frequency = false alarms per
                     station-year, log scale — an unbounded rate)

    POD = TP / (TP + FN)      FAR = FP / (TP + FP)      CSI = TP / (TP + FP + FN)
    FAF = Σ FP / Σ (n_common_hours / 8766)     [false alarms per station-year]

Output (one per flood target):
    s3://<bucket>/<prefix>analysis/figures/tc_summary_bars_Q{10,50,100}.{png,svg}

Usage:
    python scripts/09a_tc_summary_figures.py
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09a_tc")

SOURCE = "nearest"
FLOW_RPS = [10, 50, 100]
EPS = 1e-9
HOURS_PER_YEAR = 8766.0
INPUT_KEY = "analysis/event_confusion_matrix_tc.parquet"
OUTPUT_STEM = "analysis/figures/tc_summary_bars_Q"

LEFT_METRICS = [("POD", "#2c7fb8"), ("FAR", "#d95f0e"), ("CSI", "#31a354")]
FAF_COLOR = "#6a51a3"


def _read_parquet(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2g}")


def make_figure(flow_rp, sub, precip_rps, bucket, prefix):
    pooled = (sub.groupby("precip_rp_yr")[["tp", "fp", "fn"]].sum()
                 .reindex(precip_rps).fillna(0.0))
    tp, fp, fn = pooled["tp"], pooled["fp"], pooled["fn"]
    vals = {
        "POD": (tp / (tp + fn + EPS)).to_numpy(),
        "FAR": (fp / (tp + fp + EPS)).to_numpy(),
        "CSI": (tp / (tp + fp + fn + EPS)).to_numpy(),
    }
    station_years = float(sub.groupby("site_no")["n_common_hours"].first().sum()) / HOURS_PER_YEAR
    faf = (fp / (station_years + EPS)).to_numpy()          # false alarms / station-yr
    n_events = int(sub.groupby("site_no")["n_flow_events"].first().sum())
    n_stations = int(sub["site_no"].nunique())

    x = np.arange(len(precip_rps))
    width = 0.20
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax2 = ax.twinx()

    # left-axis skill bars: POD, FAR, CSI
    for k, (metric, color) in enumerate(LEFT_METRICS):
        off = (k - 1.5) * width
        ax.bar(x + off, vals[metric], width, color=color)
        for xi, v in zip(x + off, vals[metric]):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=11, rotation=90)

    # right-axis FAF bar (log), drawn from a small floor so all bars are visible
    pos = faf[faf > 0]
    floor = max(pos.min() / 5.0, 1e-3) if pos.size else 1e-3
    ax2.set_yscale("log")
    off_faf = 1.5 * width
    tops = np.where(faf > 0, faf, floor)
    ax2.bar(x + off_faf, tops - floor, width, bottom=floor, color=FAF_COLOR)
    for xi, v in zip(x + off_faf, faf):
        if v > 0:
            ax2.annotate(_fmt(v), (xi, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=10, color=FAF_COLOR, rotation=90)
    ax2.set_ylim(floor, (pos.max() * 4 if pos.size else 1))

    ax.set_ylim(0, 1.18)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_ylabel("Skill score", fontsize=15)
    ax2.set_ylabel("FAF (false alarms / station-yr)", fontsize=15, color=FAF_COLOR)
    ax.set_xticks(x); ax.set_xticklabels([f"P{int(r)}" for r in precip_rps], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13, colors=FAF_COLOR)
    ax.set_xlabel("Precipitation ARI threshold (return period, yr)", fontsize=15)
    ax.grid(axis="y", ls=":", alpha=0.4)

    handles = [Patch(color=c, label=m) for m, c in LEFT_METRICS] + [Patch(color=FAF_COLOR, label="FAF")]
    ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=13, framealpha=0.9)
    ax.set_title(f"Q{flow_rp} flood target  —  {n_events} events across "
                 f"{n_stations} stations (global pool, MRMS nearest)", fontsize=17)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}{flow_rp}.{ext}")
        log.info("Saved s3://%s/%s%s%d.%s", bucket, prefix, OUTPUT_STEM, flow_rp, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    df = _read_parquet(bucket, f"{prefix}{INPUT_KEY}",
                       ["site_no", "source", "precip_rp_yr", "flow_rp_yr",
                        "tp", "fp", "fn", "n_flow_events", "n_common_hours"])
    df = df[df["source"] == SOURCE].copy()
    if df.empty:
        log.error("No %s rows in %s", SOURCE, INPUT_KEY)
        return
    precip_rps = sorted(df["precip_rp_yr"].unique())
    for flow_rp in FLOW_RPS:
        sub = df[df["flow_rp_yr"] == flow_rp]
        if sub.empty:
            log.warning("No rows for Q%d — skipping", flow_rp)
            continue
        make_figure(flow_rp, sub, precip_rps, bucket, prefix)


if __name__ == "__main__":
    main()
