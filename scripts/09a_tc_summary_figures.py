"""09a_tc_summary_figures.py

Summary bar charts for the fixed-Tc trigger analysis (08c).

Reads analysis/event_confusion_matrix_tc.parquet, pools TP/FP/FN GLOBALLY (all
stations together, no clusters) for the MRMS nearest-pixel source, and draws a
3×1 multipanel figure — one row per flood target Q10 / Q50 / Q100.  Each panel is
a grouped bar chart over the precipitation ARI thresholds; every ARI shows three
bars — FAR, POD, CSI — with the value written above each bar.

Metrics from pooled counts:
    POD = TP / (TP + FN)      FAR = FP / (TP + FP)      CSI = TP / (TP + FP + FN)

Output:
    s3://<bucket>/<prefix>analysis/figures/tc_summary_bars.png (+ .svg)

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

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09a_tc")

SOURCE = "nearest"                 # MRMS nearest pixel (global pool)
FLOW_RPS = [10, 50, 100]
EPS = 1e-9
INPUT_KEY = "analysis/event_confusion_matrix_tc.parquet"
OUTPUT_KEY = "analysis/figures/tc_summary_bars"

METRICS = [("POD", "#2c7fb8"), ("FAR", "#d95f0e"), ("CSI", "#31a354")]


def _read_parquet(bucket: str, key: str, columns=None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    df = _read_parquet(bucket, f"{prefix}{INPUT_KEY}",
                       ["site_no", "source", "precip_rp_yr", "flow_rp_yr",
                        "tp", "fp", "fn", "n_flow_events"])
    df = df[df["source"] == SOURCE].copy()
    if df.empty:
        log.error("No %s rows in %s", SOURCE, INPUT_KEY)
        return

    precip_rps = sorted(df["precip_rp_yr"].unique())
    x = np.arange(len(precip_rps))
    width = 0.26

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for row, flow_rp in enumerate(FLOW_RPS):
        ax = axes[row]
        sub = df[df["flow_rp_yr"] == flow_rp]
        if sub.empty:
            ax.set_visible(False)
            continue
        pooled = (sub.groupby("precip_rp_yr")[["tp", "fp", "fn"]].sum()
                     .reindex(precip_rps).fillna(0.0))
        tp, fp, fn = pooled["tp"], pooled["fp"], pooled["fn"]
        vals = {
            "POD": (tp / (tp + fn + EPS)).to_numpy(),
            "FAR": (fp / (tp + fp + EPS)).to_numpy(),
            "CSI": (tp / (tp + fp + fn + EPS)).to_numpy(),
        }
        n_events = int(sub.groupby("site_no")["n_flow_events"].first().sum())
        n_stations = int(sub["site_no"].nunique())

        for k, (metric, color) in enumerate(METRICS):
            offset = (k - 1) * width
            bars = ax.bar(x + offset, vals[metric], width, label=metric, color=color)
            for xi, v in zip(x + offset, vals[metric]):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Skill score")
        ax.set_title(f"Q{flow_rp} flood target  —  {n_events} events across "
                     f"{n_stations} stations (global pool, MRMS nearest)",
                     fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.4)
        if row == 0:
            ax.legend(ncol=3, loc="upper right", framealpha=0.9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"P{int(r)}" for r in precip_rps])
    axes[-1].set_xlabel("Precipitation ARI threshold (return period, yr)")

    fig.suptitle(
        "Fixed-Tc trigger skill vs precipitation ARI threshold\n"
        "accumulation duration = round(Kirpich Tc) per station; pooled over all stations",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_KEY}.{ext}")
        log.info("Saved s3://%s/%s%s.%s", bucket, prefix, OUTPUT_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
