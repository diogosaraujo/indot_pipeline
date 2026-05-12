"""09_figures.py

Generate summary figures from the trigger analysis results produced by
script 08.  Figures are written as PNG files to:

    s3://<bucket>/<prefix>analysis/figures/

Figures produced
----------------
For each flow threshold (Q10, Q50):
    csi_heatmap_Q{rp}.png      Mean CSI by duration × precip return period
    pod_heatmap_Q{rp}.png      Mean POD by duration × precip return period
    far_heatmap_Q{rp}.png      Mean FAR by duration × precip return period
    pod_vs_far_Q{rp}.png       POD vs FAR scatter (colour = duration)

Across all thresholds:
    best_csi_per_station.png   Best achievable CSI for each gauge

A text summary of the overall skill scores is printed to the log.
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")  # no display on EC2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("09_figures")

EPS = 1e-9
FIGURES_PREFIX = "analysis/figures/"

DURATION_LABELS = {
    1: "1 h", 2: "2 h", 3: "3 h", 6: "6 h", 12: "12 h",
    24: "1 d", 48: "2 d", 72: "3 d", 96: "4 d", 120: "5 d",
    168: "7 d", 240: "10 d", 480: "20 d", 720: "30 d",
    1080: "45 d", 1440: "60 d",
}


# ---------- I/O helpers ----------

def load_trigger_analysis(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=f"{prefix}analysis/trigger_analysis.parquet")
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def save_figure(fig: plt.Figure, bucket: str, prefix: str, filename: str) -> None:
    key = f"{prefix}{FIGURES_PREFIX}{filename}"
    write_bytes_to_s3(fig_to_bytes(fig), bucket, key)
    log.info("Saved s3://%s/%s", bucket, key)
    plt.close(fig)


# ---------- Metric computation ----------

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["POD"] = df["tp"] / (df["tp"] + df["fn"] + EPS)
    df["FAR"] = df["fp"] / (df["tp"] + df["fp"] + EPS)
    df["CSI"] = df["tp"] / (df["tp"] + df["fp"] + df["fn"] + EPS)
    df["TSS"] = (
        df["tp"] / (df["tp"] + df["fn"] + EPS)
        - df["fp"] / (df["fp"] + df["tn"] + EPS)
    )
    return df


# ---------- Figure helpers ----------

def _duration_tick_labels(index):
    return [DURATION_LABELS.get(v, str(v)) for v in index]


def heatmap_fig(
    df: pd.DataFrame,
    metric: str,
    flow_rp: int,
    cmap: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    sub = df[df["flow_rp_yr"] == flow_rp]
    pivot = sub.groupby(["duration_hr", "precip_rp_yr"])[metric].mean().unstack()
    pivot.index = _duration_tick_labels(pivot.index)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
        ax=ax,
    )
    sources = ", ".join(df["mrms_source"].unique())
    ax.set_title(
        f"Mean {metric} — flow threshold Q{flow_rp}  |  MRMS source: {sources}",
        fontsize=12,
    )
    ax.set_xlabel("Precip Return Period (yr)", fontsize=10)
    ax.set_ylabel("Accumulation Duration", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return fig


def pod_vs_far_fig(df: pd.DataFrame, flow_rp: int) -> plt.Figure:
    sub = df[df["flow_rp_yr"] == flow_rp]
    agg = (
        sub.groupby(["duration_hr", "precip_rp_yr"])[["POD", "FAR"]]
        .mean()
        .reset_index()
    )

    duration_vals = sorted(agg["duration_hr"].unique())
    norm = plt.Normalize(vmin=np.log2(min(duration_vals)), vmax=np.log2(max(duration_vals)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in agg.iterrows():
        color = cmap(norm(np.log2(row["duration_hr"])))
        ax.scatter(row["FAR"], row["POD"], color=color, s=55, alpha=0.85, zorder=3)

    # Colourbar keyed to duration labels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    tick_positions = [np.log2(d) for d in duration_vals]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([DURATION_LABELS.get(d, str(d)) for d in duration_vals])
    cbar.set_label("Accumulation Duration", fontsize=9)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1, label="No skill")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Alarm Ratio (FAR)", fontsize=10)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=10)
    sources = ", ".join(df["mrms_source"].unique())
    ax.set_title(
        f"POD vs FAR — flow threshold Q{flow_rp}  |  MRMS source: {sources}",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def best_csi_per_station_fig(df: pd.DataFrame) -> plt.Figure:
    best = df.groupby("site_no")["CSI"].max().sort_values(ascending=False)
    n = len(best)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 4))
    colors = ["#2ecc71" if v >= 0.3 else "#e67e22" if v >= 0.1 else "#e74c3c" for v in best]
    ax.bar(range(n), best.values, color=colors, edgecolor="none")
    ax.set_xticks(range(n))
    ax.set_xticklabels(best.index, rotation=90, fontsize=7)
    ax.set_ylabel("Best CSI (any combination)")
    ax.set_title("Best achievable CSI per gauge  (green ≥ 0.3 | orange ≥ 0.1 | red < 0.1)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", alpha=0.3)
    return fig


# ---------- Summary log ----------

def log_summary(df: pd.DataFrame) -> None:
    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()
    tn = df["tn"].sum()
    pod  = tp / (tp + fn + EPS)
    far  = fp / (tp + fp + EPS)
    csi  = tp / (tp + fp + fn + EPS)
    tss  = tp / (tp + fn + EPS) - fp / (fp + tn + EPS)

    best_idx = df.groupby(["duration_hr", "precip_rp_yr", "flow_rp_yr"])["CSI"].mean().idxmax()
    best_csi = df.groupby(["duration_hr", "precip_rp_yr", "flow_rp_yr"])["CSI"].mean().max()

    log.info("── Overall skill (all combos aggregated) ──────────────────")
    log.info("  POD : %.3f", pod)
    log.info("  FAR : %.3f", far)
    log.info("  CSI : %.3f", csi)
    log.info("  TSS : %.3f", tss)
    log.info(
        "  Best avg CSI combo: duration=%dh  precip_rp=%dyr  flow_rp=Q%d  CSI=%.3f",
        best_idx[0], best_idx[1], best_idx[2], best_csi,
    )
    log.info("───────────────────────────────────────────────────────────")


# ---------- Main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading trigger_analysis.parquet from S3...")
    df = load_trigger_analysis(bucket, prefix)
    log.info("Loaded %d rows, %d stations", len(df), df["site_no"].nunique())

    df = add_metrics(df)
    log_summary(df)

    for flow_rp in sorted(df["flow_rp_yr"].unique()):
        log.info("Generating figures for Q%d...", flow_rp)

        fig = heatmap_fig(df, "CSI", flow_rp, cmap="YlOrRd")
        save_figure(fig, bucket, prefix, f"csi_heatmap_Q{flow_rp}.png")

        fig = heatmap_fig(df, "POD", flow_rp, cmap="Greens")
        save_figure(fig, bucket, prefix, f"pod_heatmap_Q{flow_rp}.png")

        fig = heatmap_fig(df, "FAR", flow_rp, cmap="Reds")
        save_figure(fig, bucket, prefix, f"far_heatmap_Q{flow_rp}.png")

        fig = pod_vs_far_fig(df, flow_rp)
        save_figure(fig, bucket, prefix, f"pod_vs_far_Q{flow_rp}.png")

    log.info("Generating per-station best CSI figure...")
    fig = best_csi_per_station_fig(df)
    save_figure(fig, bucket, prefix, "best_csi_per_station.png")

    log.info("All figures written to s3://%s/%s%s", bucket, prefix, FIGURES_PREFIX)


if __name__ == "__main__":
    main()
