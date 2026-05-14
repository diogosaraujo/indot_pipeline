"""09_figures.py

Generate summary figures from the trigger analysis results produced by
script 08.  Figures are written as PNG files to:

    s3://<bucket>/<prefix>analysis/figures/
    s3://<bucket>/<prefix>analysis/figures/stations/   (per-gauge)

Figures produced
----------------
Aggregate (averaged across all stations):
    csi_heatmap_Q{rp}.png        Mean CSI by duration × precip return period
    pod_heatmap_Q{rp}.png        Mean POD by duration × precip return period
    far_heatmap_Q{rp}.png        Mean FAR by duration × precip return period
    pod_vs_far_Q{rp}.png         POD vs FAR scatter (colour = duration)
    best_csi_per_station.png     Best achievable CSI bar chart per gauge
    best_combo_per_station.png   Best (duration, precip_rp) combo per gauge
    map_best_csi.png             Indiana map of gauges coloured by best CSI

Per gauge (figures/stations/):
    {site_no}_csi.png            CSI heatmap for Q10 and Q50 side by side

A text summary of the overall skill scores is printed to the log.
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")  # no display on EC2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
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
STATIONS_PREFIX = "analysis/figures/stations/"

DURATION_LABELS = {
    1: "1 h", 2: "2 h", 3: "3 h", 6: "6 h", 12: "12 h",
    24: "1 d", 48: "2 d", 72: "3 d", 96: "4 d", 120: "5 d",
    168: "7 d", 240: "10 d", 480: "20 d", 720: "30 d",
    1080: "45 d", 1440: "60 d",
}


# ---------- I/O helpers ----------

def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def load_trigger_analysis(bucket: str, prefix: str) -> pd.DataFrame:
    return _read_parquet_s3(bucket, f"{prefix}analysis/trigger_analysis.parquet")


def load_stations(bucket: str, prefix: str) -> pd.DataFrame:
    df = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    df["site_no"] = df["site_no"].astype(str)
    return df[["site_no", "station_nm", "dec_lat_va", "dec_long_va"]].dropna(
        subset=["dec_lat_va", "dec_long_va"]
    )


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def save_figure(fig: plt.Figure, bucket: str, key: str) -> None:
    write_bytes_to_s3(fig_to_bytes(fig), bucket, key)
    log.info("Saved s3://%s/%s", bucket, key)
    plt.close(fig)


# ---------- Metric computation ----------

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["POD"] = df["tp"] / (df["tp"] + df["fn"] + EPS)
    df["FAR"] = df["fp"] / (df["tp"] + df["fp"] + EPS)
    df["CSI"] = df["tp"] / (df["tp"] + df["fp"] + df["fn"] + EPS)
    # Flag degenerate rows: no events AND no triggers → metrics are meaningless
    df["degenerate"] = (df["tp"] == 0) & (df["fn"] == 0) & (df["fp"] == 0)
    return df


# ---------- Shared helpers ----------

def _duration_tick_labels(index):
    return [DURATION_LABELS.get(v, str(v)) for v in index]


def _pool_metrics(df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Sum raw TP/FP/FN counts by groupby_cols, then derive metrics from pooled totals."""
    counts = df.groupby(groupby_cols)[["tp", "fp", "fn"]].sum().reset_index()
    counts["POD"] = counts["tp"] / (counts["tp"] + counts["fn"] + EPS)
    counts["FAR"] = counts["fp"] / (counts["tp"] + counts["fp"] + EPS)
    counts["CSI"] = counts["tp"] / (counts["tp"] + counts["fp"] + counts["fn"] + EPS)
    return counts


def _csi_heatmap_ax(ax, data: pd.DataFrame, flow_rp: int, title: str) -> None:
    sub = data[data["flow_rp_yr"] == flow_rp]
    pivot = sub.groupby(["duration_hr", "precip_rp_yr"])["CSI"].mean().unstack()
    pivot.index = _duration_tick_labels(pivot.index)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        ax=ax,
        cbar=True,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Precip Return Period (yr)", fontsize=8)
    ax.set_ylabel("Duration", fontsize=8)
    ax.tick_params(axis="x", rotation=0, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


# ---------- Aggregate figures ----------

def heatmap_fig(
    df: pd.DataFrame,
    metric: str,
    flow_rp: int,
    cmap: str,
) -> plt.Figure:
    sub = df[df["flow_rp_yr"] == flow_rp]
    pooled = _pool_metrics(sub, ["duration_hr", "precip_rp_yr"])
    pivot = pooled.set_index("duration_hr")[["precip_rp_yr", metric]]
    pivot = pivot.pivot(columns="precip_rp_yr", values=metric)
    pivot.index = _duration_tick_labels(pivot.index)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        ax=ax,
    )
    sources = ", ".join(df["mrms_source"].unique())
    ax.set_title(
        f"{metric} (pooled counts) — flow threshold Q{flow_rp}  |  MRMS source: {sources}",
        fontsize=12,
    )
    ax.set_xlabel("Precip Return Period (yr)", fontsize=10)
    ax.set_ylabel("Accumulation Duration", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return fig


def pod_vs_far_fig(df: pd.DataFrame, flow_rp: int) -> plt.Figure:
    sub = df[df["flow_rp_yr"] == flow_rp]
    agg = _pool_metrics(sub, ["duration_hr", "precip_rp_yr"])

    duration_vals = sorted(agg["duration_hr"].unique())
    norm = plt.Normalize(vmin=np.log2(min(duration_vals)), vmax=np.log2(max(duration_vals)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in agg.iterrows():
        color = cmap(norm(np.log2(row["duration_hr"])))
        ax.scatter(row["FAR"], row["POD"], color=color, s=55, alpha=0.85, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_ticks([np.log2(d) for d in duration_vals])
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


def best_combo_per_station_fig(df: pd.DataFrame) -> plt.Figure:
    """For each station show which (duration, precip_rp, flow_rp) gave the best CSI."""
    idx = df.groupby("site_no")["CSI"].idxmax()
    best = df.loc[idx, ["site_no", "duration_hr", "precip_rp_yr", "flow_rp_yr", "CSI"]].copy()
    best["duration_label"] = best["duration_hr"].map(DURATION_LABELS)
    best["combo"] = best["duration_label"] + " / " + best["precip_rp_yr"].astype(str) + "yr / Q" + best["flow_rp_yr"].astype(str)
    best = best.sort_values("CSI", ascending=True)

    n = len(best)
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.22)))
    colors = ["#2ecc71" if v >= 0.3 else "#e67e22" if v >= 0.1 else "#e74c3c" for v in best["CSI"]]
    bars = ax.barh(range(n), best["CSI"], color=colors, edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{row.site_no}  ({row.combo})" for _, row in best.iterrows()],
        fontsize=7,
    )
    ax.set_xlabel("Best CSI")
    ax.set_title("Best trigger combination per gauge\n(duration / precip return period / flow threshold)")
    ax.axvline(0.3, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.1, color="#e67e22", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def map_best_csi_fig(df: pd.DataFrame, stations: pd.DataFrame) -> plt.Figure:
    best = df.groupby("site_no")["CSI"].max().reset_index()
    merged = stations.merge(best, on="site_no", how="inner")

    fig, ax = plt.subplots(figsize=(7, 9))
    norm = mcolors.Normalize(vmin=0, vmax=max(0.3, merged["CSI"].max()))
    cmap = plt.get_cmap("RdYlGn")

    sc = ax.scatter(
        merged["dec_long_va"],
        merged["dec_lat_va"],
        c=merged["CSI"],
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolors="k",
        linewidths=0.3,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label("Best CSI (any combination)", fontsize=9)

    # Annotate stations with notably good CSI
    for _, row in merged[merged["CSI"] >= 0.3].iterrows():
        ax.annotate(
            row["site_no"],
            (row["dec_long_va"], row["dec_lat_va"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=6,
            color="black",
        )

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("Indiana gauges — best achievable CSI\n(red = poor, green = skilful)", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ---------- Per-station figures ----------

def station_csi_fig(site_no: str, station_nm: str, df_site: pd.DataFrame) -> plt.Figure:
    flow_rps = sorted(df_site["flow_rp_yr"].unique())
    ncols = len(flow_rps)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for ax, flow_rp in zip(axes, flow_rps):
        title = f"Q{flow_rp} threshold"
        _csi_heatmap_ax(ax, df_site, flow_rp, title)

    sources = ", ".join(df_site["mrms_source"].unique())
    n_events = int(df_site["tp"].max() + df_site["fn"].max())
    fig.suptitle(
        f"CSI — Station {site_no}  |  {station_nm}\nMRMS: {sources}  |  max flow events detected/missed: {n_events}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------- Summary log ----------

def log_summary(df: pd.DataFrame) -> None:
    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()
    pod = tp / (tp + fn + EPS)
    far = fp / (tp + fp + EPS)
    csi = tp / (tp + fp + fn + EPS)

    pooled_combos = _pool_metrics(df, ["duration_hr", "precip_rp_yr", "flow_rp_yr"])
    best_row = pooled_combos.loc[pooled_combos["CSI"].idxmax()]
    best_idx = (int(best_row["duration_hr"]), int(best_row["precip_rp_yr"]), int(best_row["flow_rp_yr"]))
    best_csi = float(best_row["CSI"])

    n_degen = df[df["degenerate"]]["site_no"].nunique()
    log.info("── Overall skill (all combos aggregated) ──────────────────")
    log.info("  POD : %.3f", pod)
    log.info("  FAR : %.3f", far)
    log.info("  CSI : %.3f", csi)
    log.info(
        "  Best avg CSI combo: duration=%dh  precip_rp=%dyr  flow_rp=Q%d  CSI=%.3f",
        best_idx[0], best_idx[1], best_idx[2], best_csi,
    )
    log.info(
        "  Stations with no events AND no triggers (degenerate rows): %d", n_degen
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

    log.info("Loading station coordinates from S3...")
    stations = load_stations(bucket, prefix)

    # ── Aggregate figures ──────────────────────────────────────────────
    for flow_rp in sorted(df["flow_rp_yr"].unique()):
        log.info("Generating aggregate figures for Q%d...", flow_rp)

        fig = heatmap_fig(df, "CSI", flow_rp, cmap="YlOrRd")
        save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}csi_heatmap_Q{flow_rp}.png")

        fig = heatmap_fig(df, "POD", flow_rp, cmap="Greens")
        save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}pod_heatmap_Q{flow_rp}.png")

        fig = heatmap_fig(df, "FAR", flow_rp, cmap="Reds")
        save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}far_heatmap_Q{flow_rp}.png")

        fig = pod_vs_far_fig(df, flow_rp)
        save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}pod_vs_far_Q{flow_rp}.png")

    fig = best_csi_per_station_fig(df)
    save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}best_csi_per_station.png")

    fig = best_combo_per_station_fig(df)
    save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}best_combo_per_station.png")

    fig = map_best_csi_fig(df, stations)
    save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}map_best_csi.png")

    # ── Per-station figures ────────────────────────────────────────────
    station_map = stations.set_index("site_no")["station_nm"].to_dict()
    stations_in_df = sorted(df["site_no"].unique())
    log.info("Generating per-station CSI figures for %d stations...", len(stations_in_df))

    for i, site_no in enumerate(stations_in_df, 1):
        df_site = df[df["site_no"] == site_no]
        station_nm = station_map.get(site_no, "")
        fig = station_csi_fig(site_no, station_nm, df_site)
        key = f"{prefix}{STATIONS_PREFIX}{site_no}_csi.png"
        save_figure(fig, bucket, key)
        if i % 10 == 0:
            log.info("  %d / %d stations done", i, len(stations_in_df))

    log.info(
        "All figures written to s3://%s/%s%s",
        bucket, prefix, FIGURES_PREFIX,
    )


if __name__ == "__main__":
    main()
