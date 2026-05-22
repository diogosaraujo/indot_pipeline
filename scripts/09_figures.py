"""09_figures.py

Generate summary figures from the trigger analysis results produced by
script 08.  All figures are produced separately for each MRMS source
(nearest_pixel, watershed_mean).  Figures are written as PNG files to:

    s3://<bucket>/<prefix>analysis/figures/
    s3://<bucket>/<prefix>analysis/figures/stations/   (per-gauge)

Figures produced (suffix = nearest_pixel or watershed_mean)
------------------------------------------------------------
Aggregate:
    csi_heatmap_Q{rp}_{suffix}.png         Pooled CSI by duration × precip_rp
    pod_heatmap_Q{rp}_{suffix}.png         Pooled POD by duration × precip_rp
    far_heatmap_Q{rp}_{suffix}.png         Pooled FAR by duration × precip_rp
    pod_vs_far_Q{rp}_{suffix}.png          POD vs FAR scatter (colour = duration)
    best_csi_per_station_{suffix}.png      Best achievable CSI bar chart
    best_combo_per_station_{suffix}.png    Best (duration, precip_rp) combo
    map_csi_{suffix}.png                   Map: best CSI per station
    map_pod_{suffix}.png                   Map: POD at best-CSI trigger
    map_far_{suffix}.png                   Map: FAR at best-CSI trigger

Per gauge (figures/stations/):
    {site_no}_{suffix}.png                 CSI, POD, FAR heatmaps (Q10 and Q50)
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

METRIC_CMAP = {"CSI": "YlOrRd", "POD": "Greens", "FAR": "Reds"}

SOURCE_LABELS = {
    "nearest":           "MRMS Nearest Pixel",
    "watershed":         "MRMS Watershed Mean",
    "station_nearest":   "Station Nearest",
    "station_watershed": "Station Watershed Mean",
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
    df["degenerate"] = (df["tp"] == 0) & (df["fn"] == 0) & (df["fp"] == 0)
    return df


# ---------- Shared helpers ----------

def _duration_tick_labels(index):
    return [DURATION_LABELS.get(v, str(v)) for v in index]


def _pool_metrics(df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Sum raw TP/FP/FN by groupby_cols, then derive metrics from pooled totals."""
    counts = df.groupby(groupby_cols)[["tp", "fp", "fn"]].sum().reset_index()
    counts["POD"] = counts["tp"] / (counts["tp"] + counts["fn"] + EPS)
    counts["FAR"] = counts["fp"] / (counts["tp"] + counts["fp"] + EPS)
    counts["CSI"] = counts["tp"] / (counts["tp"] + counts["fp"] + counts["fn"] + EPS)
    return counts


def _metric_heatmap_ax(
    ax, data: pd.DataFrame, flow_rp: int, metric: str, title: str
) -> None:
    cmap = METRIC_CMAP.get(metric, "YlOrRd")
    sub = data[data["flow_rp_yr"] == flow_rp]
    pivot = sub.groupby(["duration_hr", "precip_rp_yr"])[metric].mean().unstack()
    pivot.index = _duration_tick_labels(pivot.index)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
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
    src_label: str,
) -> plt.Figure:
    cmap = METRIC_CMAP.get(metric, "YlOrRd")
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
    ax.set_title(
        f"{metric} (pooled counts) — flow threshold Q{flow_rp}  |  Source: {src_label}",
        fontsize=12,
    )
    ax.set_xlabel("Precip Return Period (yr)", fontsize=10)
    ax.set_ylabel("Accumulation Duration", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return fig


def pod_vs_far_fig(df: pd.DataFrame, flow_rp: int, src_label: str) -> plt.Figure:
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
    ax.set_title(
        f"POD vs FAR — flow threshold Q{flow_rp}  |  Source: {src_label}",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def best_csi_per_station_fig(df: pd.DataFrame, flow_rp: int, src_label: str) -> plt.Figure:
    sub = df[df["flow_rp_yr"] == flow_rp]
    best = sub.groupby("site_no")["CSI"].max().sort_values(ascending=False)
    n = len(best)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 4))
    colors = ["#2ecc71" if v >= 0.3 else "#e67e22" if v >= 0.1 else "#e74c3c" for v in best]
    ax.bar(range(n), best.values, color=colors, edgecolor="none")
    ax.set_xticks(range(n))
    ax.set_xticklabels(best.index, rotation=90, fontsize=7)
    ax.set_ylabel("Best CSI (any combination)")
    ax.set_title(
        f"Best achievable CSI per gauge — Q{flow_rp} threshold  |  Source: {src_label}\n"
        "(green ≥ 0.3 | orange ≥ 0.1 | red < 0.1)"
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", alpha=0.3)
    return fig


def best_combo_per_station_fig(df: pd.DataFrame, flow_rp: int, src_label: str) -> plt.Figure:
    """For each station show which (duration, precip_rp) gave the best CSI for a given flow threshold."""
    sub = df[df["flow_rp_yr"] == flow_rp]
    idx = sub.groupby("site_no")["CSI"].idxmax()
    best = sub.loc[idx, ["site_no", "duration_hr", "precip_rp_yr", "CSI"]].copy()
    best["duration_label"] = best["duration_hr"].map(DURATION_LABELS)
    best["combo"] = best["duration_label"] + " / " + best["precip_rp_yr"].astype(str) + "yr"
    best = best.sort_values("CSI", ascending=True)

    n = len(best)
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.22)))
    colors = ["#2ecc71" if v >= 0.3 else "#e67e22" if v >= 0.1 else "#e74c3c" for v in best["CSI"]]
    ax.barh(range(n), best["CSI"], color=colors, edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{row.site_no}  ({row.combo})" for _, row in best.iterrows()],
        fontsize=7,
    )
    ax.set_xlabel("Best CSI")
    ax.set_title(
        f"Best trigger combination per gauge — Q{flow_rp} threshold  |  Source: {src_label}\n"
        "(duration / precip return period)"
    )
    ax.axvline(0.3, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.1, color="#e67e22", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def map_metric_at_best_csi_fig(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    metric: str,
    flow_rp: int,
    src_label: str,
) -> plt.Figure:
    """Map stations coloured by `metric` at the best-CSI trigger for a given flow threshold."""
    sub = df[df["flow_rp_yr"] == flow_rp]
    idx = sub.groupby("site_no")["CSI"].idxmax()
    best = sub.loc[idx, ["site_no", "CSI", "POD", "FAR"]].copy()
    merged = stations.merge(best, on="site_no", how="inner")

    # FAR: lower = better → reversed colormap
    cmap = "RdYlGn" if metric in ("CSI", "POD") else "RdYlGn_r"
    labels = {
        "CSI": f"Best CSI — Q{flow_rp} threshold",
        "POD": f"POD at best-CSI trigger — Q{flow_rp} threshold",
        "FAR": f"FAR at best-CSI trigger — Q{flow_rp} threshold",
    }

    fig, ax = plt.subplots(figsize=(7, 9))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sc = ax.scatter(
        merged["dec_long_va"],
        merged["dec_lat_va"],
        c=merged[metric],
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolors="k",
        linewidths=0.3,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label(labels[metric], fontsize=9)

    for _, row in merged.iterrows():
        annotate = row[metric] <= 0.3 if metric == "FAR" else row[metric] >= 0.3
        if annotate:
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
    ax.set_title(f"{labels[metric]}\nSource: {src_label}", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ---------- Per-station figures ----------

def station_metrics_fig(
    site_no: str,
    station_nm: str,
    df_site: pd.DataFrame,
    src_label: str,
) -> plt.Figure:
    """CSI, POD, FAR heatmaps (rows) × flow thresholds (columns) for one station."""
    flow_rps = sorted(df_site["flow_rp_yr"].unique())
    metrics = ["CSI", "POD", "FAR"]
    ncols = len(flow_rps)
    nrows = len(metrics)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows))
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for row_i, metric in enumerate(metrics):
        for col_i, flow_rp in enumerate(flow_rps):
            ax = axes[row_i, col_i]
            _metric_heatmap_ax(ax, df_site, flow_rp, metric, f"{metric} — Q{flow_rp}")

    n_events = int(df_site["tp"].max() + df_site["fn"].max())
    fig.suptitle(
        f"Station {site_no}  |  {station_nm}\n"
        f"Source: {src_label}  |  max flow events detected/missed: {n_events}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------- Summary log ----------

def log_summary(df: pd.DataFrame, src_label: str) -> None:
    log.info("── Overall skill — MRMS: %s ────────────────────────────", src_label)
    for flow_rp in sorted(df["flow_rp_yr"].unique()):
        sub = df[df["flow_rp_yr"] == flow_rp]
        tp = sub["tp"].sum()
        fp = sub["fp"].sum()
        fn = sub["fn"].sum()
        pod = tp / (tp + fn + EPS)
        far = fp / (tp + fp + EPS)
        csi = tp / (tp + fp + fn + EPS)

        pooled = _pool_metrics(sub, ["duration_hr", "precip_rp_yr"])
        best_row = pooled.loc[pooled["CSI"].idxmax()]
        best_csi = float(best_row["CSI"])

        log.info(
            "  Q%d  POD=%.3f  FAR=%.3f  CSI=%.3f  |  best combo: %s / %dyr  CSI=%.3f",
            flow_rp, pod, far, csi,
            DURATION_LABELS.get(int(best_row["duration_hr"]), str(int(best_row["duration_hr"]))),
            int(best_row["precip_rp_yr"]), best_csi,
        )

    n_degen = df[df["degenerate"]]["site_no"].nunique()
    log.info("  Stations with degenerate rows (no events, no triggers): %d", n_degen)
    log.info("────────────────────────────────────────────────────────────")


# ---------- Main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading trigger_analysis.parquet from S3...")
    df_all = load_trigger_analysis(bucket, prefix)
    df_all = add_metrics(df_all)
    log.info("Loaded %d rows, %d stations", len(df_all), df_all["site_no"].nunique())

    # Keep only stations that have at least one Q10 or Q50 event in the record.
    # Stations with zero events contribute only FP to pooled metrics, inflating
    # FAR without any signal, and produce meaningless per-station figures.
    n_total = df_all["site_no"].nunique()
    sites_with_events = (
        df_all.groupby("site_no")
        .apply(lambda g: (g["tp"] + g["fn"]).max() > 0)
        .pipe(lambda s: s[s].index.tolist())
    )
    df_all = df_all[df_all["site_no"].isin(sites_with_events)]
    log.info(
        "Stations with ≥1 Q10 or Q50 event: %d / %d (excluded %d with zero events)",
        len(sites_with_events), n_total, n_total - len(sites_with_events),
    )

    log.info("Loading station coordinates from S3...")
    stations = load_stations(bucket, prefix)

    sources = sorted(df_all["mrms_source"].unique())
    log.info("MRMS sources found: %s", sources)

    for src_key in sources:
        df = df_all[df_all["mrms_source"] == src_key].copy()
        src_label = SOURCE_LABELS.get(src_key, src_key.replace("_", " ").title())
        sfx = src_key  # file suffix, e.g. "nearest_pixel"
        log.info("── Generating figures for MRMS source: %s (%d rows) ──", src_label, len(df))

        log_summary(df, src_label)

        # ── Aggregate figures (one set per flow threshold) ───────────────
        for flow_rp in sorted(df["flow_rp_yr"].unique()):
            log.info("  Aggregate figures Q%d...", flow_rp)
            for metric in ("CSI", "POD", "FAR"):
                fig = heatmap_fig(df, metric, flow_rp, src_label)
                save_figure(fig, bucket,
                            f"{prefix}{FIGURES_PREFIX}{metric.lower()}_heatmap_Q{flow_rp}_{sfx}.png")

            fig = pod_vs_far_fig(df, flow_rp, src_label)
            save_figure(fig, bucket,
                        f"{prefix}{FIGURES_PREFIX}pod_vs_far_Q{flow_rp}_{sfx}.png")

            fig = best_csi_per_station_fig(df, flow_rp, src_label)
            save_figure(fig, bucket,
                        f"{prefix}{FIGURES_PREFIX}best_csi_per_station_Q{flow_rp}_{sfx}.png")

            fig = best_combo_per_station_fig(df, flow_rp, src_label)
            save_figure(fig, bucket,
                        f"{prefix}{FIGURES_PREFIX}best_combo_per_station_Q{flow_rp}_{sfx}.png")

            for metric in ("CSI", "POD", "FAR"):
                fig = map_metric_at_best_csi_fig(df, stations, metric, flow_rp, src_label)
                save_figure(fig, bucket,
                            f"{prefix}{FIGURES_PREFIX}map_{metric.lower()}_Q{flow_rp}_{sfx}.png")

        # ── Per-station figures ──────────────────────────────────────────
        station_map = stations.set_index("site_no")["station_nm"].to_dict()
        stations_in_df = sorted(df["site_no"].unique())
        log.info("  Per-station figures: %d stations...", len(stations_in_df))

        for i, site_no in enumerate(stations_in_df, 1):
            df_site = df[df["site_no"] == site_no]
            station_nm = station_map.get(site_no, "")
            fig = station_metrics_fig(site_no, station_nm, df_site, src_label)
            key = f"{prefix}{STATIONS_PREFIX}{site_no}_{sfx}.png"
            save_figure(fig, bucket, key)
            if i % 10 == 0:
                log.info("    %d / %d stations done", i, len(stations_in_df))

    log.info(
        "All figures written to s3://%s/%s%s",
        bucket, prefix, FIGURES_PREFIX,
    )


if __name__ == "__main__":
    main()
