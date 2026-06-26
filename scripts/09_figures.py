"""09_figures.py

Generate summary figures from the event-overlap confusion matrix produced by
script 08 (analysis/event_confusion_matrix.parquet).  All figures are aggregate
(pooled across gauges) — no per-station figures.  Written to:

    s3://<bucket>/<prefix>analysis/figures/

Metrics: POD = TP/(TP+FN), FAR = FP/(TP+FP), CSI = TP/(TP+FP+FN), pooled from
raw counts.  Sources (suffix): nearest (MRMS pixel), station_nearest (gauge).

Cross-source (one figure):
    metric_comparison_boxplot.png          POD/FAR/CSI spread across all
                                           duration×precip combos: Q10/Q50/Q100
                                           groups, Station vs MRMS boxes

Per source × flow target Q{rp}:
    {csi,pod,far}_heatmap_Q{rp}_{sfx}.png          pooled metric, duration × precip_rp
    pod_vs_far_Q{rp}_{sfx}.png                      POD vs FAR (colour = duration)
    precision_recall_Q{rp}_{sfx}.png               PR curves, one per duration

Cluster breakdown (Q10 only — Q50/Q100 weren't clustered):
    {csi,pod,far}_heatmap_Q10_byCluster_{sfx}.png  pooled metric per basin cluster
    pod_vs_far_Q10_byCluster_{sfx}.png             POD vs FAR coloured by cluster
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")  # no display on EC2
import matplotlib.pyplot as plt
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

# Cluster figures are only meaningful for the return period that was actually
# clustered into groups (Q10 → k=3).  Q50/Q100 collapsed to one group with too
# few events, so cluster breakdowns are skipped for them.
CLUSTER_FLOW_RP = 10

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

# Short group label for cross-source comparisons (boxplots).
SOURCE_GROUP = {
    "nearest":         "MRMS",
    "station_nearest": "Station",
}


# ---------- I/O helpers ----------

def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def load_trigger_analysis(bucket: str, prefix: str) -> pd.DataFrame:
    return _read_parquet_s3(bucket, f"{prefix}analysis/event_confusion_matrix.parquet")


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
    # Auto colour scale to this figure's data range (improves contrast vs 0-1).
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
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
    # Auto-scale to the data with a small margin (FAR for rare targets sits near 1).
    fmin, fmax = agg["FAR"].min(), agg["FAR"].max()
    pmin, pmax = agg["POD"].min(), agg["POD"].max()
    fpad = max(0.02, (fmax - fmin) * 0.05)
    ppad = max(0.02, (pmax - pmin) * 0.05)
    ax.set_xlim(fmin - fpad, fmax + fpad)
    ax.set_ylim(pmin - ppad, pmax + ppad)
    ax.set_xlabel("False Alarm Ratio (FAR)", fontsize=10)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=10)
    ax.set_title(
        f"POD vs FAR — flow threshold Q{flow_rp}  |  Source: {src_label}",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# ---------- Cluster-pooled figures ----------

def _cluster_n_events(cdf: pd.DataFrame) -> int:
    """Number of flood (Q-target) events in a cluster subset.

    n_flow_events is constant per (station, flow_rp), so take one value per
    station and sum.  Falls back to the max TP+FN episode count if the column
    is absent (older output).
    """
    if "n_flow_events" in cdf.columns:
        return int(cdf.groupby("site_no")["n_flow_events"].first().sum())
    per_site = cdf.assign(_e=cdf["tp"] + cdf["fn"]).groupby("site_no")["_e"].max()
    return int(per_site.sum())


def cluster_heatmap_fig(df: pd.DataFrame, metric: str, flow_rp: int, src_label: str) -> plt.Figure:
    """One pooled metric heatmap (duration × precip_rp) per basin cluster, side by side."""
    sub = df[df["flow_rp_yr"] == flow_rp]
    clusters = sorted(int(c) for c in sub["cluster"].dropna().unique())
    cmap = METRIC_CMAP.get(metric, "YlOrRd")

    # Precompute each cluster's pivot, then share ONE colour scale across panels
    # (consistent within the figure, but fit to the data range, not 0-1).
    panels = []
    for c in clusters:
        cdf = sub[sub["cluster"] == c]
        pooled = _pool_metrics(cdf, ["duration_hr", "precip_rp_yr"])
        pivot = pooled.pivot(index="duration_hr", columns="precip_rp_yr", values=metric)
        pivot.index = _duration_tick_labels(pivot.index)
        panels.append((c, cdf, pivot))
    vmin = min(float(p.min().min()) for _, _, p in panels)
    vmax = max(float(p.max().max()) for _, _, p in panels)

    fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 5), squeeze=False)
    for ci, (c, cdf, pivot) in enumerate(panels):
        ax = axes[0, ci]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0.4, ax=ax, cbar=(ci == len(clusters) - 1))
        ax.set_title(
            f"Cluster {c}  ({cdf['site_no'].nunique()} gauges, "
            f"{_cluster_n_events(cdf)} Q{flow_rp} events)", fontsize=10)
        ax.set_xlabel("Precip Return Period (yr)", fontsize=8)
        ax.set_ylabel("Duration" if ci == 0 else "", fontsize=8)
        ax.tick_params(axis="x", rotation=0, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.suptitle(f"{metric} by basin cluster (pooled counts) — Q{flow_rp}  |  {src_label}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def cluster_pod_vs_far_fig(df: pd.DataFrame, flow_rp: int, src_label: str) -> plt.Figure:
    """POD-vs-FAR scatter (each point = one duration×precip_rp combo), coloured by cluster."""
    sub = df[df["flow_rp_yr"] == flow_rp]
    clusters = sorted(int(c) for c in sub["cluster"].dropna().unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    far_all, pod_all = [], []
    for c in clusters:
        cdf = sub[sub["cluster"] == c]
        agg = _pool_metrics(cdf, ["duration_hr", "precip_rp_yr"])
        far_all.append(agg["FAR"]); pod_all.append(agg["POD"])
        ax.scatter(agg["FAR"], agg["POD"], color=cmap(c % 10), s=40, alpha=0.7,
                   edgecolors="white", linewidths=0.3,
                   label=f"Cluster {c} ({cdf['site_no'].nunique()} gauges, "
                         f"{_cluster_n_events(cdf)} events)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1)
    far_all = pd.concat(far_all); pod_all = pd.concat(pod_all)
    fpad = max(0.02, (far_all.max() - far_all.min()) * 0.05)
    ppad = max(0.02, (pod_all.max() - pod_all.min()) * 0.05)
    ax.set_xlim(far_all.min() - fpad, far_all.max() + fpad)
    ax.set_ylim(pod_all.min() - ppad, pod_all.max() + ppad)
    ax.set_xlabel("False Alarm Ratio (FAR)", fontsize=10)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=10)
    ax.set_title(f"POD vs FAR by basin cluster — Q{flow_rp}  |  {src_label}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# ---------- Precision-recall ----------

def precision_recall_fig(df: pd.DataFrame, flow_rp: int, src_label: str) -> plt.Figure:
    """Precision-recall curves: one curve per accumulation duration, traced over precip RPs.

    Recall = POD = TP/(TP+FN);  Precision = TP/(TP+FP) = 1 - FAR.
    Each curve fixes a duration and connects its points as the precip threshold varies.
    """
    sub = df[df["flow_rp_yr"] == flow_rp]
    pooled = _pool_metrics(sub, ["duration_hr", "precip_rp_yr"])
    pooled["precision"] = pooled["tp"] / (pooled["tp"] + pooled["fp"] + EPS)
    pooled["recall"]    = pooled["tp"] / (pooled["tp"] + pooled["fn"] + EPS)

    durations = sorted(pooled["duration_hr"].unique())
    norm = plt.Normalize(vmin=np.log2(min(durations)), vmax=np.log2(max(durations)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9, 6))
    for d in durations:
        dd = pooled[pooled["duration_hr"] == d].sort_values("recall")
        ax.plot(dd["recall"], dd["precision"], "-", lw=1.6,
                color=cmap(norm(np.log2(d))), alpha=0.9,
                label=DURATION_LABELS.get(d, str(d)))

    # Auto-scale to the data so the (often tiny) precision range fills the plot.
    rmax = float(pooled["recall"].max())
    pmax = float(pooled["precision"].max())
    ax.set_xlim(0, max(0.05, rmax * 1.05))
    ax.set_ylim(0, max(1e-3, pmax * 1.1))
    ax.set_xlabel("Recall  (POD = TP / (TP+FN))", fontsize=10)
    ax.set_ylabel("Precision  (TP / (TP+FP) = 1 - FAR)", fontsize=10)
    ax.set_title(f"Precision-Recall — Q{flow_rp} target  |  {src_label}\n"
                 "(one curve per duration; points = precip return periods)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Accumulation", fontsize=7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


# ---------- Cross-source comparison ----------

def metric_comparison_boxplot_fig(df_all: pd.DataFrame) -> plt.Figure:
    """Boxplots of POD/FAR/CSI over all (duration × precip_rp) combos.

    x = flow target (Q10/Q50/Q100); two boxes per group = Station vs MRMS.
    Each box's distribution is the pooled metric across every trigger configuration.
    """
    rows = []
    for src in df_all["source"].unique():
        grp = SOURCE_GROUP.get(src)
        if grp is None:
            continue
        for flow_rp in sorted(df_all["flow_rp_yr"].unique()):
            sub = df_all[(df_all["source"] == src) & (df_all["flow_rp_yr"] == flow_rp)]
            pooled = _pool_metrics(sub, ["duration_hr", "precip_rp_yr"])
            for _, r in pooled.iterrows():
                rows.append({"target": f"Q{int(flow_rp)}", "group": grp,
                             "POD": r["POD"], "FAR": r["FAR"], "CSI": r["CSI"]})
    box = pd.DataFrame(rows)
    order = [f"Q{rp}" for rp in sorted(df_all["flow_rp_yr"].unique())]

    metrics = ["POD", "FAR", "CSI"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, m in zip(axes, metrics):
        sns.boxplot(data=box, x="target", y=m, hue="group", order=order,
                    hue_order=["MRMS", "Station"],
                    palette={"MRMS": "#4c72b0", "Station": "#dd8452"}, ax=ax)
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("Flow target", fontsize=10)
        ax.set_ylabel(m, fontsize=10)
        # No fixed ylim — each subplot auto-scales to its own metric range
        # (FAR clusters near 1, CSI near 0), maximising the visible spread.
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Source", fontsize=8)
    fig.suptitle("Skill distribution across all (duration × precip-RP) combinations",
                 fontsize=13, y=1.02)
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

    log.info("Loading event_confusion_matrix.parquet from S3...")
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

    sources = sorted(df_all["source"].unique())
    log.info("Sources found: %s", sources)

    # Cross-source comparison (uses both Station and MRMS) — one figure overall.
    if len(set(df_all["source"]) & set(SOURCE_GROUP)) >= 1:
        log.info("Station-vs-MRMS comparison boxplot...")
        fig = metric_comparison_boxplot_fig(df_all)
        save_figure(fig, bucket, f"{prefix}{FIGURES_PREFIX}metric_comparison_boxplot.png")

    has_cluster = "cluster" in df_all.columns

    for src_key in sources:
        df = df_all[df_all["source"] == src_key].copy()
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

            # Precision-recall (one curve per duration)
            fig = precision_recall_fig(df, flow_rp, src_label)
            save_figure(fig, bucket,
                        f"{prefix}{FIGURES_PREFIX}precision_recall_Q{flow_rp}_{sfx}.png")

            # Cluster-pooled figures — only for the clustered RP (Q10); Q50/Q100
            # collapsed to one group with too few events per cluster.
            if has_cluster and flow_rp == CLUSTER_FLOW_RP:
                for metric in ("CSI", "POD", "FAR"):
                    fig = cluster_heatmap_fig(df, metric, flow_rp, src_label)
                    save_figure(fig, bucket,
                                f"{prefix}{FIGURES_PREFIX}{metric.lower()}_heatmap_Q{flow_rp}_byCluster_{sfx}.png")
                fig = cluster_pod_vs_far_fig(df, flow_rp, src_label)
                save_figure(fig, bucket,
                            f"{prefix}{FIGURES_PREFIX}pod_vs_far_Q{flow_rp}_byCluster_{sfx}.png")

    log.info(
        "All figures written to s3://%s/%s%s",
        bucket, prefix, FIGURES_PREFIX,
    )


if __name__ == "__main__":
    main()
