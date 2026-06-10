"""cluster_stations.py

Cluster Indiana USGS streamflow stations by basin characteristics and identify
the most statistically viable number of clusters (k) for Q10, Q50, and Q100
exceedance analyses.

Pipeline
--------
1. Count daily Q10/Q50/Q100 exceedances per station (2002-2022).
2. Standardize features; Ward hierarchical clustering → dendrogram.
3. K-means for k = 3 .. min(10, max_k); compute silhouette scores.
4. Viable k: silhouette > 0.40 AND min cluster events >= 30.
5. Write clusters_k{k}.csv + statistics for each viable k.
6. Scatter plots (Area vs Slope) for each recommended k per return period.

Reads (S3 indot-bridge-pipeline/v1/):
    watersheds/basin_characteristics.parquet
    flow_stats/per_gauge_flow_stats.parquet
    streamflow/instantaneous/all_gauges_long.parquet

Writes:
    results/dendrogram.png
    results/clusters_k{k}.csv
    results/clusters_k{k}_stats.csv
    results/clusters_Q{rp}_k{k}.png
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
BUCKET  = "indot-bridge-pipeline"
PREFIX  = "v1/"
OUT_DIR = "results"

FEATURES     = ["drain_area_mi2", "slope_ft_mi", "tc_hr", "pct_u"]
FEAT_LABELS  = ["Area (mi²)", "Slope (ft/mi)", "Tc (hr)", "Impervious (%)"]
RETURN_PERIODS       = [10, 50, 100]
START_DATE           = "2002-01-01"
END_DATE             = "2022-12-31"
MIN_SILHOUETTE       = 0.40
MIN_CLUSTER_EVENTS   = 30
K_RANGE              = range(3, 11)
KMEANS_SEED          = 42
KMEANS_NINIT         = 20

os.makedirs(OUT_DIR, exist_ok=True)


# ── S3 reader ──────────────────────────────────────────────────────────────────
_s3_fs: pafs.S3FileSystem | None = None


def _s3() -> pafs.S3FileSystem:
    global _s3_fs
    if _s3_fs is None:
        _s3_fs = pafs.S3FileSystem()
    return _s3_fs


def read_s3(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(
        f"{BUCKET}/{PREFIX}{key}", filesystem=_s3(), columns=columns
    ).to_pandas()


# ── Step 1: Count exceedances ──────────────────────────────────────────────────

def count_exceedances(
    flow_stats: pd.DataFrame,
) -> dict[int, pd.Series]:
    """Return {rp: Series[site_no → n_days_exceeding_Qrp]} for 2002-2022.

    Resamples the IV record to daily max before counting so that a multi-hour
    flood peak counts as one event per day rather than once per reading.
    """
    print("\nLoading streamflow time series (may take a minute)...")
    sf = read_s3(
        "streamflow/instantaneous/all_gauges_long.parquet",
        columns=["site_no", "datetime", "value_cfs"],
    )
    sf["datetime"]  = pd.to_datetime(sf["datetime"], utc=True)
    sf["value_cfs"] = pd.to_numeric(sf["value_cfs"], errors="coerce")
    sf["site_no"]   = sf["site_no"].astype(str)

    sf = sf[
        (sf["datetime"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (sf["datetime"] <= pd.Timestamp(END_DATE, tz="UTC"))
    ]
    print(f"  {len(sf):,} IV records in analysis window across {sf['site_no'].nunique()} stations")

    # Daily maximum per station
    daily = (
        sf.set_index("datetime")
        .groupby("site_no")["value_cfs"]
        .resample("1D")
        .max()
        .reset_index()
    )

    counts: dict[int, pd.Series] = {}
    for rp in RETURN_PERIODS:
        col = f"Q{rp}"
        thresholds = (
            flow_stats.set_index("site_no")[col]
            .dropna()
            .rename("threshold")
        )
        merged  = daily.merge(thresholds, on="site_no", how="inner")
        exceed  = merged[merged["value_cfs"] >= merged["threshold"]]
        series  = exceed.groupby("site_no").size().rename(f"events_Q{rp}")
        counts[rp] = series

    return counts


def print_exceedance_summary(counts: dict[int, pd.Series]) -> None:
    print("\n── Step 1: Exceedance counts (2002–2022) ─────────────────────────────")
    for rp in RETURN_PERIODS:
        total = int(counts[rp].sum())
        n_sta = counts[rp].shape[0]
        max_k = total // MIN_CLUSTER_EVENTS
        print(
            f"  Q{rp:3d}: {total:6d} total exceedances  "
            f"({n_sta} stations with ≥1 event)  →  max_k = {max_k}"
        )


# ── Step 2: Dendrogram ─────────────────────────────────────────────────────────

def plot_dendrogram(X_scaled: np.ndarray, out_path: str) -> None:
    print("\n── Step 2: Hierarchical clustering (Ward) ────────────────────────────")
    Z = linkage(X_scaled, method="ward", metric="euclidean")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * float(Z[:, 2].max()))
    ax.set_title("Ward Hierarchical Clustering — Basin Characteristics")
    ax.set_xlabel("Station index")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Step 3: K-means + silhouette ───────────────────────────────────────────────

def run_kmeans(
    X_scaled: np.ndarray,
    site_nos: pd.Series,
    exceedances: dict[int, pd.Series],
) -> list[dict]:
    """Run k-means for each k; return list of result dicts (one per k)."""
    print("\n── Step 3: K-means evaluation (k = 3 .. 10) ─────────────────────────")
    print(
        f"  {'k':>3}  {'Silhouette':>10}  "
        + "  ".join(f"{'Q'+str(rp)+' [min-max]':>14}" for rp in RETURN_PERIODS)
    )

    results: list[dict] = []
    for k in K_RANGE:
        km     = KMeans(n_clusters=k, n_init=KMEANS_NINIT, random_state=KMEANS_SEED)
        labels = km.fit_predict(X_scaled)
        sil    = float(silhouette_score(X_scaled, labels))

        row: dict = {"k": k, "silhouette": round(sil, 4), "labels": labels}

        cluster_map = pd.Series(labels, index=site_nos.values, name="cluster")

        for rp in RETURN_PERIODS:
            ev = exceedances[rp]
            cluster_events = (
                cluster_map.reset_index()
                .rename(columns={"index": "site_no"})
                .merge(ev.reset_index(), on="site_no", how="left")
                .fillna(0)
                .groupby("cluster")[f"events_Q{rp}"]
                .sum()
                .astype(int)
            )
            row[f"min_events_Q{rp}"] = int(cluster_events.min())
            row[f"max_events_Q{rp}"] = int(cluster_events.max())

        results.append(row)
        print(
            f"  {k:>3}  {sil:>10.4f}  "
            + "  ".join(
                f"  [{row[f'min_events_Q{rp}']:>4}-{row[f'max_events_Q{rp}']:>4}]"
                for rp in RETURN_PERIODS
            )
        )

    return results


# ── Step 4: Viable k ───────────────────────────────────────────────────────────

def find_viable(results: list[dict]) -> dict[int, list[dict]]:
    """Return {rp: [rows sorted by silhouette desc]} where k passes both filters."""
    viable: dict[int, list[dict]] = {}
    for rp in RETURN_PERIODS:
        candidates = [
            r for r in results
            if r["silhouette"] > MIN_SILHOUETTE
            and r[f"min_events_Q{rp}"] >= MIN_CLUSTER_EVENTS
        ]
        viable[rp] = sorted(candidates, key=lambda x: x["silhouette"], reverse=True)
    return viable


def print_viable_table(results: list[dict], viable: dict[int, list[dict]]) -> None:
    print("\n── Step 4: Viable k table (silhouette > 0.40 AND min events ≥ 30) ────")
    header = f"  {'k':>3}  {'Silhouette':>10}  " + "  ".join(
        f"{'minEv_Q'+str(rp):>11}" for rp in RETURN_PERIODS
    ) + "  viable_for"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(results, key=lambda x: x["k"]):
        flags = "/".join(
            f"Q{rp}" for rp in RETURN_PERIODS
            if any(v["k"] == r["k"] for v in viable[rp])
        )
        recommended = any(
            viable[rp] and viable[rp][0]["k"] == r["k"]
            for rp in RETURN_PERIODS
        )
        tag = " ← RECOMMENDED" if recommended else ""
        print(
            f"  {r['k']:>3}  {r['silhouette']:>10.4f}  "
            + "  ".join(f"{r[f'min_events_Q{rp}']:>11}" for rp in RETURN_PERIODS)
            + f"  {flags or '—'}{tag}"
        )


# ── Step 5: Write cluster CSVs + stats ────────────────────────────────────────

def write_cluster_outputs(
    stations: pd.DataFrame,
    results: list[dict],
    viable: dict[int, list[dict]],
) -> None:
    """Write clusters_k{k}.csv and clusters_k{k}_stats.csv for each viable k."""
    viable_ks = {r["k"] for rp_list in viable.values() for r in rp_list}
    if not viable_ks:
        print("\n  No viable k values — skipping CSV outputs.")
        return

    print("\n── Step 5: Cluster CSV outputs ────────────────────────────────────────")
    for r in sorted(results, key=lambda x: x["k"]):
        k = r["k"]
        if k not in viable_ks:
            continue

        df = stations[["site_no"] + FEATURES].copy()
        df["cluster"] = r["labels"]

        # Assignment CSV
        assign_path = os.path.join(OUT_DIR, f"clusters_k{k}.csv")
        df[["site_no", "cluster"]].to_csv(assign_path, index=False)

        # Statistics CSV: mean and std per cluster, plus centroids
        stats = df.groupby("cluster")[FEATURES].agg(["mean", "std"])
        stats.columns = ["_".join(c) for c in stats.columns]
        stats_path = os.path.join(OUT_DIR, f"clusters_k{k}_stats.csv")
        stats.to_csv(stats_path)

        n_per_cluster = df["cluster"].value_counts().sort_index()
        print(f"\n  k={k}: {assign_path}, {stats_path}")
        print(f"  Stations per cluster: {n_per_cluster.to_dict()}")
        print("  Centroids (original units):")
        centroids = df.groupby("cluster")[FEATURES].mean()
        centroids.columns = FEAT_LABELS
        print(centroids.round(3).to_string(index=True))


# ── Step 6: Scatter plots ──────────────────────────────────────────────────────

def plot_scatter(
    stations: pd.DataFrame,
    viable: dict[int, list[dict]],
) -> None:
    """Area vs Slope scatter coloured by cluster for each recommended (best) k."""
    print("\n── Step 6: Scatter plots ──────────────────────────────────────────────")
    for rp in RETURN_PERIODS:
        if not viable[rp]:
            print(f"  Q{rp}: no viable k — skipping")
            continue

        best   = viable[rp][0]
        k      = best["k"]
        labels = best["labels"]

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("tab10")
        for c in range(k):
            mask = labels == c
            ax.scatter(
                stations.loc[mask, "drain_area_mi2"],
                stations.loc[mask, "slope_ft_mi"],
                color=cmap(c),
                label=f"Cluster {c}  (n={mask.sum()})",
                alpha=0.75,
                s=55,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Drainage Area (mi²)")
        ax.set_ylabel("Channel Slope (ft/mi)")
        ax.set_title(
            f"Q{rp} — k={k} clusters\n"
            f"Silhouette = {best['silhouette']:.3f}  |  "
            f"min cluster events = {best[f'min_events_Q{rp}']}"
        )
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"clusters_Q{rp}_k{k}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# ── Recommendation summary ─────────────────────────────────────────────────────

def print_recommendation_summary(
    counts: dict[int, pd.Series],
    viable: dict[int, list[dict]],
) -> None:
    print("\n── Recommendation summary ─────────────────────────────────────────────")
    print("Recommended k values:")
    for rp in RETURN_PERIODS:
        total = int(counts[rp].sum())
        if viable[rp]:
            best = viable[rp][0]
            line = f"  Q{rp:3d} ({total:6d} events): k={best['k']} (Silhouette={best['silhouette']:.4f})"
            if len(viable[rp]) > 1:
                top3 = ", ".join(
                    f"k={v['k']} (sil={v['silhouette']:.4f}, minEv={v[f'min_events_Q{rp}']})"
                    for v in viable[rp][:3]
                )
                line += f"\n    Top 3 candidates: {top3}"
        else:
            # No viable k — report best available for guidance
            best_avail = sorted(results_global, key=lambda x: x["silhouette"], reverse=True)[:3]
            candidates = ", ".join(
                f"k={r['k']} (sil={r['silhouette']:.4f}, minEv={r[f'min_events_Q{rp}']})"
                for r in best_avail
            )
            line = (
                f"  Q{rp:3d} ({total:6d} events): no viable k "
                f"(criteria: sil>{MIN_SILHOUETTE}, minEv≥{MIN_CLUSTER_EVENTS})\n"
                f"    Best available: {candidates}"
            )
        print(line)


# ── Main ───────────────────────────────────────────────────────────────────────

results_global: list[dict] = []  # used by recommendation summary


def main() -> None:
    global results_global

    # Load data
    print("Loading basin characteristics...")
    basin = read_s3("watersheds/basin_characteristics.parquet")
    basin["site_no"] = basin["site_no"].astype(str)

    print("Loading flow statistics...")
    flow_stats = read_s3("flow_stats/per_gauge_flow_stats.parquet")
    flow_stats["site_no"] = flow_stats["site_no"].astype(str)

    # Drop stations missing any feature (can't cluster without complete inputs)
    stations = basin.dropna(subset=FEATURES).reset_index(drop=True)
    n_dropped = len(basin) - len(stations)
    print(
        f"Stations with complete basin characteristics: "
        f"{len(stations)} / {len(basin)}  ({n_dropped} dropped)"
    )

    X_orig   = stations[FEATURES].values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

    # Step 1
    counts = count_exceedances(flow_stats)
    print_exceedance_summary(counts)

    # Step 2
    plot_dendrogram(X_scaled, os.path.join(OUT_DIR, "dendrogram.png"))

    # Step 3
    results = run_kmeans(X_scaled, stations["site_no"], counts)
    results_global = results

    # Step 4
    viable = find_viable(results)
    print_viable_table(results, viable)

    # Step 5
    write_cluster_outputs(stations, results, viable)

    # Step 6
    plot_scatter(stations, viable)

    # Summary
    print_recommendation_summary(counts, viable)


if __name__ == "__main__":
    main()
