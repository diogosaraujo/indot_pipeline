"""03c_cluster_stations.py

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

Writes (local results/ and s3://indot-bridge-pipeline/v1/clusters/):
    dendrogram.png
    clusters_k{k}.csv
    clusters_k{k}_stats.csv
    clusters_Q{rp}_k{k}.png
"""
from __future__ import annotations

import io
import os
import warnings

import boto3
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
BUCKET     = "indot-bridge-pipeline"
PREFIX     = "v1/"
OUT_DIR    = "results"
S3_OUT_PRE = "v1/clusters/"

# Three weakly-correlated basin axes: size, steepness, imperviousness.
# tc_hr is intentionally excluded — Kirpich derives it from slope and channel
# length (which tracks area), so it double-counts the size/slope signal.
FEATURES     = ["drain_area_mi2", "slope_ft_mi", "pct_u"]
FEAT_LABELS  = ["Area (mi²)", "Slope (ft/mi)", "Impervious (%)"]
RETURN_PERIODS       = [10, 50, 100]
START_DATE           = "2002-01-01"
END_DATE             = "2022-12-31"
MIN_CLUSTER_EVENTS   = 30     # each cluster needs ≥ this many events; total needs ≥ 2×
K_RANGE              = range(2, 11)
LOG_FEATURES         = ["drain_area_mi2", "slope_ft_mi"]  # log-transform before scaling
KMEANS_SEED          = 42
KMEANS_NINIT         = 20

PRECIP_DURATIONS_HR = [1, 3, 6, 12, 24]           # Atlas 14 durations to check
MRMS_PRODUCT_KEY    = "QPE_01H_Pass2"

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


_s3_client = None


def _upload(local_path: str, filename: str, content_type: str) -> None:
    """Upload a local file to s3://BUCKET/S3_OUT_PRE/filename."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    key = S3_OUT_PRE + filename
    _s3_client.upload_file(
        local_path, BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    print(f"  → s3://{BUCKET}/{key}")


# ── Step 1: Count exceedances ──────────────────────────────────────────────────

def count_exceedances(
    flow_stats: pd.DataFrame,
) -> dict[int, pd.Series]:
    """Return {rp: Series[site_no → n_distinct_events]}.

    Two above-threshold segments are merged into one event when the wall-clock
    gap between them is ≤24 h; a gap >24 h starts a new event.  This is
    determined directly from IV timestamps — no daily resampling.  No station QC
    is applied: the LP3 peak-series thresholds are trustworthy, and negative
    readings cannot exceed a positive Q threshold so they never create events.
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

    counts: dict[int, pd.Series] = {}
    for rp in RETURN_PERIODS:
        col = f"Q{rp}"
        thresholds = (
            flow_stats.set_index("site_no")[col]
            .dropna()
            .rename("threshold")
        )
        merged = sf.merge(thresholds, on="site_no", how="inner")

        # Keep only above-threshold readings, sorted by station then time
        exceed = (
            merged[merged["value_cfs"] >= merged["threshold"]]
            .sort_values(["site_no", "datetime"])
            .copy()
        )

        # Wall-clock gap between consecutive above-threshold readings (same station)
        exceed["prev_dt"] = exceed.groupby("site_no")["datetime"].shift(1)
        exceed["gap_hr"]  = (
            (exceed["datetime"] - exceed["prev_dt"])
            .dt.total_seconds()
            .div(3600)
        )

        # New event: first reading for this station, or gap to previous > 24 h
        exceed["new_event"] = exceed["prev_dt"].isna() | (exceed["gap_hr"] > 24)

        series = (
            exceed.groupby("site_no")["new_event"]
            .sum()
            .rename(f"events_Q{rp}")
        )
        counts[rp] = series[series > 0]

    return counts


def print_exceedance_summary(counts: dict[int, pd.Series]) -> None:
    print("\n── Step 1: Exceedance counts (2002–2022) ─────────────────────────────")
    print("  (thresholds: LP3 peak-series Q values, source=lp3_peak_series)")
    for rp in RETURN_PERIODS:
        total = int(counts[rp].sum())
        n_sta = counts[rp].shape[0]
        max_k = total // MIN_CLUSTER_EVENTS
        print(
            f"  Q{rp:3d}: {total:5d} events  "
            f"({n_sta} stations with ≥1 event)  →  max_k = {max_k}"
        )


# ── Step 1b: Precipitation exceedances ────────────────────────────────────────

def count_precip_exceedances() -> tuple[dict[int, pd.Series], pd.Timestamp, pd.Timestamp]:
    """Count distinct precipitation exceedance events per station (MRMS + Atlas 14).

    For each return period an event is counted whenever the rolling accumulation
    for ANY duration in PRECIP_DURATIONS_HR exceeds the Atlas 14 threshold.
    Two above-threshold periods separated by ≤24 h are merged into one event;
    a gap >24 h starts a new one — same rule as the streamflow analysis.

    Returns (counts_dict, mrms_start, mrms_end).
    """
    print("\nLoading MRMS nearest-pixel precipitation...")
    mrms = read_s3(
        f"mrms/{MRMS_PRODUCT_KEY}/nearest_pixel.parquet",
        columns=["site_no", "datetime_utc", "value"],
    )
    mrms["datetime_utc"] = pd.to_datetime(mrms["datetime_utc"], utc=True)
    mrms = mrms.rename(columns={"value": "precip_in"})
    mrms["site_no"] = mrms["site_no"].astype(str)

    print("Loading Atlas 14 precipitation frequency...")
    atlas14 = read_s3("atlas14/precipitation_frequency.parquet")
    atlas14["site_no"] = atlas14["site_no"].astype(str)

    mrms_start = mrms["datetime_utc"].min()
    mrms_end   = mrms["datetime_utc"].max()
    n_yrs = (mrms_end - mrms_start).days / 365.25
    print(f"  MRMS period: {mrms_start.date()} → {mrms_end.date()} ({n_yrs:.1f} yr)")

    sites = sorted(set(mrms["site_no"]) & set(atlas14["site_no"]))
    print(f"  Stations with MRMS + Atlas 14 data: {len(sites)}")

    # Pre-group for efficiency
    mrms_by_site   = {s: g.set_index("datetime_utc")["precip_in"].sort_index()
                      for s, g in mrms.groupby("site_no")}
    atlas14_by_site = {s: g for s, g in atlas14.groupby("site_no")}

    site_records: list[dict] = []

    for site_no in sites:
        precip = mrms_by_site[site_no]
        # Fill to a complete hourly grid (missing hours = no rain)
        full_idx = pd.date_range(precip.index.min(), precip.index.max(),
                                 freq="1h", tz="UTC")
        precip = precip.reindex(full_idx, fill_value=0.0)

        a14 = atlas14_by_site.get(site_no, pd.DataFrame())
        if a14.empty:
            continue

        for rp in RETURN_PERIODS:
            any_above = pd.Series(False, index=precip.index)
            for dur_h in PRECIP_DURATIONS_HR:
                row = a14[
                    (a14["duration_hr"] == dur_h) &
                    (a14["return_period_yr"] == rp)
                ]
                if row.empty:
                    continue
                threshold = float(row["depth_in"].iloc[0])
                rolling   = precip.rolling(dur_h, min_periods=dur_h).sum()
                any_above = any_above | (rolling >= threshold)

            above_ts = any_above[any_above].index
            if len(above_ts) == 0:
                continue

            gaps_hr  = pd.Series(above_ts).diff().dt.total_seconds().div(3600)
            n_events = 1 + int((gaps_hr > 24).sum())
            site_records.append({"site_no": site_no, "rp": rp, "n_events": n_events})

    df = pd.DataFrame(site_records) if site_records else pd.DataFrame(
        columns=["site_no", "rp", "n_events"]
    )

    result: dict[int, pd.Series] = {}
    for rp in RETURN_PERIODS:
        sub = df[df["rp"] == rp].set_index("site_no")["n_events"] if not df.empty else pd.Series(dtype=int)
        result[rp] = sub[sub > 0] if not sub.empty else sub

    return result, mrms_start, mrms_end


def print_precip_summary(
    counts: dict[int, pd.Series],
    mrms_start: pd.Timestamp,
    mrms_end: pd.Timestamp,
) -> None:
    yrs = (mrms_end - mrms_start).days / 365.25
    print(
        f"\n── Step 1b: Precipitation exceedances  "
        f"(MRMS {mrms_start.date()} → {mrms_end.date()}, {yrs:.1f} yr) "
        f"─────────────────────────────────────────"
    )
    print(f"  Durations: {PRECIP_DURATIONS_HR} h  |  any-duration exceedance, >24 h gap separates events")
    for rp in RETURN_PERIODS:
        total = int(counts[rp].sum()) if not counts[rp].empty else 0
        n_sta = counts[rp].shape[0]
        print(f"  P{rp:3d}: {total:5d} events  ({n_sta} stations with ≥1 event)")


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
    _upload(out_path, os.path.basename(out_path), "image/png")


# ── Step 3: K-means + silhouette ───────────────────────────────────────────────

def run_kmeans(
    X_scaled: np.ndarray,
    site_nos: pd.Series,
    exceedances: dict[int, pd.Series],
) -> list[dict]:
    """Run k-means for each k; return list of result dicts (one per k)."""
    print("\n── Step 3: K-means evaluation (k = 2 .. 10) ─────────────────────────")
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


# ── Step 4: Recommend k per return period (hybrid rule) ───────────────────────

def recommend_k(results: list[dict], counts: dict[int, pd.Series]) -> dict[int, dict]:
    """Choose the number of clusters per return period.

    Rule (per RP):
      total = Σ exceedance events.
      • total < 2×MIN_CLUSTER_EVENTS (60)  → 1 group (too few events to split).
      • otherwise choose the k with the HIGHEST silhouette among all k that satisfy:
            k ≤ total // MIN_CLUSTER_EVENTS         (event budget)
            min cluster events ≥ MIN_CLUSTER_EVENTS (every group is usable)
        If none qualify → 1 group.

    Returns {rp: {"k", "silhouette", "min_events", "reason"}}.
    """
    rec: dict[int, dict] = {}
    for rp in RETURN_PERIODS:
        total    = int(counts[rp].sum())
        k_budget = total // MIN_CLUSTER_EVENTS
        if k_budget < 2:
            rec[rp] = {"k": 1, "silhouette": None, "min_events": total,
                       "reason": f"only {total} events (< {2*MIN_CLUSTER_EVENTS})"}
            continue

        cands = [
            r for r in results
            if r["k"] <= k_budget
            and r[f"min_events_Q{rp}"] >= MIN_CLUSTER_EVENTS
        ]
        if cands:
            best = max(cands, key=lambda r: r["silhouette"])
            rec[rp] = {"k": best["k"], "silhouette": best["silhouette"],
                       "min_events": best[f"min_events_Q{rp}"], "reason": "ok"}
        else:
            rec[rp] = {"k": 1, "silhouette": None, "min_events": total,
                       "reason": f"no k≤{k_budget} with every cluster ≥ "
                                 f"{MIN_CLUSTER_EVENTS} events"}
    return rec


def print_recommendation_table(
    results: list[dict],
    rec: dict[int, dict],
    counts: dict[int, pd.Series],
) -> None:
    print(f"\n── Step 4: k evaluation (best silhouette with every cluster ≥ "
          f"{MIN_CLUSTER_EVENTS} events; total ≥ {2*MIN_CLUSTER_EVENTS} to split) ──")
    header = f"  {'k':>3}  {'Silhouette':>10}  " + "  ".join(
        f"{'minEv_Q'+str(rp):>11}" for rp in RETURN_PERIODS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: x["k"]):
        print(
            f"  {r['k']:>3}  {r['silhouette']:>10.4f}  "
            + "  ".join(f"{r[f'min_events_Q{rp}']:>11}" for rp in RETURN_PERIODS)
        )

    print("\n  Recommended k per return period:")
    for rp in RETURN_PERIODS:
        d = rec[rp]
        total = int(counts[rp].sum())
        if d["k"] == 1:
            print(f"    Q{rp:3d} ({total:5d} events): 1 group  — {d['reason']}")
        else:
            print(f"    Q{rp:3d} ({total:5d} events): k={d['k']}  "
                  f"(silhouette={d['silhouette']:.4f}, min cluster events={d['min_events']})")


# ── Step 5: Write cluster CSVs + stats ────────────────────────────────────────

def _labels_for_k(k: int, results: list[dict], n: int) -> np.ndarray:
    """Return cluster labels for a given k (k=1 → all stations in group 0)."""
    if k == 1:
        return np.zeros(n, dtype=int)
    for r in results:
        if r["k"] == k:
            return r["labels"]
    return np.zeros(n, dtype=int)


def write_cluster_outputs(
    stations: pd.DataFrame,
    results: list[dict],
    rec: dict[int, dict],
) -> None:
    """Write clusters_k{k}.csv and clusters_k{k}_stats.csv for each recommended k."""
    rec_ks = sorted({d["k"] for d in rec.values()})

    print("\n── Step 5: Cluster CSV outputs ────────────────────────────────────────")
    n = len(stations)
    for k in rec_ks:
        labels = _labels_for_k(k, results, n)

        df = stations[["site_no"] + FEATURES].copy()
        df["cluster"] = labels

        # Assignment CSV
        assign_path = os.path.join(OUT_DIR, f"clusters_k{k}.csv")
        df[["site_no", "cluster"]].to_csv(assign_path, index=False)
        _upload(assign_path, f"clusters_k{k}.csv", "text/csv")

        # Statistics CSV: mean and std per cluster, plus centroids
        stats = df.groupby("cluster")[FEATURES].agg(["mean", "std"])
        stats.columns = ["_".join(c) for c in stats.columns]
        stats_path = os.path.join(OUT_DIR, f"clusters_k{k}_stats.csv")
        stats.to_csv(stats_path)
        _upload(stats_path, f"clusters_k{k}_stats.csv", "text/csv")

        n_per_cluster = df["cluster"].value_counts().sort_index()
        print(f"\n  k={k}: {assign_path}, {stats_path}")
        print(f"  Stations per cluster: {n_per_cluster.to_dict()}")
        print("  Centroids (original units):")
        centroids = df.groupby("cluster")[FEATURES].mean()
        centroids.columns = FEAT_LABELS
        print(centroids.round(3).to_string(index=True))


# ── Step 6: Scatter plots ──────────────────────────────────────────────────────

# Pairwise feature combinations to plot (all three pairs of the 3 features).
_PAIRS = [
    ("drain_area_mi2", "slope_ft_mi"),
    ("drain_area_mi2", "pct_u"),
    ("slope_ft_mi",    "pct_u"),
]
_LOG_FEATURES = {"drain_area_mi2", "slope_ft_mi"}   # span orders of magnitude → log
_AXIS_LABEL   = dict(zip(FEATURES, FEAT_LABELS))
_TAG          = {"drain_area_mi2": "area", "slope_ft_mi": "slope", "pct_u": "imperv"}


def plot_scatter(
    stations: pd.DataFrame,
    results: list[dict],
    rec: dict[int, dict],
) -> None:
    """Three pairwise cluster scatters per RP whose recommended k > 1.

    Single-group (k=1) recommendations are skipped — there is no clustering to
    show.  '%' imperviousness includes zeros, so that axis is linear; area and
    slope use log scales.
    """
    print("\n── Step 6: Scatter plots ──────────────────────────────────────────────")
    n = len(stations)
    cmap = plt.get_cmap("tab10")
    for rp in RETURN_PERIODS:
        d = rec[rp]
        k = d["k"]
        if k == 1:
            print(f"  Q{rp}: single group — no scatter ({d['reason']})")
            continue
        labels = _labels_for_k(k, results, n)

        for xcol, ycol in _PAIRS:
            fig, ax = plt.subplots(figsize=(8, 6))
            for c in range(k):
                mask = labels == c
                ax.scatter(
                    stations.loc[mask, xcol],
                    stations.loc[mask, ycol],
                    color=cmap(c),
                    label=f"Cluster {c}  (n={mask.sum()})",
                    alpha=0.75, s=55, edgecolors="white", linewidths=0.4,
                )
            if xcol in _LOG_FEATURES:
                ax.set_xscale("log")
            if ycol in _LOG_FEATURES:
                ax.set_yscale("log")
            ax.set_xlabel(_AXIS_LABEL[xcol])
            ax.set_ylabel(_AXIS_LABEL[ycol])
            ax.set_title(
                f"Q{rp} — k={k} clusters  "
                f"(silhouette={d['silhouette']:.3f}, min cluster events={d['min_events']})"
            )
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            fname = f"clusters_Q{rp}_k{k}_{_TAG[xcol]}_vs_{_TAG[ycol]}.png"
            path  = os.path.join(OUT_DIR, fname)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")
            _upload(path, fname, "image/png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    print("Loading basin characteristics...")
    basin = read_s3("watersheds/basin_characteristics.parquet")
    basin["site_no"] = basin["site_no"].astype(str)

    print("Loading flow statistics...")
    flow_stats = read_s3("flow_stats/per_gauge_flow_stats.parquet")
    flow_stats["site_no"] = flow_stats["site_no"].astype(str)

    # Cluster ONLY stations with a valid peak-series LP3 fit.  Regulated and
    # insufficient-record stations (source != 'lp3_peak_series', null Q) are
    # excluded entirely — their flood frequency is undefined, so they should
    # neither shape the clusters nor inherit a regional Q from one.
    fitted_sites = set(flow_stats.loc[flow_stats["source"] == "lp3_peak_series", "site_no"])
    print(f"LP3-fitted stations (source=lp3_peak_series): {len(fitted_sites)}")

    # Drop stations missing any feature (can't cluster without complete inputs)
    stations = (
        basin[basin["site_no"].isin(fitted_sites)]
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )
    print(
        f"Clusterable stations (fitted AND complete basin characteristics): "
        f"{len(stations)} of {len(fitted_sites)} fitted"
    )

    # Log-transform the heavy-tailed geomorphic features (area, slope span
    # orders of magnitude) so a handful of giant basins don't dominate the
    # Euclidean distance and peel off as singleton outlier clusters.  This also
    # matches the log axes used in the scatter plots.  Imperviousness is bounded
    # [0,100] and includes zeros, so it stays linear.
    feat = stations[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        feat[col] = np.log10(feat[col].clip(lower=1e-6))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat.values)

    # Step 1
    counts = count_exceedances(flow_stats)
    print_exceedance_summary(counts)

    # Step 1b
    try:
        precip_counts, mrms_start, mrms_end = count_precip_exceedances()
        print_precip_summary(precip_counts, mrms_start, mrms_end)
    except Exception as e:
        print(f"\n  WARNING: precipitation exceedance count failed: {e}")

    # Step 2
    plot_dendrogram(X_scaled, os.path.join(OUT_DIR, "dendrogram.png"))

    # Step 3
    results = run_kmeans(X_scaled, stations["site_no"], counts)

    # Step 4
    rec = recommend_k(results, counts)
    print_recommendation_table(results, rec, counts)

    # Step 5
    write_cluster_outputs(stations, results, rec)

    # Step 6
    plot_scatter(stations, results, rec)


if __name__ == "__main__":
    main()
