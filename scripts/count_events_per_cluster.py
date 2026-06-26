"""count_events_per_cluster.py

Take the Q10-based k=3 cluster assignment (clusters/clusters_k3.csv) and count
how many distinct Q10 / Q50 / Q100 flood events fall in each cluster — to gauge
whether the SAME clustering could support Q50 and Q100 analyses.

Events: IV streamflow (2002-2022) exceedances of each Q threshold, declustered
by a 24-hour gap — identical rule to 03c_cluster_stations.py, so the Q10 numbers
match the clustering decision.

Usage:
    python scripts/count_events_per_cluster.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
START, END = "2002-01-01", "2022-12-31"
RPS = [10, 50, 100]
MIN_CLUSTER_EVENTS = 30          # 03c viability threshold (per cluster)


def _read_parquet(key: str, columns=None) -> pd.DataFrame:
    o = boto3.client("s3").get_object(Bucket=BUCKET, Key=PREFIX + key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def count_events(sf: pd.DataFrame, thr: pd.Series) -> pd.Series:
    """Distinct declustered events per site for a Q-threshold Series[site_no]."""
    thr_df = thr.rename("thr").reset_index()                  # site_no, thr
    m = sf.merge(thr_df, on="site_no", how="inner")
    ex = m[m["value_cfs"] >= m["thr"]].sort_values(["site_no", "datetime"]).copy()
    ex["prev"] = ex.groupby("site_no")["datetime"].shift(1)
    ex["gap"]  = (ex["datetime"] - ex["prev"]).dt.total_seconds() / 3600
    ex["new"]  = ex["prev"].isna() | (ex["gap"] > 24)
    return ex.groupby("site_no")["new"].sum()


def main() -> None:
    s3 = boto3.client("s3")
    o = s3.get_object(Bucket=BUCKET, Key=PREFIX + "clusters/clusters_k3.csv")
    clusters = pd.read_csv(io.BytesIO(o["Body"].read()), dtype={"site_no": str})
    print(f"Clustered stations: {len(clusters)}  "
          f"({clusters['cluster'].nunique()} clusters)")

    flow = _read_parquet("flow_stats/per_gauge_flow_stats.parquet",
                         ["site_no"] + [f"Q{rp}" for rp in RPS])
    flow["site_no"] = flow["site_no"].astype(str)

    print("Loading IV streamflow (2002-2022)...")
    sf = _read_parquet("streamflow/instantaneous/all_gauges_long.parquet",
                       ["site_no", "datetime", "value_cfs"])
    sf["site_no"]   = sf["site_no"].astype(str)
    sf["datetime"]  = pd.to_datetime(sf["datetime"], utc=True)
    sf["value_cfs"] = pd.to_numeric(sf["value_cfs"], errors="coerce")
    sf = sf[(sf["datetime"] >= pd.Timestamp(START, tz="UTC"))
            & (sf["datetime"] <= pd.Timestamp(END, tz="UTC"))]
    sf = sf[sf["site_no"].isin(set(clusters["site_no"]))]

    per = clusters.copy()
    for rp in RPS:
        thr = flow.set_index("site_no")[f"Q{rp}"].dropna()
        per = per.merge(count_events(sf, thr).rename(f"Q{rp}"), on="site_no", how="left")
    for rp in RPS:
        per[f"Q{rp}"] = per[f"Q{rp}"].fillna(0).astype(int)

    # ── Per-cluster summary: total events and #stations-with-events per RP ─────
    rows = []
    for c, g in per.groupby("cluster"):
        row = {"cluster": int(c), "n_stations": g["site_no"].nunique()}
        for rp in RPS:
            row[f"Q{rp}_events"] = int(g[f"Q{rp}"].sum())
            row[f"Q{rp}_stns"]   = int((g[f"Q{rp}"] > 0).sum())
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("cluster").sort_index()

    print("\n=== Events per cluster (IV 2002-2022, 24-h declustered) ===")
    print(summary.to_string())
    print("\nState totals:", {f"Q{rp}": int(per[f"Q{rp}"].sum()) for rp in RPS})

    print(f"\nViability vs MIN_CLUSTER_EVENTS={MIN_CLUSTER_EVENTS} (per cluster):")
    for rp in RPS:
        col = f"Q{rp}_events"
        ok  = int((summary[col] >= MIN_CLUSTER_EVENTS).sum())
        print(f"  Q{rp:3d}: {ok}/{len(summary)} clusters >= {MIN_CLUSTER_EVENTS}  "
              f"(min cluster = {int(summary[col].min())}, "
              f"clusters with 0 = {int((summary[col] == 0).sum())})")


if __name__ == "__main__":
    main()
