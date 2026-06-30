"""cluster_basin_summary.py

Per-cluster summary of basin characteristics — especially the Kirpich time of
concentration (tc_hr) — to compare against the chosen trigger durations
(Cluster 0 ~12h, Cluster 1 ~1d, Cluster 2 ~3d).

Usage:
    python scripts/cluster_basin_summary.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"


def _read_parquet(key: str) -> pd.DataFrame:
    o = boto3.client("s3").get_object(Bucket=BUCKET, Key=PREFIX + key)
    return pq.read_table(io.BytesIO(o["Body"].read())).to_pandas()


def main() -> None:
    o = boto3.client("s3").get_object(Bucket=BUCKET, Key=PREFIX + "clusters/clusters_k3.csv")
    cl = pd.read_csv(io.BytesIO(o["Body"].read()), dtype={"site_no": str})

    b = _read_parquet("watersheds/basin_characteristics.parquet")
    b["site_no"] = b["site_no"].astype(str)

    m = cl.merge(b, on="site_no", how="left")

    g = m.groupby("cluster").agg(
        n=("site_no", "nunique"),
        tc_hr_mean=("tc_hr", "mean"),
        tc_hr_median=("tc_hr", "median"),
        tc_hr_min=("tc_hr", "min"),
        tc_hr_max=("tc_hr", "max"),
        area_mi2_median=("drain_area_mi2", "median"),
        slope_ftmi_median=("slope_ft_mi", "median"),
        pct_u_mean=("pct_u", "mean"),
    ).round(2)

    print("Per-cluster basin characteristics:\n")
    print(g.to_string())
    print("\ntc_hr = Kirpich time of concentration (hours).")
    print("Chosen trigger durations:  Cluster 0 = 12h,  Cluster 1 = 24h (1d),  Cluster 2 = 72h (3d)")


if __name__ == "__main__":
    main()
