"""check_dropped_stations.py

Show which LP3-fitted stations were dropped from clustering for missing basin
characteristics, and exactly which field(s) are null.

Usage:
    python scripts/check_dropped_stations.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
FEATURES = ["drain_area_mi2", "slope_ft_mi", "pct_u"]
ALL_BASIN_COLS = ["drain_area_mi2", "stream_length_mi", "slope_ft_mi",
                  "tc_hr", "pct_u", "pct_w"]


def _read(key: str, columns=None) -> pd.DataFrame:
    obj = boto3.client("s3").get_object(Bucket=BUCKET, Key=PREFIX + key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def main() -> None:
    flow = _read("flow_stats/per_gauge_flow_stats.parquet", ["site_no", "source"])
    flow["site_no"] = flow["site_no"].astype(str)
    fitted = set(flow.loc[flow["source"] == "lp3_peak_series", "site_no"])

    basin = _read("watersheds/basin_characteristics.parquet")
    basin["site_no"] = basin["site_no"].astype(str)

    fb = basin[basin["site_no"].isin(fitted)].copy()
    cols = [c for c in ALL_BASIN_COLS if c in fb.columns]

    # Stations missing ANY of the 3 clustering features
    missing_mask = fb[FEATURES].isna().any(axis=1)
    dropped = fb[missing_mask]

    print(f"Fitted stations: {len(fitted)}")
    print(f"With complete clustering features {FEATURES}: {len(fb) - len(dropped)}")
    print(f"Dropped for missing a feature: {len(dropped)}")

    # Also flag fitted stations entirely absent from the basin table
    absent = fitted - set(basin["site_no"])
    if absent:
        print(f"\nFitted but ABSENT from basin_characteristics ({len(absent)}): "
              f"{sorted(absent)}")

    if len(dropped):
        print("\nDropped stations and their basin characteristics "
              "(NaN = missing):")
        show = dropped[["site_no"] + cols].copy()
        print(show.to_string(index=False))
        print("\nWhich of the 3 clustering features is missing per station:")
        for _, r in dropped.iterrows():
            miss = [c for c in FEATURES if pd.isna(r[c])]
            print(f"  {r['site_no']}: missing {miss}")


if __name__ == "__main__":
    main()
