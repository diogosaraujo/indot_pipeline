"""check_record_length.py

Quick audit of the instantaneous-value (IV) streamflow record we actually hold
in the bucket: per-station start date, end date, span, observation count, and
number of distinct water years.

This answers "how much record feeds our at-site LP3 fit" — which is NOT the same
as the USGS annual-peak record length that StreamStats gage statistics use.

Usage:
    python scripts/check_record_length.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET = "indot-bridge-pipeline"
PREFIX = "v1/"
IV_KEY = f"{PREFIX}streamflow/instantaneous/all_gauges_long.parquet"


def main() -> None:
    print("Loading IV streamflow...")
    obj = boto3.client("s3").get_object(Bucket=BUCKET, Key=IV_KEY)
    df = pq.read_table(
        io.BytesIO(obj["Body"].read()),
        columns=["site_no", "datetime", "value_cfs"],
    ).to_pandas()

    df["site_no"]  = df["site_no"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["value_cfs"] > 0]

    # Water year: Oct-Sep
    wy = df["datetime"].dt.year + (df["datetime"].dt.month >= 10).astype(int)

    g = df.groupby("site_no")
    rec = pd.DataFrame({
        "start":   g["datetime"].min().dt.date,
        "end":     g["datetime"].max().dt.date,
        "span_yr": ((g["datetime"].max() - g["datetime"].min()).dt.days / 365.25).round(1),
        "n_obs":   g["value_cfs"].count(),
        "n_wy":    wy.groupby(df["site_no"]).nunique(),
    }).sort_values("span_yr", ascending=False)

    print(f"\nStations in IV dataset : {len(rec)}")
    print(f"Overall date range     : {df['datetime'].min().date()} -> {df['datetime'].max().date()}")

    print("\nSpan distribution (years):")
    print(rec["span_yr"].describe().round(1).to_string())

    print("\nWater-year distribution:")
    print(rec["n_wy"].describe().round(1).to_string())
    print(f"Stations with >= 10 water-years: {(rec['n_wy'] >= 10).sum()}")
    print(f"Stations with <  10 water-years: {(rec['n_wy'] <  10).sum()}")

    print(f"\nLongest 10 records:\n{rec.head(10).to_string()}")
    print(f"\nShortest 10 records:\n{rec.tail(10).to_string()}")

    # Save full table for inspection
    out_key = f"{PREFIX}analysis/record_length_audit.csv"
    boto3.client("s3").put_object(
        Bucket=BUCKET, Key=out_key,
        Body=rec.reset_index().to_csv(index=False).encode(),
        ContentType="text/csv",
    )
    print(f"\nFull table -> s3://{BUCKET}/{out_key}")


if __name__ == "__main__":
    main()
