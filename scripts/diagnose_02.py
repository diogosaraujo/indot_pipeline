"""diagnose_02.py

Diagnostic for script 02: checks station inventory vs S3 contents,
then samples a few missing stations to show what the USGS API returns.

Usage (from scripts/ folder):
    python diagnose_02.py [--sample N]
"""
from __future__ import annotations

import argparse
import io
import sys

import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, s3_client
from dataretrieval import waterdata


def list_s3_per_gauge_keys(bucket: str, prefix: str) -> set[str]:
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}streamflow/instantaneous/per_gauge/"):
        for obj in page.get("Contents", []):
            # key looks like .../per_gauge/03324000.parquet
            keys.add(obj["Key"].split("/")[-1].replace(".parquet", ""))
    return keys


def read_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def probe_site(site_no: str, begin: str, end: str) -> None:
    monitoring_id = f"USGS-{site_no}" if not site_no.startswith("USGS-") else site_no
    # Only probe the first window (first 3 years) to keep it fast
    time_string = f"{begin[:10]}/{end[:10]}"
    print(f"  Probing {site_no}: time={time_string}")
    try:
        df, meta = waterdata.get_continuous(
            monitoring_location_id=monitoring_id,
            parameter_code="00060",
            time=time_string,
        )
        if df is None or df.empty:
            print(f"    -> API returned empty DataFrame")
        else:
            print(f"    -> {len(df)} rows, columns: {list(df.columns)}")
            if "time" not in df.columns:
                print(f"    -> WARNING: 'time' column missing — will be treated as no-data")
            if "value" not in df.columns:
                print(f"    -> WARNING: 'value' column missing — will be treated as no-data")
    except Exception as e:
        print(f"    -> ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5,
                        help="Number of missing stations to probe via the API (default: 5)")
    args = parser.parse_args()

    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    print("Reading station inventory from S3...")
    inv = read_inventory(bucket, prefix)
    inv["site_no"] = inv["site_no"].astype(str)
    total = len(inv)

    print(f"Listing per-gauge files already in S3...")
    done = list_s3_per_gauge_keys(bucket, prefix)

    inv_set = set(inv["site_no"].tolist())
    missing = inv_set - done
    extra = done - inv_set

    print()
    print(f"=== Summary ===")
    print(f"  Total stations in inventory : {total}")
    print(f"  Files already in S3         : {len(done)}")
    print(f"  Stations still missing      : {len(missing)}")
    if extra:
        print(f"  In S3 but not in inventory  : {len(extra)}  (unexpected)")

    if not missing:
        print("\nAll stations accounted for — nothing to do.")
        return

    # Show the missing stations
    missing_df = inv[inv["site_no"].isin(missing)].copy()
    missing_df["begin_date"] = missing_df["begin_date"].astype(str).str[:10]
    missing_df["end_date"] = missing_df["end_date"].astype(str).str[:10]
    print(f"\n=== First 20 missing stations ===")
    print(missing_df[["site_no", "station_nm", "begin_date", "end_date"]].head(20).to_string(index=False))

    # Probe a sample via the API
    sample_n = min(args.sample, len(missing_df))
    if sample_n == 0:
        return

    print(f"\n=== Probing {sample_n} missing stations via USGS API ===")
    sample = missing_df.head(sample_n)
    default_start = cfg["usgs"]["start_date"]
    default_end = pd.Timestamp.utcnow().date().isoformat()

    for row in sample.itertuples(index=False):
        begin = row.begin_date if pd.notna(row.begin_date) and row.begin_date != "nan" else default_start
        end = row.end_date if pd.notna(row.end_date) and row.end_date != "nan" else default_end
        # Clamp end to today
        end = min(end, default_end)
        # Clamp window to first 3 years for speed
        from datetime import date, timedelta
        start_d = date.fromisoformat(begin)
        end_d = min(date.fromisoformat(end), start_d + timedelta(days=3*365))
        probe_site(row.site_no, start_d.isoformat(), end_d.isoformat())


if __name__ == "__main__":
    main()
