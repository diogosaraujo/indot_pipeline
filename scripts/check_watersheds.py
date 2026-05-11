"""check_watersheds.py

Quick diagnostic: count watersheds downloaded by step 03 and break them
down by active / inactive station status.
"""
import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

s3 = boto3.client("s3")
BUCKET = "indot-bridge-pipeline"
PREFIX = "v1/"


def read_parquet(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


ws  = read_parquet(f"{PREFIX}watersheds/workspace_index.parquet")
inv = read_parquet(f"{PREFIX}stations/indiana_streamflow_sites.parquet")

print("=== workspace_index status counts ===")
print(ws["status"].value_counts().to_string())
print(f"\nTotal sites processed: {len(ws)}")

inv["is_active"] = pd.to_datetime(inv["end_date"], errors="coerce") >= "2018-01-01"
merged = ws.merge(inv[["site_no", "is_active"]], on="site_no", how="left")

downloaded = merged[merged["status"].isin(["ok", "skipped"])]
print(f"\n=== Downloaded watersheds (ok + skipped): {len(downloaded)} ===")
active   = downloaded["is_active"].fillna(False)
print(f"  Active   (end_date >= 2018-01-01): {int(active.sum())}")
print(f"  Inactive:                          {int((~active).sum())}")

not_downloaded = merged[~merged["status"].isin(["ok", "skipped"])]
if len(not_downloaded):
    print(f"\n=== Sites with no watershed file: {len(not_downloaded)} ===")
    print(not_downloaded["status"].value_counts().to_string())

# Active stations missing delineation
active_inv = inv[inv["is_active"]]
missing_active = active_inv[~active_inv["site_no"].isin(downloaded["site_no"])].copy()
print(f"\n=== Active stations missing delineation: {len(missing_active)} ===")
if len(missing_active):
    report = missing_active[["site_no", "station_nm", "end_date"]].reset_index(drop=True)
    report.index += 1
    print(report.to_string())
