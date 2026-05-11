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
    report = missing_active[["site_no", "station_nm", "end_date"]].merge(
        ws[["site_no", "status"]], on="site_no", how="left"
    ).reset_index(drop=True)
    report["status"] = report["status"].fillna("not_in_workspace_index")
    report.index += 1
    print(report.to_string())

# Probe NLDI for all 21 not_found sites
import requests
NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
not_found_sites = ws[ws["status"] == "not_found"]["site_no"].tolist()
if not_found_sites:
    print(f"\n=== NLDI probe for all {len(not_found_sites)} not_found sites ===")
    for site_no in not_found_sites:
        url = f"{NLDI_BASE}/USGS-{site_no}/basin"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            features = r.json().get("features", [])
            note = f"OK — {len(features)} feature(s) returned"
        elif r.status_code == 404:
            r2 = requests.get(f"{NLDI_BASE}/USGS-{site_no}", timeout=30)
            if r2.status_code == 404:
                note = "404 — site not in NLDI (not on NHDPlus network)"
            else:
                # Site is known — try basin via its NHDPlus COMID
                comid = None
                try:
                    features = r2.json().get("features", [])
                    if features:
                        comid = features[0].get("properties", {}).get("comid")
                except Exception:
                    pass
                if comid:
                    comid_url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/basin"
                    r3 = requests.get(comid_url, timeout=30)
                    if r3.status_code == 200:
                        feats = r3.json().get("features", [])
                        note = f"no basin via nwissite, but COMID {comid} basin OK ({len(feats)} feature(s))"
                    else:
                        note = f"no basin via nwissite or COMID {comid} (HTTP {r3.status_code})"
                else:
                    note = "site known to NLDI but COMID not found in response"
        else:
            note = f"HTTP {r.status_code}"
        print(f"  {site_no}: {note}")
