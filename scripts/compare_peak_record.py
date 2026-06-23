"""compare_peak_record.py

Compare the IV record we hold against the USGS annual PEAK-FLOW record for each
station.  The peak service is the authoritative basis for flood-frequency
analysis (and for StreamStats gage statistics); it usually extends decades
before the telemetry-era instantaneous-value (IV) record we downloaded.

For every site it reports:
    iv_start / iv_n_wy     – from our bucket's IV parquet
    peak_start / peak_n    – from NWIS peak service (full annual peak series)
    extra_years            – how many more years the peak record adds

Output: console summary + s3://.../v1/analysis/peak_vs_iv_record.csv

Usage:
    python scripts/compare_peak_record.py
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import pyarrow.parquet as pq
import requests

BUCKET = "indot-bridge-pipeline"
PREFIX = "v1/"
IV_KEY = f"{PREFIX}streamflow/instantaneous/all_gauges_long.parquet"
PEAK_URL = "https://nwis.waterdata.usgs.gov/nwis/peak"
MAX_WORKERS = 8
TIMEOUT = 30


def fetch_peak(site_no: str) -> dict:
    """Query the NWIS peak-flow service (RDB) for one site."""
    out = {"site_no": site_no, "peak_start": None, "peak_end": None,
           "peak_n": 0, "peak_error": None}
    try:
        r = requests.get(
            PEAK_URL,
            params={"site_no": site_no, "agency_cd": "USGS", "format": "rdb"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        # RDB: comment lines start with '#'; then a header row + a format row.
        lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 3:
            return out                      # no peak data published
        header = lines[0].split("\t")
        rows   = [ln.split("\t") for ln in lines[2:]]   # skip the '5s/10d' format row
        tbl    = pd.DataFrame(rows, columns=header)
        if "peak_dt" not in tbl.columns:
            return out
        years = pd.to_datetime(tbl["peak_dt"], errors="coerce").dropna()
        # peak_va present (not just a date with null value)
        if "peak_va" in tbl.columns:
            valid = tbl["peak_va"].str.strip().replace("", pd.NA).notna()
            years = pd.to_datetime(tbl.loc[valid, "peak_dt"], errors="coerce").dropna()
        if len(years):
            out["peak_start"] = years.min().date()
            out["peak_end"]   = years.max().date()
            out["peak_n"]     = int(len(years))
    except Exception as e:
        out["peak_error"] = str(e)[:80]
    return out


def main() -> None:
    print("Loading IV record summary from bucket...")
    obj = boto3.client("s3").get_object(Bucket=BUCKET, Key=IV_KEY)
    df = pq.read_table(io.BytesIO(obj["Body"].read()),
                       columns=["site_no", "datetime", "value_cfs"]).to_pandas()
    df["site_no"]  = df["site_no"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["value_cfs"] > 0]
    wy = df["datetime"].dt.year + (df["datetime"].dt.month >= 10).astype(int)
    g  = df.groupby("site_no")
    iv = pd.DataFrame({
        "iv_start": g["datetime"].min().dt.date,
        "iv_end":   g["datetime"].max().dt.date,
        "iv_n_wy":  wy.groupby(df["site_no"]).nunique(),
    })

    sites = iv.index.tolist()
    print(f"Querying USGS peak service for {len(sites)} sites...")
    peaks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_peak, s): s for s in sites}
        for i, fut in enumerate(as_completed(futs), 1):
            peaks.append(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(sites)}")

    pk = pd.DataFrame(peaks).set_index("site_no")
    merged = iv.join(pk)
    merged["iv_start_yr"]   = pd.to_datetime(merged["iv_start"]).dt.year
    merged["peak_start_yr"] = pd.to_datetime(merged["peak_start"]).dt.year
    merged["extra_years"]   = (merged["iv_start_yr"] - merged["peak_start_yr"])
    merged = merged.sort_values("extra_years", ascending=False)

    has_peak = merged["peak_n"] > 0
    print(f"\nSites with a published peak record : {int(has_peak.sum())} / {len(merged)}")
    print(f"Peak record longer than IV (start earlier): "
          f"{int((merged['extra_years'] > 0).sum())}")
    print(f"\nExtra years the peak record adds vs our IV start (where peak exists):")
    print(merged.loc[has_peak, "extra_years"].describe().round(1).to_string())
    print(f"\nPeak annual-count distribution:")
    print(merged.loc[has_peak, "peak_n"].describe().round(1).to_string())
    print(f"Sites with >= 10 annual peaks: {int((merged['peak_n'] >= 10).sum())}")

    print(f"\nTop 15 — peak record reaches furthest before our IV:")
    cols = ["iv_start", "iv_n_wy", "peak_start", "peak_end", "peak_n", "extra_years"]
    print(merged.head(15)[cols].to_string())

    out_key = f"{PREFIX}analysis/peak_vs_iv_record.csv"
    boto3.client("s3").put_object(
        Bucket=BUCKET, Key=out_key,
        Body=merged.reset_index().to_csv(index=False).encode(),
        ContentType="text/csv",
    )
    print(f"\nFull table -> s3://{BUCKET}/{out_key}")


if __name__ == "__main__":
    main()
