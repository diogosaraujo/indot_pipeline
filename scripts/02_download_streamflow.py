"""02_download_streamflow.py

Download the full instantaneous/unit-value discharge time series
(parameterCd 00060) for every gauge identified in step 01. Writes one Parquet
per gauge to S3, plus a single combined long-format Parquet.

Per-gauge output schema:
    datetime, site_no, value_cfs, qualifier

Writes:
    s3://<bucket>/<prefix>streamflow/instantaneous/per_gauge/{site_no}.parquet
    s3://<bucket>/<prefix>streamflow/instantaneous/all_gauges_long.parquet
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataretrieval import nwis

from utils import (
    RetryPolicy,
    load_config,
    s3_client,
    s3_object_exists,
    with_retries,
    write_parquet_to_s3,
)

log = logging.getLogger("02_streamflow")


def read_station_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def fetch_one(site_no: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Instantaneous/unit-value discharge for one site, full record."""
    def _call():
        df, _ = nwis.get_iv(
            sites=site_no,
            parameterCd="00060",
            start=start,
            end=end,
            multi_index=False,
        )
        return df

    df = with_retries(_call, RetryPolicy(max_attempts=4, base_delay=3.0))
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "site_no", "value_cfs", "qualifier"])

    df = df.reset_index()
    if "datetime" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "datetime"})
    # Instantaneous-value columns are typically "00060" and "00060_cd".
    val_col = "00060" if "00060" in df.columns else None
    if val_col is None:
        val_col = next((c for c in df.columns if c.startswith("00060") and not c.endswith("_cd")), None)
    qua_col = "00060_cd" if "00060_cd" in df.columns else None
    if qua_col is None:
        qua_col = next((c for c in df.columns if c.startswith("00060") and c.endswith("_cd")), None)
    if val_col is None:
        return pd.DataFrame(columns=["datetime", "site_no", "value_cfs", "qualifier"])

    out = pd.DataFrame({
        "datetime": pd.to_datetime(df["datetime"], utc=True),
        "site_no": site_no,
        "value_cfs": pd.to_numeric(df[val_col], errors="coerce"),
        "qualifier": df[qua_col] if qua_col else None,
    })
    return out


def process_site(site_no: str, cfg: dict) -> tuple[str, int]:
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    key = f"{prefix}streamflow/instantaneous/per_gauge/{site_no}.parquet"
    if s3_object_exists(bucket, key):
        return site_no, -1  # already done

    df = fetch_one(
        site_no,
        cfg["usgs"]["start_date"],
        cfg["usgs"]["end_date"],
    )
    if df.empty:
        log.info("No data for %s", site_no)
        return site_no, 0
    write_parquet_to_s3(df, bucket, key)
    return site_no, len(df)


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    inv = read_station_inventory(bucket, prefix)
    site_list = inv["site_no"].astype(str).unique().tolist()
    log.info("Downloading streamflow for %d sites", len(site_list))

    max_workers = min(cfg["execution"]["max_workers_io"], 16)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_site, s, cfg): s for s in site_list}
        for fut in as_completed(futures):
            site = futures[fut]
            try:
                _, n = fut.result()
                completed += 1
                if n == -1:
                    log.info("[%d/%d] %s skipped (already in S3)", completed, len(site_list), site)
                else:
                    log.info("[%d/%d] %s -> %d rows", completed, len(site_list), site, n)
            except Exception as e:
                log.error("Site %s failed: %s", site, e)

    # Build the combined long-format file by concatenating all per-gauge Parquets.
    log.info("Building combined long-format Parquet...")
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    parts = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}streamflow/instantaneous/per_gauge/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            parts.append(pq.read_table(io.BytesIO(body)))
    if not parts:
        log.warning("No per-gauge parquet files found.")
        return
    combined = pa.concat_tables(parts)
    buf = io.BytesIO()
    pq.write_table(combined, buf, compression="zstd")
    buf.seek(0)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}streamflow/instantaneous/all_gauges_long.parquet",
        Body=buf.getvalue(),
    )
    log.info("Wrote combined: %d rows total", combined.num_rows)


if __name__ == "__main__":
    main()
