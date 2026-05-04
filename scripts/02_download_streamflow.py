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
import os
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataretrieval import waterdata

from utils import (
    RetryPolicy,
    load_config,
    s3_client,
    s3_object_exists,
    with_retries,
    write_parquet_to_s3,
)

log = logging.getLogger("02_streamflow")
WATERDATA_MAX_YEARS = 3
REQUEST_PAUSE_SEC = 0.5


def read_station_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def _parse_end_date(end: Optional[str]) -> str:
    if end is None:
        return pd.Timestamp.utcnow().date().isoformat()
    return end


def _iter_time_windows(start: str, end: Optional[str]) -> list[tuple[str, str]]:
    """Split a long IV request into <=3-year windows for waterdata.get_continuous()."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(_parse_end_date(end))
    windows: list[tuple[str, str]] = []
    current = start_date
    while current <= end_date:
        next_start = (pd.Timestamp(current) + pd.DateOffset(years=WATERDATA_MAX_YEARS)).date()
        window_end = min(end_date, next_start - timedelta(days=1))
        windows.append((current.isoformat(), window_end.isoformat()))
        current = window_end + timedelta(days=1)
    return windows


def fetch_one(site_no: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Instantaneous/unit-value discharge for one site, full record."""
    monitoring_location_id = site_no if site_no.startswith("USGS-") else f"USGS-{site_no}"
    frames: list[pd.DataFrame] = []
    windows = _iter_time_windows(start, end)

    for i, (window_start, window_end) in enumerate(windows, 1):
        time_string = f"{window_start}/{window_end}"
        log.info(
            "Site %s window %d/%d: %s",
            site_no,
            i,
            len(windows),
            time_string,
        )

        def _call():
            df, _ = waterdata.get_continuous(
                monitoring_location_id=monitoring_location_id,
                parameter_code="00060",
                time=time_string,
            )
            return df

        df = with_retries(_call, RetryPolicy(max_attempts=4, base_delay=3.0))
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(REQUEST_PAUSE_SEC)

    if not frames:
        return pd.DataFrame(columns=["datetime", "site_no", "value_cfs", "qualifier"])

    df = pd.concat(frames, ignore_index=True)
    if "time" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["datetime", "site_no", "value_cfs", "qualifier"])

    out = pd.DataFrame({
        "datetime": pd.to_datetime(df["time"], utc=True),
        "site_no": site_no,
        "value_cfs": pd.to_numeric(df["value"], errors="coerce"),
        "qualifier": df["qualifier"] if "qualifier" in df.columns else None,
    })
    out = out.drop_duplicates().sort_values("datetime").reset_index(drop=True)
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
    pat = os.getenv("API_USGS_PAT")
    if pat:
        import dataretrieval.utils as _dru
        _dru.session.headers.update({"Authorization": f"Bearer {pat}"})
        log.info("Using USGS API token from API_USGS_PAT")
    else:
        log.warning("API_USGS_PAT is not set; Water Data API requests may be more rate-limited")

    inv = read_station_inventory(bucket, prefix)
    site_list = inv["site_no"].astype(str).unique().tolist()
    log.info("Downloading streamflow for %d sites", len(site_list))
    log.info(
        "Processing streamflow serially, one station at a time, in <=%d-year windows",
        WATERDATA_MAX_YEARS,
    )

    for i, site in enumerate(site_list, 1):
        try:
            _, n = process_site(site, cfg)
            if n == -1:
                log.info("[%d/%d] %s skipped (already in S3)", i, len(site_list), site)
            else:
                log.info("[%d/%d] %s -> %d rows", i, len(site_list), site, n)
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
