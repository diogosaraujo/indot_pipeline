"""07_extract_atlas14.py

Download NOAA Atlas 14 precipitation frequency estimates from the NOAA
Precipitation Frequency Data Server (PFDS) API for every active Indiana
streamflow gauge.

Durations (matching MRMS hourly rolling windows used in script 08):
    1, 2, 3, 6, 12, 24 h and 2, 3, 4, 5, 7, 10, 20, 30, 45, 60 days

Return periods: 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000 years

Output schema (one row per site × duration × return period):
    site_no, duration_hr, return_period_yr, depth_in

Writes:
    s3://<bucket>/<prefix>atlas14/precipitation_frequency.parquet
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import RetryPolicy, load_config, s3_client, with_retries, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("07_atlas14")

PFDS_URL = "https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py"
REQUEST_PAUSE_SEC = 1.0

# Standard Atlas 14 Volume 2 (Ohio River Basin) duration sequence in order.
# The PFDS API does not return duration labels — order is fixed by the dataset.
ATLAS14_DURATION_LABELS = [
    "5-min", "10-min", "15-min", "30-min", "60-min",
    "2-hr",  "3-hr",   "6-hr",   "12-hr",  "24-hr",
    "2-day", "3-day",  "4-day",  "5-day",  "7-day",
    "10-day","20-day", "30-day", "45-day", "60-day",
]

# Subset we care about (>= 1h, matching MRMS hourly resolution)
DURATION_MAP: dict[str, int] = {
    "60-min": 1,
    "2-hr":   2,   "3-hr":   3,   "6-hr":   6,   "12-hr":  12,
    "24-hr":  24,  "2-day":  48,  "3-day":  72,  "4-day":  96,
    "5-day":  120, "7-day":  168, "10-day": 240, "20-day": 480,
    "30-day": 720, "45-day": 1080,"60-day": 1440,
}

RETURN_PERIODS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]


def read_active_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites_active.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def fetch_atlas14(lat: float, lon: float) -> Optional[str]:
    """Query the NOAA PFDS API and return the raw response text.

    The API returns JavaScript-style variable assignments, not JSON.
    """
    params = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "type": "pf",
        "data": "depth",
        "units": "english",
        "series": "pds",
    }

    def _call():
        r = requests.get(PFDS_URL, params=params, timeout=30)
        r.raise_for_status()
        if not r.text.strip():
            raise ValueError("Empty response from Atlas 14 API")
        return r.text

    return with_retries(_call, RetryPolicy(max_attempts=4, base_delay=5.0))


def parse_atlas14(site_no: str, text: str) -> pd.DataFrame:
    """Parse PFDS JavaScript-format response into long-format rows.

    The response contains variable assignments like:
        quantiles = [['0.387', ...], ...];
    Duration labels are not returned — order follows ATLAS14_DURATION_LABELS.
    """
    match = re.search(r"quantiles\s*=\s*(\[\[.*?\]\]);", text, re.DOTALL)
    if not match:
        log.warning("Site %s: could not find quantiles in response", site_no)
        return pd.DataFrame()

    quantiles = json.loads(match.group(1).replace("'", '"'))
    n_durations = len(quantiles)
    expected = len(ATLAS14_DURATION_LABELS)
    if n_durations != expected:
        log.warning(
            "Site %s: got %d duration rows, expected %d — "
            "using last %d labels from ATLAS14_DURATION_LABELS",
            site_no, n_durations, expected, n_durations,
        )
    duration_labels = ATLAS14_DURATION_LABELS[-n_durations:]

    records = []
    for dur_label, depths in zip(duration_labels, quantiles):
        if dur_label not in DURATION_MAP:
            continue
        duration_hr = DURATION_MAP[dur_label]
        for rp, depth in zip(RETURN_PERIODS, depths):
            records.append({
                "site_no": site_no,
                "duration_hr": duration_hr,
                "return_period_yr": rp,
                "depth_in": float(depth),
            })

    return pd.DataFrame.from_records(records)


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    inv = read_active_inventory(bucket, prefix)
    inv["site_no"] = inv["site_no"].astype(str)
    log.info("Fetching Atlas 14 data for %d active stations", len(inv))

    all_records: list[pd.DataFrame] = []
    for i, row in enumerate(inv.itertuples(index=False), 1):
        site = str(row.site_no)
        lat = float(row.dec_lat_va)
        lon = float(row.dec_long_va)

        try:
            data = fetch_atlas14(lat, lon)
            if data is None:
                log.warning("[%d/%d] %s: fetch failed after retries", i, len(inv), site)
                continue
            df = parse_atlas14(site, data)
            if df.empty:
                log.warning("[%d/%d] %s: no records parsed", i, len(inv), site)
            else:
                all_records.append(df)
                log.info("[%d/%d] %s: %d rows", i, len(inv), site, len(df))
        except Exception as e:
            log.error("[%d/%d] %s failed: %s", i, len(inv), site, e)

        time.sleep(REQUEST_PAUSE_SEC)

    if not all_records:
        log.error("No Atlas 14 data retrieved.")
        return

    combined = pd.concat(all_records, ignore_index=True)
    write_parquet_to_s3(combined, bucket, f"{prefix}atlas14/precipitation_frequency.parquet")
    log.info(
        "Wrote atlas14/precipitation_frequency.parquet (%d rows, %d stations)",
        len(combined), combined["site_no"].nunique(),
    )


if __name__ == "__main__":
    main()
