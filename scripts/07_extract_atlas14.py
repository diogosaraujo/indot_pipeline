"""07_extract_atlas14.py

Download NOAA Atlas 14 precipitation frequency estimates from the NOAA
Precipitation Frequency Data Server (PFDS) API for every Indiana streamflow
gauge in the inventory.  DDF depends only on lat/lon, so it is fetched for the
FULL inventory by default (previously only the active inventory was covered,
which left ~52 LP3-fitted but discontinued gauges without DDF — the bottleneck
in 08c).  Resume-safe: only stations missing from the existing DDF parquet are
fetched, then appended.

    python scripts/07_extract_atlas14.py                 # full inventory, resume
    python scripts/07_extract_atlas14.py --inventory active
    python scripts/07_extract_atlas14.py --refetch       # re-fetch everything

Durations (matching MRMS hourly rolling windows used in script 08):
    1, 2, 3, 6, 12, 24 h and 2, 3, 4, 5, 7, 10, 20, 30, 45, 60 days

Return periods: 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000 years

Output schema (one row per site × duration × return period):
    site_no, duration_hr, return_period_yr, depth_in

Writes:
    s3://<bucket>/<prefix>atlas14/precipitation_frequency.parquet
"""
from __future__ import annotations

import argparse
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

FULL_INVENTORY_KEY   = "stations/indiana_streamflow_sites.parquet"
ACTIVE_INVENTORY_KEY = "stations/indiana_streamflow_sites_active.parquet"
OUTPUT_KEY           = "atlas14/precipitation_frequency.parquet"

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


def read_inventory(bucket: str, prefix: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=f"{prefix}{key}")
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def load_existing_atlas14(bucket: str, prefix: str) -> pd.DataFrame:
    """Existing DDF parquet (empty frame if none) — for resume-safe fetching."""
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=f"{prefix}{OUTPUT_KEY}")
        df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
        df["site_no"] = df["site_no"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", choices=["full", "active"], default="full",
                    help="which station inventory to cover (default: full — closes "
                         "the gap where DDF was only fetched for active gauges)")
    ap.add_argument("--refetch", action="store_true",
                    help="ignore existing DDF and re-fetch every station")
    args = ap.parse_args()

    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    key = FULL_INVENTORY_KEY if args.inventory == "full" else ACTIVE_INVENTORY_KEY
    inv = read_inventory(bucket, prefix, key)
    inv["site_no"] = inv["site_no"].astype(str)
    inv = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).drop_duplicates("site_no")

    existing = pd.DataFrame() if args.refetch else load_existing_atlas14(bucket, prefix)
    have = set(existing["site_no"]) if not existing.empty else set()
    todo = inv[~inv["site_no"].isin(have)]
    log.info("Inventory (%s): %d stations | already have DDF: %d | to fetch: %d",
             args.inventory, len(inv), len(have), len(todo))
    if todo.empty:
        log.info("Nothing to fetch — Atlas 14 already covers this inventory.")
        return

    new_records: list[pd.DataFrame] = []
    n = len(todo)
    for i, row in enumerate(todo.itertuples(index=False), 1):
        site = str(row.site_no)
        lat = float(row.dec_lat_va)
        lon = float(row.dec_long_va)
        try:
            data = fetch_atlas14(lat, lon)
            if data is None:
                log.warning("[%d/%d] %s: fetch failed after retries", i, n, site)
                continue
            df = parse_atlas14(site, data)
            if df.empty:
                log.warning("[%d/%d] %s: no records parsed", i, n, site)
            else:
                new_records.append(df)
                log.info("[%d/%d] %s: %d rows", i, n, site, len(df))
        except Exception as e:
            log.error("[%d/%d] %s failed: %s", i, n, site, e)
        time.sleep(REQUEST_PAUSE_SEC)

    parts = [existing] if not existing.empty else []
    if new_records:
        parts.append(pd.concat(new_records, ignore_index=True))
    if not parts:
        log.error("No Atlas 14 data retrieved.")
        return

    combined = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["site_no", "duration_hr", "return_period_yr"], keep="last")
    write_parquet_to_s3(combined, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info(
        "Wrote %s (%d rows, %d stations; +%d newly fetched)",
        OUTPUT_KEY, len(combined), combined["site_no"].nunique(), len(new_records),
    )


if __name__ == "__main__":
    main()
