#!/usr/bin/env python3
"""07c_extract_atlas14_stations.py

Atlas 14 DDF at the PRECIPITATION-STATION locations (ISD/GHCNh), keyed by
station_id, so the station-gauge trigger judges a gauge's rain against ITS OWN
climatology — fixing the mismatch where the station's rain was compared to the
streamgage's Atlas 14 (08h / 09l). Atlas 14 depends only on lat/lon, so this
reuses 07's PFDS fetch/parse at the station coordinates.

Reads:  precip/noaa/stations_isd.parquet, precip/noaa/stations_ghcnh.parquet
Writes: atlas14/precipitation_frequency_stations.parquet
        (station_id, duration_hr, return_period_yr, depth_in)   -- resume-safe

Usage:
    python scripts/07c_extract_atlas14_stations.py
"""
from __future__ import annotations

import importlib.util
import io
import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, s3_client, write_parquet_to_s3

# Reuse 07's PFDS fetch + parse (digit-prefixed → importlib).
_spec = importlib.util.spec_from_file_location("atlas14_07", Path(__file__).with_name("07_extract_atlas14.py"))
a07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a07)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("07c_atlas14_sta")

STATION_INV_KEYS = ["precip/noaa/stations_isd.parquet", "precip/noaa/stations_ghcnh.parquet"]
OUTPUT_KEY       = "atlas14/precipitation_frequency_stations.parquet"


def _read(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def load_existing(bucket, prefix) -> pd.DataFrame:
    try:
        df = _read(bucket, f"{prefix}{OUTPUT_KEY}")
        df["station_id"] = df["station_id"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    frames = []
    for key in STATION_INV_KEYS:
        try:
            frames.append(_read(bucket, f"{prefix}{key}", columns=["station_id", "latitude", "longitude"]))
        except Exception as e:                                   # noqa: BLE001
            log.warning("Could not read %s: %s", key, e)
    if not frames:
        log.error("No station inventories found.")
        return
    inv = pd.concat(frames, ignore_index=True)
    inv["station_id"] = inv["station_id"].astype(str)
    inv = inv.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")

    existing = load_existing(bucket, prefix)
    have = set(existing["station_id"]) if not existing.empty else set()
    todo = inv[~inv["station_id"].isin(have)]
    log.info("Precip stations: %d | already have DDF: %d | to fetch: %d", len(inv), len(have), len(todo))
    if todo.empty:
        log.info("Nothing to fetch — station Atlas 14 already complete.")
        return

    new: list[pd.DataFrame] = []
    n = len(todo)
    for i, row in enumerate(todo.itertuples(index=False), 1):
        sid = str(row.station_id)
        try:
            data = a07.fetch_atlas14(float(row.latitude), float(row.longitude))
            if data is None:
                log.warning("[%d/%d] %s: fetch failed after retries", i, n, sid)
                continue
            df = a07.parse_atlas14(sid, data).rename(columns={"site_no": "station_id"})
            if df.empty:
                log.warning("[%d/%d] %s: no records parsed", i, n, sid)
            else:
                new.append(df)
                log.info("[%d/%d] %s: %d rows", i, n, sid, len(df))
        except Exception as e:                                   # noqa: BLE001
            log.error("[%d/%d] %s failed: %s", i, n, sid, e)
        time.sleep(a07.REQUEST_PAUSE_SEC)

    parts = [existing] if not existing.empty else []
    if new:
        parts.append(pd.concat(new, ignore_index=True))
    if not parts:
        log.error("No Atlas 14 data retrieved.")
        return
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["station_id", "duration_hr", "return_period_yr"], keep="last")
    write_parquet_to_s3(combined, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s (%d rows, %d stations; +%d newly fetched)",
             OUTPUT_KEY, len(combined), combined["station_id"].nunique(), len(new))


if __name__ == "__main__":
    main()
