"""12_download_noaa_precip.py

Download hourly precipitation from two NOAA/NCEI sources:

  COOP Hourly Precipitation (HPD v2)
      Inventory  : NOAA CDO API (free token at https://www.ncdc.noaa.gov/cdo-web/token)
      Data       : NCEI direct-download CSV, one file per station
      URL pattern: ncei.noaa.gov/data/coop-hourly-precipitation/v2/access/USC00{6-digit}.csv
      CDO ID fmt : COOP:XXXXXX  →  file prefix USC00XXXXXX

  GHCNh (Global Historical Climatology Network – Hourly)
      Inventory  : NCEI station-list CSV, no token required
      Data       : NCEI direct-download CSV, one file per station
      URL pattern: ncei.noaa.gov/data/global-historical-climatology-network-hourly/access/{id}.csv

Station selection (watershed union — Option A):
    Loads all per-gauge watershed GeoJSONs from S3 (written by script 03),
    unions them into one Shapely polygon, and keeps only NOAA stations that
    fall inside it.  Falls back to the Indiana bounding box if no watershed
    files are found.

Gap-filling:
    If a combined parquet already exists in S3, only data newer than the
    per-station max(datetime_utc) is downloaded.  Old + new rows are merged
    and the parquet is rewritten.

Reads:
    s3://<bucket>/<prefix>watersheds/per_gauge/*.geojson
    s3://<bucket>/<prefix>precip/noaa/coop_hourly.parquet   (optional)
    s3://<bucket>/<prefix>precip/noaa/ghcnh_hourly.parquet  (optional)

Writes:
    s3://<bucket>/<prefix>precip/noaa/stations_coop.parquet
    s3://<bucket>/<prefix>precip/noaa/stations_ghcnh.parquet
    s3://<bucket>/<prefix>precip/noaa/coop_hourly.parquet
    s3://<bucket>/<prefix>precip/noaa/ghcnh_hourly.parquet

Schema (both hourly parquets):
    station_id    str
    name          str
    latitude      float64
    longitude     float64
    datetime_utc  datetime64[ns, UTC]
    precip_in     float64    — trace stored as 0.001; NaN = missing/masked
"""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import botocore.exceptions
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import (
    RetryPolicy,
    build_watershed_union,
    filter_by_polygon,
    load_config,
    s3_client,
    with_retries,
    write_parquet_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("12_noaa_precip")

# ── URLs ──────────────────────────────────────────────────────────────────────
CDO_API_BASE     = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
HPD_DATA_BASE    = "https://www.ncei.noaa.gov/data/coop-hourly-precipitation/v2/access"
GHCNH_LIST_URL   = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/doc/ghcnh-station-list.csv"
GHCNH_DATA_BASE  = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/access"

# Indiana bounding box fallback (minlat, minlon, maxlat, maxlon)
INDIANA_BBOX = (37.77, -88.10, 41.76, -84.78)


# ── CDO API helpers ───────────────────────────────────────────────────────────

def _cdo_get(endpoint: str, token: str, params: dict) -> dict:
    headers = {"token": token}
    url = f"{CDO_API_BASE}/{endpoint}"

    def _call():
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(10)
            r2 = requests.get(url, headers=headers, params=params, timeout=30)
            r2.raise_for_status()
            return r2.json()
        r.raise_for_status()
        return r.json()

    return with_retries(_call, RetryPolicy(max_attempts=4, base_delay=2.0))


def fetch_cdo_stations(
    token: str,
    bbox: tuple[float, float, float, float],
    dataset: str = "PRECIP_HLY",
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Return COOP station inventory from CDO API for the given bbox."""
    extent = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    records: list[dict] = []
    offset = 1
    limit = 1000
    while True:
        params = dict(
            datasetid=dataset,
            extent=extent,
            startdate=start_date,
            limit=limit,
            offset=offset,
        )
        data = _cdo_get("stations", token, params)
        results = data.get("results") or []
        records.extend(results)
        meta = data.get("metadata", {}).get("resultset", {})
        total = int(meta.get("count", len(records)))
        log.info("CDO %s: fetched %d / %d", dataset, len(records), total)
        if len(records) >= total:
            break
        offset += limit
        time.sleep(0.25)

    if not records:
        return pd.DataFrame(columns=["station_id", "name", "latitude", "longitude", "min_date", "max_date"])

    df = pd.DataFrame(records)
    df = df.rename(columns={"id": "station_id", "mindate": "min_date", "maxdate": "max_date"})
    keep = [c for c in ["station_id", "name", "latitude", "longitude", "min_date", "max_date"] if c in df.columns]
    return df[keep].reset_index(drop=True)


# ── GHCNh station inventory ───────────────────────────────────────────────────

def fetch_ghcnh_stations() -> pd.DataFrame:
    """Download and parse the GHCNh station list CSV from NCEI."""
    log.info("Downloading GHCNh station list from NCEI...")
    r = requests.get(GHCNH_LIST_URL, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]

    id_col  = next((c for c in df.columns if c in ("ID", "STATION_ID", "STATIONID")), None)
    lat_col = next((c for c in df.columns if c in ("LATITUDE", "LAT")), None)
    lon_col = next((c for c in df.columns if c in ("LONGITUDE", "LON")), None)
    nm_col  = next((c for c in df.columns if "NAME" in c), None)

    if not all([id_col, lat_col, lon_col]):
        raise ValueError(f"Unexpected GHCNh station list columns: {list(df.columns)}")

    out = pd.DataFrame({
        "station_id": df[id_col].str.strip(),
        "name":       df[nm_col].str.strip() if nm_col else "",
        "latitude":   pd.to_numeric(df[lat_col], errors="coerce"),
        "longitude":  pd.to_numeric(df[lon_col], errors="coerce"),
    }).dropna(subset=["latitude", "longitude"])
    log.info("GHCNh station list: %d stations worldwide", len(out))
    return out.reset_index(drop=True)


# ── Existing-data gap-filling ─────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def get_max_dates(df: Optional[pd.DataFrame], id_col: str = "station_id") -> dict:
    """Return {station_id: pd.Timestamp (UTC)} for gap-filling."""
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.groupby(id_col)["datetime_utc"].max().to_dict()


# ── COOP data download ────────────────────────────────────────────────────────

def _cdo_id_to_ncei(cdo_id: str) -> str:
    """COOP:011084 → USC00011084."""
    return "USC00" + cdo_id.replace("COOP:", "").strip().zfill(6)


def _find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def download_coop_station(
    cdo_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> Optional[pd.DataFrame]:
    ncei_id = _cdo_id_to_ncei(cdo_id)
    url = f"{HPD_DATA_BASE}/{ncei_id}.csv"
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            log.debug("COOP %s (%s): 404 — not in HPD v2 direct access", cdo_id, ncei_id)
            return None
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("COOP %s: %s", cdo_id, e)
        return None

    try:
        raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as e:
        log.warning("COOP %s: CSV parse error: %s", cdo_id, e)
        return None

    raw.columns = [c.strip() for c in raw.columns]
    date_col = _find_col(raw.columns, ["DATE", "DATE_TIME", "DATETIME"])
    prcp_col = _find_col(raw.columns, ["HourlyPrecipitation", "HPCP", "PRECIP", "PRECIPITATION"])
    if date_col is None or prcp_col is None:
        log.warning("COOP %s: unexpected columns %s", cdo_id, list(raw.columns)[:8])
        return None

    raw[date_col] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw[(raw[date_col] >= start_dt) & (raw[date_col] <= end_dt)]
    if raw.empty:
        return None

    str_vals = raw[prcp_col].astype(str)
    is_trace = str_vals.str.upper().str.contains("T", na=False)
    numeric  = pd.to_numeric(str_vals.str.extract(r"([\d.]+)")[0], errors="coerce")
    # 9999.99 CDO sentinel → treat as trace
    numeric  = numeric.where(numeric < 9990, other=np.nan)
    numeric  = numeric.where(~is_trace, other=np.nan)  # overwrite below
    numeric[is_trace] = 0.001
    numeric  = numeric.where(numeric >= 0, other=np.nan)

    return pd.DataFrame({
        "station_id":   cdo_id,
        "datetime_utc": raw[date_col].values,
        "precip_in":    numeric.values,
    }).dropna(subset=["datetime_utc"]).reset_index(drop=True)


# ── GHCNh data download ───────────────────────────────────────────────────────

def download_ghcnh_station(
    station_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> Optional[pd.DataFrame]:
    url = f"{GHCNH_DATA_BASE}/{station_id}.csv"
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            log.debug("GHCNh %s: 404", station_id)
            return None
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("GHCNh %s: %s", station_id, e)
        return None

    try:
        raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as e:
        log.warning("GHCNh %s: CSV parse error: %s", station_id, e)
        return None

    raw.columns = [c.strip() for c in raw.columns]
    date_col = _find_col(raw.columns, ["DATE", "DATE_TIME", "DATETIME"])
    prcp_col = _find_col(
        raw.columns,
        ["HourlyPrecipitation", "precipitation_amount", "PRCP", "PCP01",
         "PRECIPITATION", "P01I", "AA1"],
    )
    if date_col is None or prcp_col is None:
        log.warning("GHCNh %s: unexpected columns %s", station_id, list(raw.columns)[:10])
        return None

    raw[date_col] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw[(raw[date_col] >= start_dt) & (raw[date_col] <= end_dt)]
    if raw.empty:
        return None

    str_vals = raw[prcp_col].astype(str)
    is_trace = str_vals.str.upper().str.contains("\\bT\\b", na=False, regex=True)
    numeric  = pd.to_numeric(str_vals.str.extract(r"([\d.]+)")[0], errors="coerce")
    numeric[is_trace] = 0.001
    numeric  = numeric.where(numeric >= 0, other=np.nan)

    return pd.DataFrame({
        "station_id":   station_id,
        "datetime_utc": raw[date_col].values,
        "precip_in":    numeric.values,
    }).dropna(subset=["datetime_utc"]).reset_index(drop=True)


# ── Concurrent download orchestration ────────────────────────────────────────

def download_all(
    stations: pd.DataFrame,
    source: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_dates: dict,
    max_workers: int = 8,
) -> pd.DataFrame:
    def _worker(row) -> Optional[pd.DataFrame]:
        sid = row["station_id"]
        eff_start = max_dates.get(sid, start_dt)
        # shift one hour past last known record to avoid duplicates
        if isinstance(eff_start, pd.Timestamp) and eff_start.tzinfo is not None:
            eff_start = eff_start + pd.Timedelta(hours=1)
        if pd.Timestamp(eff_start) >= end_dt:
            return None
        if source == "coop":
            return download_coop_station(sid, pd.Timestamp(eff_start), end_dt)
        return download_ghcnh_station(sid, pd.Timestamp(eff_start), end_dt)

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, row): row["station_id"] for _, row in stations.iterrows()}
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
                    log.info("[%d/%d] %s: %d rows", i, total, sid, len(df))
                else:
                    log.info("[%d/%d] %s: no new data", i, total, sid)
            except Exception as e:
                log.warning("[%d/%d] %s: %s", i, total, sid, e)

    if not frames:
        return pd.DataFrame(columns=["station_id", "datetime_utc", "precip_in"])
    return pd.concat(frames, ignore_index=True)


def _merge_and_write(
    new_df: pd.DataFrame,
    existing: Optional[pd.DataFrame],
    stations_meta: pd.DataFrame,
    bucket: str,
    key: str,
    label: str,
) -> None:
    if new_df.empty and (existing is None or existing.empty):
        log.info("%s: nothing to write.", label)
        return

    if not new_df.empty:
        meta_cols = ["station_id", "name", "latitude", "longitude"]
        meta_cols = [c for c in meta_cols if c in stations_meta.columns]
        new_df = new_df.merge(
            stations_meta[meta_cols], on="station_id", how="left"
        )

    parts = [p for p in [existing, new_df] if p is not None and not p.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    combined = (
        combined
        .drop_duplicates(subset=["station_id", "datetime_utc"], keep="last")
        .sort_values(["station_id", "datetime_utc"])
        .reset_index(drop=True)
    )
    write_parquet_to_s3(combined, bucket, key)
    log.info(
        "%s: wrote %d rows for %d stations.",
        label, len(combined), combined["station_id"].nunique(),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    bucket      = cfg["aws"]["output_bucket"]
    prefix      = cfg["aws"]["output_prefix"]
    noaa_cfg    = cfg.get("noaa_precip", {})
    cdo_token   = noaa_cfg.get("cdo_token", "")
    start_str   = noaa_cfg.get("start_date", "2020-10-14")
    end_str     = noaa_cfg.get("end_date") or datetime.utcnow().strftime("%Y-%m-%d")
    max_workers = cfg.get("execution", {}).get("max_workers_io", 8)

    start_dt = pd.Timestamp(start_str, tz="UTC")
    end_dt   = pd.Timestamp(end_str,   tz="UTC")

    # ── Watershed polygon ──────────────────────────────────────────────────
    log.info("Building watershed union from S3...")
    polygon = build_watershed_union(bucket, prefix)
    if polygon is not None:
        minx, miny, maxx, maxy = polygon.bounds
        bbox = (miny, minx, maxy, maxx)  # (minlat, minlon, maxlat, maxlon)
        log.info("Watershed union bbox: %.3f,%.3f → %.3f,%.3f", miny, minx, maxy, maxx)
    else:
        bbox = INDIANA_BBOX
        log.info("No watershed files found — using Indiana bounding box.")

    # ── COOP Hourly ────────────────────────────────────────────────────────
    if not cdo_token:
        log.warning(
            "noaa_precip.cdo_token is not set in config.yaml — skipping COOP Hourly.\n"
            "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token"
        )
    else:
        log.info("Fetching COOP station inventory from CDO API...")
        coop_all = fetch_cdo_stations(cdo_token, bbox, start_date=start_str)
        coop_stations = filter_by_polygon(coop_all, polygon, fallback_bbox=INDIANA_BBOX)
        log.info("COOP stations in watershed: %d", len(coop_stations))
        write_parquet_to_s3(coop_stations, bucket, f"{prefix}precip/noaa/stations_coop.parquet")

        existing_coop = _read_parquet_s3(bucket, f"{prefix}precip/noaa/coop_hourly.parquet")
        max_dates_coop = get_max_dates(existing_coop)
        new_coop = download_all(
            coop_stations, "coop", start_dt, end_dt, max_dates_coop, max_workers
        )
        _merge_and_write(
            new_coop, existing_coop, coop_stations,
            bucket, f"{prefix}precip/noaa/coop_hourly.parquet", "COOP Hourly",
        )

    # ── GHCNh ──────────────────────────────────────────────────────────────
    log.info("Fetching GHCNh station list from NCEI...")
    try:
        ghcnh_all = fetch_ghcnh_stations()
        ghcnh_stations = filter_by_polygon(ghcnh_all, polygon, fallback_bbox=INDIANA_BBOX)
        log.info("GHCNh stations in watershed: %d", len(ghcnh_stations))
        write_parquet_to_s3(ghcnh_stations, bucket, f"{prefix}precip/noaa/stations_ghcnh.parquet")

        existing_ghcnh = _read_parquet_s3(bucket, f"{prefix}precip/noaa/ghcnh_hourly.parquet")
        max_dates_ghcnh = get_max_dates(existing_ghcnh)
        new_ghcnh = download_all(
            ghcnh_stations, "ghcnh", start_dt, end_dt, max_dates_ghcnh, max_workers
        )
        _merge_and_write(
            new_ghcnh, existing_ghcnh, ghcnh_stations,
            bucket, f"{prefix}precip/noaa/ghcnh_hourly.parquet", "GHCNh",
        )
    except Exception as e:
        log.error("GHCNh section failed: %s", e, exc_info=True)

    log.info("Done.")


if __name__ == "__main__":
    main()
