"""12_download_noaa_precip.py

Download hourly precipitation from two NOAA/NCEI sources:

  ISD / LCD (Integrated Surface Database — Local Climatological Data)
      Inventory  : NCEI ISD station-history CSV (no token required)
      Data       : NCEI LCD annual CSV files, one file per station per year
      URL pattern: ncei.noaa.gov/data/local-climatological-data/access/{YEAR}/{USAF}{WBAN}.csv
      Coverage   : ASOS/AWOS automated stations + some SYNOP; ~200–400 stations
                   inside the Indiana watershed union. HourlyPrecipitation column
                   gives values in inches; trace = "T".

  GHCNh (Global Historical Climatology Network – Hourly)
      Inventory  : NCEI station-list CSV (no token required); multiple URL fallbacks
      Data       : NCEI direct-download CSV, one file per station
      Coverage   : broader network including COOP co-ops; overlaps with ISD

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
    s3://<bucket>/<prefix>precip/noaa/isd_hourly.parquet    (optional)
    s3://<bucket>/<prefix>precip/noaa/ghcnh_hourly.parquet  (optional)

Writes:
    s3://<bucket>/<prefix>precip/noaa/stations_isd.parquet
    s3://<bucket>/<prefix>precip/noaa/stations_ghcnh.parquet
    s3://<bucket>/<prefix>precip/noaa/isd_hourly.parquet
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import botocore.exceptions
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import (
    build_watershed_union,
    filter_by_polygon,
    load_config,
    s3_client,
    write_parquet_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("12_noaa_precip")

# ── URLs ──────────────────────────────────────────────────────────────────────
ISD_HISTORY_URL  = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
LCD_DATA_BASE    = "https://www.ncei.noaa.gov/data/local-climatological-data/access"

# Try multiple known GHCNh paths — NCEI occasionally moves files between prefixes
GHCNH_LIST_URLS = [
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/doc/ghcnh-station-list.csv",
    "https://www.ncei.noaa.gov/pub/data/ghcn/hourly/ghcnh-station-list.csv",
    "https://www.ncei.noaa.gov/oa/climate/ghcn/hourly/ghcnh-station-list.csv",
]
GHCNH_DATA_BASES = [
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/access",
    "https://www.ncei.noaa.gov/pub/data/ghcn/hourly/access",
]

# Indiana bounding box fallback (minlat, minlon, maxlat, maxlon)
INDIANA_BBOX = (37.77, -88.10, 41.76, -84.78)


# ── ISD station inventory ─────────────────────────────────────────────────────

def fetch_isd_stations(start_date: str) -> pd.DataFrame:
    """Download NCEI ISD station history and return US stations active after start_date."""
    log.info("Downloading ISD station history from NCEI...")
    r = requests.get(ISD_HISTORY_URL, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Filter to US stations with valid coordinates
    df = df[df.get("CTRY", df.get("COUNTRY", pd.Series())) == "US"].copy()
    df["latitude"]  = pd.to_numeric(df.get("LAT",  df.get("LATITUDE",  pd.Series())), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("LON",  df.get("LONGITUDE", pd.Series())), errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Filter to stations still active after start_date
    end_col = "END" if "END" in df.columns else None
    if end_col:
        df[end_col] = pd.to_datetime(df[end_col].astype(str), format="%Y%m%d", errors="coerce")
        cutoff = pd.Timestamp(start_date)
        df = df[df[end_col] >= cutoff]

    # Build 11-char LCD station ID: zero-padded USAF (6) + WBAN (5)
    usaf = df.get("USAF", pd.Series([""] * len(df))).astype(str).str.strip().str.zfill(6)
    wban = df.get("WBAN", pd.Series(["99999"] * len(df))).astype(str).str.strip().str.zfill(5)
    df["station_id"] = usaf.values + wban.values

    # Drop stations with no WBAN (wban=99999) — these rarely have LCD files
    df = df[wban.values != "99999"]

    name_col = next((c for c in df.columns if "STATION" in c.upper() and "NAME" in c.upper()), None)
    df["name"] = df[name_col].str.strip() if name_col else ""

    out = df[["station_id", "name", "latitude", "longitude"]].drop_duplicates("station_id")
    log.info("ISD: %d US stations with WBAN, active since %s", len(out), start_date)
    return out.reset_index(drop=True)


# ── GHCNh station inventory ───────────────────────────────────────────────────

def fetch_ghcnh_stations() -> pd.DataFrame:
    """Try each known GHCNh station-list URL until one succeeds."""
    for url in GHCNH_LIST_URLS:
        try:
            log.info("Trying GHCNh station list: %s", url)
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                log.debug("  404, trying next URL")
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            log.debug("  %s, trying next URL", e)
            continue

        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]

        id_col  = next((c for c in df.columns if c in ("ID", "STATION_ID", "STATIONID")), None)
        lat_col = next((c for c in df.columns if c in ("LATITUDE", "LAT")), None)
        lon_col = next((c for c in df.columns if c in ("LONGITUDE", "LON")), None)
        nm_col  = next((c for c in df.columns if "NAME" in c), None)

        if not all([id_col, lat_col, lon_col]):
            log.warning("  Unexpected columns %s — skipping this URL", list(df.columns)[:8])
            continue

        out = pd.DataFrame({
            "station_id": df[id_col].str.strip(),
            "name":       df[nm_col].str.strip() if nm_col else "",
            "latitude":   pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude":  pd.to_numeric(df[lon_col], errors="coerce"),
        }).dropna(subset=["latitude", "longitude"])
        log.info("GHCNh station list: %d stations worldwide (from %s)", len(out), url)
        return out.reset_index(drop=True)

    raise RuntimeError(
        "Could not download GHCNh station list from any known URL.\n"
        "Known URLs tried:\n" + "\n".join(f"  {u}" for u in GHCNH_LIST_URLS)
    )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def get_max_dates(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.groupby("station_id")["datetime_utc"].max().to_dict()


def _find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def _parse_precip(series: pd.Series) -> pd.Series:
    """Convert a string precipitation column to float inches; trace → 0.001."""
    s = series.astype(str).str.strip()
    is_trace = s.str.upper() == "T"
    numeric = pd.to_numeric(s.str.extract(r"([\d.]+)")[0], errors="coerce")
    numeric = numeric.where(numeric < 9990, other=np.nan)   # sentinel cleanup
    numeric = numeric.where(numeric >= 0, other=np.nan)
    numeric[is_trace] = 0.001
    return numeric


# ── ISD / LCD data download ───────────────────────────────────────────────────

def download_lcd_station(
    station_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """Download NCEI LCD annual CSV files for all years in [start_dt, end_dt]."""
    frames: list[pd.DataFrame] = []
    for year in range(start_dt.year, end_dt.year + 1):
        url = f"{LCD_DATA_BASE}/{year}/{station_id}.csv"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            log.debug("LCD %s %d: %s", station_id, year, e)
            continue

        try:
            raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:
            log.debug("LCD %s %d: parse error %s", station_id, year, e)
            continue

        raw.columns = [c.strip() for c in raw.columns]
        date_col = _find_col(raw.columns, ["DATE", "DATE_TIME", "DATETIME"])
        prcp_col = _find_col(raw.columns, ["HourlyPrecipitation", "HPCP", "PRECIP"])
        if date_col is None or prcp_col is None:
            log.debug("LCD %s %d: no date/precip column in %s", station_id, year, list(raw.columns)[:8])
            continue

        raw[date_col] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
        raw = raw.dropna(subset=[date_col])
        raw = raw[(raw[date_col] >= start_dt) & (raw[date_col] <= end_dt)]
        if raw.empty:
            continue

        frames.append(pd.DataFrame({
            "station_id":   station_id,
            "datetime_utc": raw[date_col].values,
            "precip_in":    _parse_precip(raw[prcp_col]).values,
        }))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).dropna(subset=["datetime_utc"])


# ── GHCNh data download ───────────────────────────────────────────────────────

def download_ghcnh_station(
    station_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """Try each known GHCNh data base URL until one returns data."""
    for base in GHCNH_DATA_BASES:
        url = f"{base}/{station_id}.csv"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                continue
            r.raise_for_status()
        except requests.RequestException:
            continue

        try:
            raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception:
            continue

        raw.columns = [c.strip() for c in raw.columns]
        date_col = _find_col(raw.columns, ["DATE", "DATE_TIME", "DATETIME"])
        prcp_col = _find_col(
            raw.columns,
            ["HourlyPrecipitation", "precipitation_amount", "PRCP",
             "PCP01", "PRECIPITATION", "P01I"],
        )
        if date_col is None or prcp_col is None:
            log.debug("GHCNh %s: no date/precip column in %s", station_id, list(raw.columns)[:8])
            continue

        raw[date_col] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
        raw = raw.dropna(subset=[date_col])
        raw = raw[(raw[date_col] >= start_dt) & (raw[date_col] <= end_dt)]
        if raw.empty:
            return None

        return pd.DataFrame({
            "station_id":   station_id,
            "datetime_utc": raw[date_col].values,
            "precip_in":    _parse_precip(raw[prcp_col]).values,
        }).dropna(subset=["datetime_utc"]).reset_index(drop=True)

    return None


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
        if isinstance(eff_start, pd.Timestamp) and eff_start.tzinfo is not None:
            eff_start = eff_start + pd.Timedelta(hours=1)
        if pd.Timestamp(eff_start) >= end_dt:
            return None
        if source == "isd":
            return download_lcd_station(sid, pd.Timestamp(eff_start), end_dt)
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
        meta_cols = [c for c in ["station_id", "name", "latitude", "longitude"] if c in stations_meta.columns]
        new_df = new_df.merge(stations_meta[meta_cols], on="station_id", how="left")

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
    log.info("%s: wrote %d rows for %d stations.", label, len(combined), combined["station_id"].nunique())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    bucket      = cfg["aws"]["output_bucket"]
    prefix      = cfg["aws"]["output_prefix"]
    noaa_cfg    = cfg.get("noaa_precip", {})
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
        log.info("Watershed union bbox: %.3f,%.3f → %.3f,%.3f", miny, minx, maxy, maxx)
    else:
        log.info("No watershed files found — using Indiana bounding box.")

    # ── ISD / LCD ──────────────────────────────────────────────────────────
    log.info("Fetching ISD station inventory...")
    isd_all = fetch_isd_stations(start_str)
    isd_stations = filter_by_polygon(isd_all, polygon, fallback_bbox=INDIANA_BBOX)
    log.info("ISD stations in watershed: %d", len(isd_stations))
    write_parquet_to_s3(isd_stations, bucket, f"{prefix}precip/noaa/stations_isd.parquet")

    existing_isd  = _read_parquet_s3(bucket, f"{prefix}precip/noaa/isd_hourly.parquet")
    max_dates_isd = get_max_dates(existing_isd)
    new_isd = download_all(isd_stations, "isd", start_dt, end_dt, max_dates_isd, max_workers)
    _merge_and_write(
        new_isd, existing_isd, isd_stations,
        bucket, f"{prefix}precip/noaa/isd_hourly.parquet", "ISD/LCD",
    )

    # ── GHCNh ──────────────────────────────────────────────────────────────
    log.info("Fetching GHCNh station list from NCEI...")
    try:
        ghcnh_all = fetch_ghcnh_stations()
        ghcnh_stations = filter_by_polygon(ghcnh_all, polygon, fallback_bbox=INDIANA_BBOX)
        log.info("GHCNh stations in watershed: %d", len(ghcnh_stations))
        write_parquet_to_s3(ghcnh_stations, bucket, f"{prefix}precip/noaa/stations_ghcnh.parquet")

        existing_ghcnh  = _read_parquet_s3(bucket, f"{prefix}precip/noaa/ghcnh_hourly.parquet")
        max_dates_ghcnh = get_max_dates(existing_ghcnh)
        new_ghcnh = download_all(ghcnh_stations, "ghcnh", start_dt, end_dt, max_dates_ghcnh, max_workers)
        _merge_and_write(
            new_ghcnh, existing_ghcnh, ghcnh_stations,
            bucket, f"{prefix}precip/noaa/ghcnh_hourly.parquet", "GHCNh",
        )
    except Exception as e:
        log.error("GHCNh section failed: %s", e)

    log.info("Done.")


if __name__ == "__main__":
    main()
