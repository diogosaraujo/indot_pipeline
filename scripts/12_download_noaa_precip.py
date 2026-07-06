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
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
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

# GHCNh station list — primary URL confirmed working as of 2025; legacy paths kept
# as fallbacks in case NCEI reorganizes again.
GHCNH_LIST_URLS = [
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv",
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/doc/ghcnh-station-list.csv",
    "https://www.ncei.noaa.gov/pub/data/ghcn/hourly/ghcnh-station-list.csv",
]
# Per-year PSV files: {GHCNH_DATA_BASE}/{YEAR}/psv/GHCNh_{STATIONID}_{YEAR}.psv
GHCNH_DATA_BASE = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network"
    "/hourly/access/by-year"
)

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

        id_col  = next((c for c in df.columns if c in ("GHCN_ID", "ID", "STATION_ID", "STATIONID")), None)
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

_S3FS: "pafs.S3FileSystem | None" = None


def _s3fs() -> "pafs.S3FileSystem":
    global _S3FS
    if _S3FS is None:
        _S3FS = pafs.S3FileSystem()
    return _S3FS


def _to_utc(v):
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _existing_bounds(bucket: str, key: str) -> tuple[dict, dict]:
    """Per-station (max_datetime, min_datetime) for the gap-fill, read with only
    the two needed columns so we never materialize the whole (up to ~200M-row)
    hourly table in memory.  Returns ({sid: max}, {sid: min}); empty if absent."""
    try:
        tbl = pq.read_table(f"{bucket}/{key}", filesystem=_s3fs(),
                            columns=["station_id", "datetime_utc"])
    except (FileNotFoundError, OSError):
        return {}, {}
    if tbl.num_rows == 0:
        return {}, {}
    agg = tbl.group_by("station_id").aggregate(
        [("datetime_utc", "max"), ("datetime_utc", "min")])
    sids = agg.column("station_id").to_pylist()
    maxs = agg.column("datetime_utc_max").to_pylist()
    mins = agg.column("datetime_utc_min").to_pylist()
    return ({s: _to_utc(m) for s, m in zip(sids, maxs) if m is not None},
            {s: _to_utc(m) for s, m in zip(sids, mins) if m is not None})


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
    """Download GHCNh per-year PSV files for each year in [start_dt, end_dt].

    NCEI reorganised GHCNh in 2024: data is now split by calendar year and
    served as pipe-separated files at:
      {GHCNH_DATA_BASE}/{YEAR}/psv/GHCNh_{STATIONID}_{YEAR}.psv
    """
    frames: list[pd.DataFrame] = []
    for year in range(start_dt.year, end_dt.year + 1):
        url = f"{GHCNH_DATA_BASE}/{year}/psv/GHCNh_{station_id}_{year}.psv"
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            log.debug("GHCNh %s %d: %s", station_id, year, e)
            continue

        try:
            raw = pd.read_csv(io.StringIO(r.text), sep="|", low_memory=False)
        except Exception as e:
            log.debug("GHCNh %s %d: parse error %s", station_id, year, e)
            continue

        raw.columns = [c.strip() for c in raw.columns]
        date_col = _find_col(
            raw.columns,
            ["DATE", "date", "DATE_TIME", "DATETIME", "datetime"],
        )
        prcp_col = _find_col(
            raw.columns,
            ["precipitation_amount", "HourlyPrecipitation", "PRCP",
             "PCP01", "PRECIPITATION", "P01I", "p01i", "HPCP"],
        )
        if date_col is None or prcp_col is None:
            log.debug(
                "GHCNh %s %d: no date/precip column — found %s",
                station_id, year, list(raw.columns)[:10],
            )
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
    return pd.concat(frames, ignore_index=True).dropna(subset=["datetime_utc"]).reset_index(drop=True)


# ── Concurrent download orchestration ────────────────────────────────────────

def download_all(
    stations: pd.DataFrame,
    source: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_dates: dict,
    min_dates: dict,
    max_workers: int = 8,
) -> pd.DataFrame:
    def _fetch(sid, s, e) -> Optional[pd.DataFrame]:
        if source == "isd":
            return download_lcd_station(sid, s, e)
        return download_ghcnh_station(sid, s, e)

    def _worker(row) -> Optional[pd.DataFrame]:
        sid = row["station_id"]
        parts: list[pd.DataFrame] = []

        # Historical gap: existing data starts after configured start_dt
        existing_min = min_dates.get(sid)
        if existing_min is not None:
            existing_min = pd.Timestamp(existing_min)
            if existing_min > start_dt + pd.Timedelta(hours=1):
                hist_end = existing_min - pd.Timedelta(hours=1)
                df = _fetch(sid, start_dt, hist_end)
                if df is not None and not df.empty:
                    parts.append(df)

        # Recent gap: new data since existing max date
        eff_start = max_dates.get(sid, start_dt)
        if isinstance(eff_start, pd.Timestamp) and eff_start.tzinfo is not None:
            eff_start = eff_start + pd.Timedelta(hours=1)
        if pd.Timestamp(eff_start) < end_dt:
            df = _fetch(sid, pd.Timestamp(eff_start), end_dt)
            if df is not None and not df.empty:
                parts.append(df)

        return pd.concat(parts, ignore_index=True) if parts else None

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


def _row_keys(tbl: "pa.Table") -> "pa.Array":
    """station_id@epoch key for de-duplication on (station_id, datetime_utc)."""
    sid = pc.cast(tbl.column("station_id"), pa.string())
    ep  = pc.cast(pc.cast(tbl.column("datetime_utc"), pa.timestamp("us", "UTC")), pa.int64())
    return pc.binary_join_element_wise(sid, pc.cast(ep, pa.string()), "@").combine_chunks()


def _merge_and_write(
    new_df: pd.DataFrame,
    bucket: str,
    key: str,
    stations_meta: pd.DataFrame,
    label: str,
) -> None:
    """Merge freshly-downloaded rows into the existing hourly parquet and write it
    back — STREAMING one row-group at a time, so peak memory stays ~one batch plus
    the (small) new rows rather than the whole 200M-row table.  The pandas
    concat+drop_duplicates+sort_values version OOM-killed the process on GHCNh.

    Existing rows whose (station_id, datetime_utc) also appear in the new batch are
    dropped so the new value wins (matches the old keep="last").  Output is not
    globally re-sorted; downstream loaders sort as needed.
    """
    if new_df is None or new_df.empty:
        log.info("%s: no new rows; leaving existing parquet unchanged.", label)
        return

    meta_cols = [c for c in ["station_id", "name", "latitude", "longitude"]
                 if c in stations_meta.columns]
    new_df = new_df.merge(stations_meta[meta_cols], on="station_id", how="left")
    new_df["datetime_utc"] = pd.to_datetime(new_df["datetime_utc"], utc=True)

    fs = _s3fs()
    s3_path = f"{bucket}/{key}"
    try:
        pf = pq.ParquetFile(fs.open_input_file(s3_path))
        schema, have_existing = pf.schema_arrow, True
    except (FileNotFoundError, OSError):
        pf, schema, have_existing = None, None, False

    if have_existing:
        new_tbl = pa.Table.from_pandas(
            new_df[[f.name for f in schema]], schema=schema, preserve_index=False)
    else:
        new_tbl = pa.Table.from_pandas(new_df, preserve_index=False)
        schema = new_tbl.schema

    new_keys = _row_keys(new_tbl)
    tmpdir = tempfile.mkdtemp(prefix="noaa_merge_")
    tmp = f"{tmpdir}/out.parquet"
    n = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
            if have_existing:
                for batch in pf.iter_batches(batch_size=1_000_000):
                    bt = pa.Table.from_batches([batch], schema=pf.schema_arrow)
                    bt = bt.filter(pc.invert(pc.is_in(_row_keys(bt), value_set=new_keys)))
                    if bt.num_rows:
                        w.write_table(bt)
                        n += bt.num_rows
            w.write_table(new_tbl)
            n += new_tbl.num_rows
        s3_client().upload_file(tmp, bucket, key)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    log.info("%s: wrote %d rows (streamed, +%d new) to s3://%s/%s",
             label, n, len(new_df), bucket, key)


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

    isd_key = f"{prefix}precip/noaa/isd_hourly.parquet"
    max_dates_isd, min_dates_isd = _existing_bounds(bucket, isd_key)
    new_isd = download_all(isd_stations, "isd", start_dt, end_dt, max_dates_isd, min_dates_isd, max_workers)
    _merge_and_write(new_isd, bucket, isd_key, isd_stations, "ISD/LCD")

    # ── GHCNh ──────────────────────────────────────────────────────────────
    log.info("Fetching GHCNh station list from NCEI...")
    try:
        ghcnh_all = fetch_ghcnh_stations()
        ghcnh_stations = filter_by_polygon(ghcnh_all, polygon, fallback_bbox=INDIANA_BBOX)
        log.info("GHCNh stations in watershed: %d", len(ghcnh_stations))
        write_parquet_to_s3(ghcnh_stations, bucket, f"{prefix}precip/noaa/stations_ghcnh.parquet")

        ghcnh_key = f"{prefix}precip/noaa/ghcnh_hourly.parquet"
        max_dates_ghcnh, min_dates_ghcnh = _existing_bounds(bucket, ghcnh_key)
        new_ghcnh = download_all(ghcnh_stations, "ghcnh", start_dt, end_dt, max_dates_ghcnh, min_dates_ghcnh, max_workers)
        _merge_and_write(new_ghcnh, bucket, ghcnh_key, ghcnh_stations, "GHCNh")
    except Exception as e:
        log.error("GHCNh section failed: %s", e)

    log.info("Done.")


if __name__ == "__main__":
    main()
