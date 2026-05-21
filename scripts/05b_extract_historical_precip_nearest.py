"""05b_extract_historical_precip_nearest.py

Extends the MRMS nearest-pixel record backwards using two sources:

  1. Iowa State University MRMS archive (~2015-01-01 onwards):
     Same QPE_01H_Pass2 product, same GRIB2 format and grid as noaa-mrms-pds.
     URL: https://mtarchive.geol.iastate.edu/{YYYY}/{MM}/{DD}/mrms/ncep/
          {FOLDER}/{FOLDER}_{YYYYMMDD}-{HH}0000.grib2.gz

  2. NOAA Stage IV Multi-Sensor QPE (2002-01-01 onwards, fallback):
     4-km HRAP polar-stereographic grid; units mm → converted to inches.
     URL: https://mesonet.agron.iastate.edu/archive/data/{YYYY}/{MM}/{DD}/stage4/
          ST4.{YYYYMMDDHH}.01h.grb[2]

For each hour: ISU MRMS is tried first; Stage IV is the fallback.

Gap-filling: reads the existing nearest_pixel.parquet to determine the
earliest available timestamp and only downloads dates before that.

Config key (config.yaml):
    historical_precip:
      start_date: "2002-01-01"   # earliest possible; Stage IV starts here
      end_date:   "2020-10-13"   # day before noaa-mrms-pds starts

Reads:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites_active.parquet
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/nearest_pixel.parquet   (optional)

Writes (prepends historical rows, sorted, deduped):
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/nearest_pixel.parquet

Output schema (identical to script 05):
    datetime_utc, site_no, value
"""
from __future__ import annotations

import io
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from utils import (
    apply_units,
    canonicalize_mrms_grid,
    decompress_gz,
    ensure_dir,
    load_config,
    open_mrms_grib,
    parse_iso_or_none,
    s3_client,
    write_parquet_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("05b_historical_nearest")

ISU_MRMS_BASE   = "https://mtarchive.geol.iastate.edu"
ISU_STAGE4_BASE = "https://mesonet.agron.iastate.edu/archive/data"

STAGE4_START    = date(2002,  1,  1)   # Stage IV operational start
ISU_MRMS_START  = date(2015,  1,  1)   # approximate ISU MRMS archive start
MRMS_PDS_START  = date(2020, 10, 14)   # noaa-mrms-pds archive start

MM_TO_IN = 1.0 / 25.4


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 60) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        return r.content if r.status_code == 200 else None
    except requests.RequestException:
        return None


def fetch_isu_mrms(folder: str, dt: datetime) -> bytes | None:
    fname = f"{folder}_{dt.strftime('%Y%m%d')}-{dt.strftime('%H')}0000.grib2.gz"
    url = (f"{ISU_MRMS_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}"
           f"/mrms/ncep/{folder}/{fname}")
    return _get(url)


def fetch_stage4(dt: datetime) -> bytes | None:
    base = (f"{ISU_STAGE4_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}/stage4")
    stem = f"ST4.{dt.strftime('%Y%m%d%H')}.01h"
    for ext in (".grb2", ".grb"):
        raw = _get(f"{base}/{stem}{ext}")
        if raw is not None:
            return raw
    return None


# ── Stage IV grid helpers ─────────────────────────────────────────────────────

def open_stage4_grib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a Stage IV GRIB file; return (data_mm, lats_2d, lons_2d).

    Tries cfgrib first (handles both GRIB1 and GRIB2), then eccodes.
    The HRAP grid lat/lon are 2-D arrays stored in the GRIB message.
    """
    import cfgrib
    datasets = cfgrib.open_datasets(str(path), indexpath="")
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            lat_key = next((k for k in da.coords if "lat" in k.lower()), None)
            lon_key = next((k for k in da.coords if "lon" in k.lower()), None)
            if lat_key and lon_key:
                data = da.values.astype(float)
                lats = da.coords[lat_key].values
                lons = da.coords[lon_key].values
                lons = np.where(lons > 180, lons - 360, lons)
                return data, lats, lons
    raise ValueError(f"No lat/lon variables found in Stage IV file: {path}")


_stage4_kdtree = None   # module-level cache; built once per process


def _get_stage4_kdtree(lats_2d: np.ndarray, lons_2d: np.ndarray):
    global _stage4_kdtree
    if _stage4_kdtree is None:
        from scipy.spatial import KDTree
        pts = np.column_stack([lats_2d.ravel(), lons_2d.ravel()])
        _stage4_kdtree = KDTree(pts)
    return _stage4_kdtree


def stage4_nearest_values(
    tree,
    data_mm: np.ndarray,
    gauge_lats: np.ndarray,
    gauge_lons: np.ndarray,
) -> np.ndarray:
    query = np.column_stack([gauge_lats, gauge_lons])
    _, idxs = tree.query(query, workers=1)
    vals = data_mm.ravel()[idxs].astype(float)
    return np.where(vals < 0, np.nan, vals * MM_TO_IN)


# ── Per-day worker ────────────────────────────────────────────────────────────

def process_day(args: tuple) -> str:
    (day_iso, folder, gauges_records, shard_dir) = args
    day   = date.fromisoformat(day_iso)
    lats  = np.array([float(g["dec_lat_va"])  for g in gauges_records])
    lons  = np.array([float(g["dec_long_va"]) for g in gauges_records])
    sids  = np.array([str(g["site_no"])        for g in gauges_records])

    rows: list[tuple] = []
    stage4_tree = None

    with tempfile.TemporaryDirectory() as scratch_str:
        scratch = Path(scratch_str)
        for hour in range(24):
            dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)

            raw    = None
            source = None

            if day >= ISU_MRMS_START:
                raw = fetch_isu_mrms(folder, dt)
                if raw:
                    source = "mrms"

            if raw is None and day >= STAGE4_START:
                raw = fetch_stage4(dt)
                if raw:
                    source = "stage4"

            if raw is None:
                continue

            try:
                if source == "mrms":
                    if raw[:2] == b"\x1f\x8b":
                        raw = decompress_gz(raw)
                    tmp = scratch / f"mrms_{hour:02d}.grib2"
                    tmp.write_bytes(raw)
                    ds = open_mrms_grib(tmp)
                    dvars = list(ds.data_vars)
                    if not dvars:
                        ds.close()
                        continue
                    arr, lats_g, lons_g = canonicalize_mrms_grid(ds[dvars[0]])
                    ds.close()
                    arr = apply_units(arr, kind="qpe", units="in")
                    lat_idx = np.abs(lats_g[:, None] - lats[None, :]).argmin(axis=0)
                    lon_idx = np.abs(lons_g[:, None] - lons[None, :]).argmin(axis=0)
                    vals = arr[lat_idx, lon_idx]

                else:  # stage4
                    tmp = scratch / f"stage4_{hour:02d}.grb"
                    tmp.write_bytes(raw)
                    data_mm, lats_2d, lons_2d = open_stage4_grib(tmp)
                    if stage4_tree is None:
                        stage4_tree = _get_stage4_kdtree(lats_2d, lons_2d)
                    vals = stage4_nearest_values(stage4_tree, data_mm, lats, lons)

            except Exception as e:
                log.debug("Error %s %s: %s", source, dt, e)
                continue

            for sid, v in zip(sids, vals):
                rows.append((dt, sid, float(v) if np.isfinite(v) else None))

    if not rows:
        return ""

    df = pd.DataFrame(rows, columns=["datetime_utc", "site_no", "value"])
    out = shard_dir / f"hist_nearest_{day:%Y%m%d}.parquet"
    df.to_parquet(out, compression="zstd")
    return str(out)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_gauges(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket,
        Key=f"{prefix}stations/indiana_streamflow_sites_active.parquet",
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def read_existing(bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    hist_cfg = cfg.get("historical_precip", {})
    cfg_start = parse_iso_or_none(hist_cfg.get("start_date", "")) or STAGE4_START
    cfg_end   = parse_iso_or_none(hist_cfg.get("end_date",   "")) or (MRMS_PDS_START - timedelta(days=1))

    product_key = cfg["mrms"]["products"][0]["key"]
    folder      = cfg["mrms"]["products"][0]["folder"]
    parquet_key = f"{prefix}mrms/{product_key}/nearest_pixel.parquet"

    gauges = read_gauges(bucket, prefix).dropna(
        subset=["dec_lat_va", "dec_long_va"]
    ).reset_index(drop=True)
    gauges_records = gauges[["site_no", "dec_lat_va", "dec_long_va"]].to_dict("records")
    log.info("Gauges: %d", len(gauges_records))

    existing = read_existing(bucket, parquet_key)
    start, end = cfg_start, cfg_end

    if existing is not None:
        existing["datetime_utc"] = pd.to_datetime(existing["datetime_utc"], utc=True)
        earliest = existing["datetime_utc"].min().date()
        end = min(end, earliest - timedelta(days=1))
        log.info(
            "Existing parquet: %d rows, earliest %s → downloading back to %s",
            len(existing), earliest, start,
        )

    if start > end:
        log.info("No historical gap to fill (start=%s > end=%s).", start, end)
        return

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    log.info("Days to process: %d  (%s → %s)", len(days), start, end)

    shard_dir  = ensure_dir("./hist_shards_nearest")
    n_workers  = cfg["execution"].get("max_workers_io", 8)

    args  = [(d.isoformat(), folder, gauges_records, shard_dir) for d in days]
    paths: list[str] = []

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(process_day, a): a[0] for a in args}
        for i, fut in enumerate(as_completed(futs), 1):
            day_iso = futs[fut]
            try:
                p = fut.result()
                if p:
                    paths.append(p)
            except Exception as e:
                log.error("Day %s failed: %s", day_iso, e)
            if i % 30 == 0:
                log.info("[%d/%d] days complete", i, len(args))

    if not paths:
        log.info("No new data downloaded.")
        return

    new_df = pa.concat_tables(
        [pq.read_table(p) for p in paths if Path(p).exists()]
    ).to_pandas()

    parts = [p for p in [new_df, existing] if p is not None and not p.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    combined = (
        combined
        .drop_duplicates(subset=["datetime_utc", "site_no"])
        .sort_values(["datetime_utc", "site_no"])
        .reset_index(drop=True)
    )

    write_parquet_to_s3(combined, bucket, parquet_key)
    log.info(
        "Wrote %s: %d rows  (%s → %s)",
        parquet_key, len(combined),
        combined["datetime_utc"].min().date(),
        combined["datetime_utc"].max().date(),
    )


if __name__ == "__main__":
    main()
