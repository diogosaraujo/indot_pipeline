"""extract_annual_precip_tiff.py

Build a mean-annual-precipitation GeoTIFF for Indiana from one sample year of
DAILY 24-hour QPE accumulations, and upload it to S3 (for ArcGIS).

Two sources, matching the pipeline's era split:

  Stage IV (2002-2014 era; sample year 2013)
      Iowa Environmental Mesonet Stage IV archive, daily 24-h accumulation
      valid 12Z:
        https://mesonet.agron.iastate.edu/archive/data/{Y}/{M}/{D}/stage4/
            ST4.{YYYYMMDD}12.24h[.grb2]
      4-km HRAP polar-stereographic grid (2-D lat/lon), mm. Read with cfgrib
      and resampled to a regular EPSG:4326 Indiana grid (nearest KDTree).

  MRMS (2015-present era; sample year 2022)
      Public noaa-mrms-pds bucket, MultiSensor 24-h QPE Pass2, one file/day @ 12Z:
        https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_24H_Pass2_00.00/
            {YYYYMMDD}/MRMS_..._{YYYYMMDD}-120000.grib2.gz
      1-km regular lat/lon grid, mm. Read with the pipeline's open_mrms_grib /
      canonicalize_mrms_grid. (This source only covers 2020-10-14 onward.)

Statistic: each daily file is a non-overlapping 24-h total. We accumulate a
per-pixel sum and a per-pixel day-count, then

    mean_annual = (sum / days_with_data) * days_in_year

which annualizes correctly even if some days are missing. Output units: inches
per year (default) or mm per year.

Output S3 key (default):
    s3://<bucket>/<prefix>samples/<source>_indiana_annual_<year>_<units>.tif

Usage:
    python extract_annual_precip_tiff.py --source stage4 --year 2013 --units in
    python extract_annual_precip_tiff.py --source mrms   --year 2022 --units in
    python extract_annual_precip_tiff.py --source mrms   --year 2022 --local ./mrms2022.tif

Requires: cfgrib, scipy, rasterio, numpy, requests, boto3.  Run from repo root.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    apply_units,
    canonicalize_mrms_grid,
    decompress_gz,
    load_config,
    open_mrms_grib,
    write_bytes_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("annual_precip")

ISU_STAGE4_BASE = "https://mesonet.agron.iastate.edu/archive/data"
MRMS_HTTP_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_24H_FOLDER = "MultiSensor_QPE_24H_Pass2_00.00"
MRMS_PDS_START = date(2020, 10, 14)

# Indiana bounding box (minlon, minlat, maxlon, maxlat), padded ~0.25 deg.
INDIANA_BOUNDS = (-88.35, 37.55, -84.55, 42.00)

MM_TO_IN = 1.0 / 25.4
NODATA_OUT = -9999.0
MAX_NEIGHBOR_DEG = 0.10   # Stage IV regrid: drop target cells with no source pixel this close


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 120) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        log.debug("  %s", e)
        return None
    return r.content if r.status_code == 200 and r.content else None


def fetch_stage4_24h(day: date) -> bytes | None:
    """Daily Stage IV 24-h accumulation valid 12Z on `day`."""
    base = f"{ISU_STAGE4_BASE}/{day:%Y/%m/%d}/stage4"
    stem = f"ST4.{day:%Y%m%d}12.24h"
    for ext in (".grib", ".grb2", ".grb"):
        raw = _get(f"{base}/{stem}{ext}")
        if raw is not None:
            return raw
    return None


def fetch_mrms_24h(day: date) -> bytes | None:
    """Daily MRMS MultiSensor 24-h QPE (Pass2) @ 12Z from noaa-mrms-pds."""
    d = f"{day:%Y%m%d}"
    stem = f"MRMS_{MRMS_24H_FOLDER}_{d}-120000.grib2.gz"
    raw = _get(f"{MRMS_HTTP_BASE}/CONUS/{MRMS_24H_FOLDER}/{d}/{stem}")
    if raw is not None:
        return raw
    # Fallback: list the day and pick the file nearest 12Z.
    body = _get(f"{MRMS_HTTP_BASE}/?list-type=2&prefix=CONUS/{MRMS_24H_FOLDER}/{d}/")
    if body is None:
        return None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [e.text for e in ET.fromstring(body).findall(".//s3:Contents/s3:Key", ns)]
    keys = [k for k in keys if k and k.endswith(".grib2.gz")]
    if not keys:
        return None
    best = min(keys, key=lambda k: abs(int(k.split("-")[-1].split(".")[0]) - 120000))
    return _get(f"{MRMS_HTTP_BASE}/{best}")


# ── GRIB reading ──────────────────────────────────────────────────────────────

def open_stage4_grib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cfgrib read of a Stage IV GRIB -> (data_mm, lats_2d, lons_2d)."""
    import cfgrib
    for ds in cfgrib.open_datasets(str(path), indexpath=""):
        for var in ds.data_vars:
            da = ds[var]
            lat_key = next((k for k in da.coords if "lat" in k.lower()), None)
            lon_key = next((k for k in da.coords if "lon" in k.lower()), None)
            if lat_key and lon_key:
                lons = da.coords[lon_key].values
                lons = np.where(lons > 180, lons - 360, lons)
                return da.values.astype(float), da.coords[lat_key].values, lons
    raise ValueError(f"No lat/lon variables found in Stage IV file: {path}")


def read_mrms_grid(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompress + read an MRMS GRIB -> canonical (arr_mm, lats_1d, lons_1d)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mrms.grib2"
        p.write_bytes(decompress_gz(raw) if raw[:2] == b"\x1f\x8b" else raw)
        ds = open_mrms_grib(p)
        dvars = list(ds.data_vars)
        if not dvars:
            ds.close()
            raise ValueError("No data variables in MRMS GRIB.")
        arr, lats, lons = canonicalize_mrms_grid(ds[dvars[0]])
        ds.close()
    return arr, lats, lons


# ── Target grid setup (one-time, per source) ──────────────────────────────────

def _target_grid(res_deg: float):
    west, south0, east0, north = INDIANA_BOUNDS
    width = int(round((east0 - west) / res_deg))
    height = int(round((north - south0) / res_deg))
    east = west + width * res_deg
    south = north - height * res_deg
    tlons = west + (np.arange(width) + 0.5) * res_deg
    tlats = north - (np.arange(height) + 0.5) * res_deg
    transform = from_bounds(west, south, east, north, width, height)
    return tlons, tlats, (height, width), transform


def setup_stage4(ref_raw: bytes, res_deg: float):
    """Build the nearest-neighbour index from the (static) Stage IV grid to the
    Indiana target grid. Returns a dict of everything the per-day worker needs."""
    from scipy.spatial import KDTree
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.grb"
        p.write_bytes(ref_raw)
        data_mm, lats2d, lons2d = open_stage4_grib(p)
    tlons, tlats, shape, transform = _target_grid(res_deg)
    glon, glat = np.meshgrid(tlons, tlats)
    tree = KDTree(np.column_stack([lats2d.ravel(), lons2d.ravel()]))
    dist, idx = tree.query(np.column_stack([glat.ravel(), glon.ravel()]), workers=-1)
    return {
        "idx": idx,
        "static_valid": dist <= MAX_NEIGHBOR_DEG,
        "source_size": data_mm.size,
        "shape": shape,
        "transform": transform,
    }


def setup_mrms(ref_raw: bytes):
    """Compute the Indiana row/col masks on the (static) MRMS grid + transform."""
    arr, lats, lons = read_mrms_grid(ref_raw)
    west, south, east, north = INDIANA_BOUNDS
    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)
    sub_lats, sub_lons = lats[lat_mask], lons[lon_mask]
    res_x = float(np.abs(np.diff(sub_lons)).mean())
    res_y = float(np.abs(np.diff(sub_lats)).mean())
    transform = from_bounds(
        sub_lons.min() - res_x / 2, sub_lats.min() - res_y / 2,
        sub_lons.max() + res_x / 2, sub_lats.max() + res_y / 2,
        sub_lons.size, sub_lats.size,
    )
    return {
        "lat_mask": lat_mask,
        "lon_mask": lon_mask,
        "shape": (int(lat_mask.sum()), int(lon_mask.sum())),
        "transform": transform,
    }


# ── Per-day workers: return an Indiana-grid array of daily inches (NaN=missing) ─

def stage4_day(day: date, ctx: dict, units: str) -> np.ndarray | None:
    raw = fetch_stage4_24h(day)
    if raw is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "d.grb"
        p.write_bytes(raw)
        data_mm, _, _ = open_stage4_grib(p)
    if data_mm.size != ctx["source_size"]:
        log.warning("%s: Stage IV grid size mismatch, skipping", day)
        return None
    vals = data_mm.ravel()[ctx["idx"]]
    vals = np.where(ctx["static_valid"] & np.isfinite(vals) & (vals >= 0), vals, np.nan)
    if units == "in":
        vals = vals * MM_TO_IN
    return vals.reshape(ctx["shape"])


def mrms_day(day: date, ctx: dict, units: str) -> np.ndarray | None:
    raw = fetch_mrms_24h(day)
    if raw is None:
        return None
    arr, _, _ = read_mrms_grid(raw)
    arr = apply_units(arr, kind="qpe", units=units)   # NaN negatives; mm->in if requested
    return arr[np.ix_(ctx["lat_mask"], ctx["lon_mask"])].astype("float64")


# ── GeoTIFF out ───────────────────────────────────────────────────────────────

def geotiff_bytes(array: np.ndarray, transform, units_label: str, product: str) -> bytes:
    filled = np.where(np.isfinite(array), array, NODATA_OUT).astype("float32")
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": array.shape[0], "width": array.shape[1],
        "crs": "EPSG:4326", "transform": transform,
        "nodata": NODATA_OUT, "compress": "lzw",
    }
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(filled, 1)
            dst.update_tags(1, UNITS=units_label, PRODUCT=product)
        return mem.read()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["stage4", "mrms"], required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--units", choices=["mm", "in"], default="in",
                   help="Output units, per YEAR (default: in -> inches/year)")
    p.add_argument("--res", type=float, default=0.03,
                   help="Stage IV target pixel size in degrees (default 0.03 ~ 3 km); "
                        "MRMS keeps its native ~1-km grid")
    p.add_argument("--stride", type=int, default=1,
                   help="Sample every Nth day (default 1 = every day)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--s3-key", default=None, help="Override S3 key")
    p.add_argument("--local", default=None, help="Also write a local copy to this path")
    args = p.parse_args()

    if args.source == "mrms" and date(args.year, 1, 1) < MRMS_PDS_START:
        raise SystemExit(
            f"MRMS on noaa-mrms-pds starts {MRMS_PDS_START}; year {args.year} is before that. "
            "Use --source stage4 for earlier years (or extend to the ISU GaugeCorr archive)."
        )

    days = [date(args.year, 1, 1) + timedelta(d)
            for d in range((date(args.year + 1, 1, 1) - date(args.year, 1, 1)).days)]
    days = days[::args.stride]
    days_in_year = (date(args.year + 1, 1, 1) - date(args.year, 1, 1)).days
    log.info("%s %d: %d candidate days (stride=%d)", args.source, args.year, len(days), args.stride)

    fetch = fetch_stage4_24h if args.source == "stage4" else fetch_mrms_24h

    # Establish the (static) grid from the first day that downloads.
    ctx = None
    for d in days:
        ref = fetch(d)
        if ref is not None:
            log.info("Reference grid from %s", d)
            ctx = setup_stage4(ref, args.res) if args.source == "stage4" else setup_mrms(ref)
            break
    if ctx is None:
        raise SystemExit("Could not download any daily file for the requested year.")

    day_fn = stage4_day if args.source == "stage4" else mrms_day
    cfg = load_config(args.config)
    n_workers = cfg.get("execution", {}).get("max_workers_io", 8)

    sum_arr = np.zeros(ctx["shape"], dtype="float64")
    cnt_arr = np.zeros(ctx["shape"], dtype="int32")
    n_ok = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(day_fn, d, ctx, args.units): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                arr = fut.result()
            except Exception as e:
                log.warning("%s failed: %s", d, e)
                continue
            if arr is None:
                continue
            m = np.isfinite(arr)
            sum_arr[m] += arr[m]
            cnt_arr[m] += 1
            n_ok += 1
            if i % 30 == 0:
                log.info("[%d/%d] days processed (%d with data)", i, len(days), n_ok)

    if n_ok == 0:
        raise SystemExit("No daily data accumulated.")

    # Annualize: per-pixel mean daily total * days_in_year.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_annual = np.where(cnt_arr > 0, sum_arr / cnt_arr * days_in_year, np.nan)

    units_label = "in/yr" if args.units == "in" else "mm/yr"
    valid = np.isfinite(mean_annual)
    log.info(
        "Accumulated %d days; mean-annual %s: min=%.1f mean=%.1f max=%.1f (median day-count/pixel=%d)",
        n_ok, units_label,
        float(np.nanmin(mean_annual)) if valid.any() else 0.0,
        float(np.nanmean(mean_annual)) if valid.any() else 0.0,
        float(np.nanmax(mean_annual)) if valid.any() else 0.0,
        int(np.median(cnt_arr[cnt_arr > 0])) if (cnt_arr > 0).any() else 0,
    )

    product = (f"{'NOAA Stage IV' if args.source == 'stage4' else 'NOAA MRMS MultiSensor'} "
               f"24h QPE - mean annual precip {args.year}")
    data = geotiff_bytes(mean_annual, ctx["transform"], units_label, product)

    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    key = args.s3_key or f"{prefix}samples/{args.source}_indiana_annual_{args.year}_{args.units}.tif"
    write_bytes_to_s3(data, bucket, key)
    log.info("Uploaded to s3://%s/%s (%d bytes)", bucket, key, len(data))

    if args.local:
        Path(args.local).write_bytes(data)
        log.info("Also wrote local copy: %s", args.local)

    log.info("Done. Open in ArcGIS (EPSG:4326 / WGS84).")


if __name__ == "__main__":
    main()
