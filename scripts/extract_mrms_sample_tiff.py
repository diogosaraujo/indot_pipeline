"""extract_mrms_sample_tiff.py

Download a single MRMS MultiSensor QPE record, clip it to Indiana, and write a
GeoTIFF that opens directly in ArcGIS. Companion to extract_stage4_sample_tiff.py
so you can compare the two precip sources side by side (MRMS covers the
2020-10-14 -> present era; Stage IV covers 2002 -> present).

Source is the public NOAA MRMS archive on S3 (same bucket script 05 uses),
fetched anonymously over HTTPS so no AWS credentials are needed:

    https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{FOLDER}/{YYYYMMDD}/
        MRMS_{FOLDER}_{YYYYMMDD}-{HH}0000.grib2.gz

MRMS is a regular 0.01-deg lat/lon grid (native units mm), but the GRIB stores
longitudes in 0..360. Rather than fight GDAL's longitude wrap, we reuse the
pipeline's tested readers (open_mrms_grib -> canonicalize_mrms_grid ->
apply_units), which return the grid with lons in -180..180 and lats descending,
then subset to Indiana and write the GeoTIFF from that regular sub-grid.

Usage:
    python extract_mrms_sample_tiff.py                       # default sample hour
    python extract_mrms_sample_tiff.py --datetime "2021-06-26 12"
    python extract_mrms_sample_tiff.py --units in --out indiana_mrms.tif

Requires: rasterio, cfgrib, xarray, numpy, requests (all in requirements.txt).
Run from the scripts/ directory (imports utils), or from anywhere — the script
adds its own directory to sys.path.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    apply_units,
    canonicalize_mrms_grid,
    decompress_gz,
    open_mrms_grib,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("mrms_sample")

MRMS_HTTP_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"
# Default product: hourly multi-sensor QPE, pass 2 (matches config.yaml).
DEFAULT_FOLDER = "MultiSensor_QPE_01H_Pass2_00.00"

# Indiana bounding box (minlon, minlat, maxlon, maxlat), padded ~0.25 deg.
INDIANA_BOUNDS = (-88.35, 37.55, -84.55, 42.00)
NODATA_OUT = -9999.0


def _get(url: str, timeout: int = 120) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        log.debug("  %s", e)
        return None
    return r.content if r.status_code == 200 and r.content else None


def fetch_mrms(dt: datetime, folder: str) -> bytes:
    """Download the gzipped MRMS GRIB2 for `dt` (UTC).

    Tries the exact top-of-hour filename first, then falls back to listing the
    day's prefix and choosing the file whose timestamp is nearest the request."""
    day = dt.strftime("%Y%m%d")
    stem = f"MRMS_{folder}_{day}-{dt.strftime('%H')}0000.grib2.gz"
    direct = f"{MRMS_HTTP_BASE}/CONUS/{folder}/{day}/{stem}"
    log.info("Trying %s", direct)
    raw = _get(direct)
    if raw is not None:
        log.info("  got %d bytes", len(raw))
        return raw

    # Fallback: list the day and pick the closest timestamp to HH0000.
    log.info("  not found at top of hour; listing day prefix...")
    list_url = f"{MRMS_HTTP_BASE}/?list-type=2&prefix=CONUS/{folder}/{day}/"
    body = _get(list_url)
    if body is None:
        raise RuntimeError(f"Could not list MRMS prefix for {day}.")
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [e.text for e in ET.fromstring(body).findall(".//s3:Contents/s3:Key", ns)]
    keys = [k for k in keys if k and k.endswith(".grib2.gz")]
    if not keys:
        raise RuntimeError(
            f"No MRMS files for {day}. The noaa-mrms-pds archive starts 2020-10-14."
        )
    target = dt.strftime("%Y%m%d-%H0000")
    best = min(keys, key=lambda k: abs(int(k.split("-")[-1].split(".")[0])
                                       - int(target.split("-")[1])))
    log.info("  nearest file: %s", best.rsplit("/", 1)[-1])
    raw = _get(f"{MRMS_HTTP_BASE}/{best}")
    if raw is None:
        raise RuntimeError(f"Failed to download {best}")
    log.info("  got %d bytes", len(raw))
    return raw


def clip_to_indiana(arr: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Subset the canonicalized MRMS grid (lats descending, lons ascending) to
    the Indiana bbox and build the matching EPSG:4326 affine transform."""
    west, south, east, north = INDIANA_BOUNDS
    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)
    if not lat_mask.any() or not lon_mask.any():
        raise RuntimeError("Indiana bbox does not overlap the MRMS grid.")

    sub_lats = lats[lat_mask]          # descending -> row 0 is north
    sub_lons = lons[lon_mask]          # ascending
    sub = arr[np.ix_(lat_mask, lon_mask)]

    res_x = float(np.abs(np.diff(sub_lons)).mean())
    res_y = float(np.abs(np.diff(sub_lats)).mean())
    transform = from_bounds(
        sub_lons.min() - res_x / 2, sub_lats.min() - res_y / 2,
        sub_lons.max() + res_x / 2, sub_lats.max() + res_y / 2,
        sub_lons.size, sub_lats.size,
    )
    return sub.astype("float32"), transform


def write_geotiff(array: np.ndarray, transform, out_path: Path, units: str) -> None:
    filled = np.where(np.isfinite(array), array, NODATA_OUT).astype("float32")
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": array.shape[0], "width": array.shape[1],
        "crs": "EPSG:4326", "transform": transform,
        "nodata": NODATA_OUT, "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filled, 1)
        dst.update_tags(1, UNITS=units, PRODUCT="NOAA MRMS MultiSensor QPE 01H Pass2")
    valid = np.isfinite(array)
    log.info(
        "Wrote %s  (%dx%d, EPSG:4326; valid pixels=%d, max=%.3f %s)",
        out_path, array.shape[1], array.shape[0], int(valid.sum()),
        float(np.nanmax(array)) if valid.any() else 0.0, units,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datetime", default="2021-06-26 12",
                   help='Sample hour in UTC, "YYYY-MM-DD HH" (default: 2021-06-26 12)')
    p.add_argument("--units", choices=["mm", "in"], default="mm",
                   help="Output units (default: mm, the MRMS native unit)")
    p.add_argument("--folder", default=DEFAULT_FOLDER,
                   help=f"MRMS product folder (default: {DEFAULT_FOLDER})")
    p.add_argument("--out", default=None,
                   help="Output GeoTIFF path (default: mrms_indiana_<stamp>.tif)")
    args = p.parse_args()

    dt = datetime.strptime(args.datetime, "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    out_path = Path(args.out) if args.out else Path(
        f"mrms_indiana_{dt:%Y%m%d%H}_{args.units}.tif")

    raw = fetch_mrms(dt, args.folder)
    with tempfile.TemporaryDirectory() as tmp:
        grib_path = Path(tmp) / "mrms.grib2"
        grib_path.write_bytes(decompress_gz(raw) if raw[:2] == b"\x1f\x8b" else raw)
        ds = open_mrms_grib(grib_path)
        dvars = list(ds.data_vars)
        if not dvars:
            raise RuntimeError("No data variables found in the MRMS GRIB.")
        arr, lats, lons = canonicalize_mrms_grid(ds[dvars[0]])
        ds.close()

    arr = apply_units(arr, kind="qpe", units=args.units)
    sub, transform = clip_to_indiana(arr, lats, lons)
    write_geotiff(sub, transform, out_path, args.units)
    log.info("Done. Open %s in ArcGIS (EPSG:4326 / WGS84).", out_path)


if __name__ == "__main__":
    main()
