"""extract_stage4_sample_tiff.py

Download a single NOAA Stage IV hourly QPE record, clip it to Indiana, and
write a GeoTIFF that opens directly in ArcGIS.

This is a standalone inspection utility (not part of the S3 pipeline). It pulls
one hour from the Iowa State University / Iowa Environmental Mesonet Stage IV
archive — the same source script 05b uses:

    https://mesonet.agron.iastate.edu/archive/data/{YYYY}/{MM}/{DD}/stage4/
        ST4.{YYYYMMDDHH}.01h.grb[2]

Stage IV lives on a 4-km HRAP polar-stereographic grid in millimetres. GDAL's
GRIB driver (via rasterio) reads that grid definition and georeferences it
automatically, so we let it do the projection math, then warp/clip to a small
Indiana window in EPSG:4326 (lon/lat) — the easiest thing to overlay in ArcGIS.

Usage:
    python extract_stage4_sample_tiff.py                       # default sample hour
    python extract_stage4_sample_tiff.py --datetime "2021-06-26 12"
    python extract_stage4_sample_tiff.py --units in --out indiana_st4.tif

Requires: rasterio, requests, numpy  (all in requirements.txt / the conda env).
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("stage4_sample")

ISU_STAGE4_BASE = "https://mesonet.agron.iastate.edu/archive/data"

# Indiana bounding box (minlon, minlat, maxlon, maxlat), padded ~0.25 deg so the
# state edge is not clipped tight. Mirrors INDIANA_BBOX used elsewhere.
INDIANA_BOUNDS = (-88.35, 37.55, -84.55, 42.00)

MM_TO_IN = 1.0 / 25.4
NODATA_OUT = -9999.0


def fetch_stage4(dt: datetime) -> bytes:
    """Download the Stage IV 1-hour QPE GRIB for `dt` (UTC). Tries the known
    extensions (newer files are GRIB2 .grb2, older are GRIB1 .grb/.grib)."""
    base = f"{ISU_STAGE4_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}/stage4"
    stem = f"ST4.{dt.strftime('%Y%m%d%H')}.01h"
    for ext in (".grib", ".grb2", ".grb"):
        url = f"{base}/{stem}{ext}"
        log.info("Trying %s", url)
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as e:
            log.debug("  %s", e)
            continue
        if r.status_code == 200 and r.content:
            log.info("  got %d bytes", len(r.content))
            return r.content
    raise RuntimeError(
        f"No Stage IV file found for {dt:%Y-%m-%d %HZ}. "
        "Pick another hour (the archive starts 2002-01-01)."
    )


def clip_to_indiana(grib_path: Path, units: str, res_deg: float) -> tuple:
    """Reproject the Stage IV grid to EPSG:4326 clipped to Indiana.

    Returns (array, transform, profile-crs). Nearest-neighbour resampling keeps
    the native pixel values (no smoothing of the precip field)."""
    west, south, east, north = INDIANA_BOUNDS
    width = int(round((east - west) / res_deg))
    height = int(round((north - south) / res_deg))
    dst_transform = from_bounds(west, south, east, north, width, height)
    dst = np.full((height, width), np.nan, dtype="float32")

    with rasterio.open(grib_path) as src:
        if src.crs is None:
            raise RuntimeError(
                "GDAL could not georeference this GRIB (no CRS). "
                "Check that the GDAL GRIB driver is available."
            )
        log.info("Source grid: %s, %dx%d, CRS=%s",
                 src.driver, src.width, src.height, src.crs.to_string())
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    # Mask Stage IV missing/negative sentinels, then optionally convert mm -> in.
    dst = np.where(np.isfinite(dst) & (dst >= 0), dst, np.nan)
    if units == "in":
        dst = dst * MM_TO_IN
    return dst, dst_transform


def write_geotiff(array: np.ndarray, transform, out_path: Path, units: str) -> None:
    filled = np.where(np.isfinite(array), array, NODATA_OUT).astype("float32")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": NODATA_OUT,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filled, 1)
        dst.update_tags(1, UNITS=units, PRODUCT="NOAA Stage IV 1h QPE")
    valid = np.isfinite(array)
    log.info(
        "Wrote %s  (%dx%d, %s; valid pixels=%d, max=%.3f %s)",
        out_path, array.shape[1], array.shape[0], "EPSG:4326",
        int(valid.sum()), float(np.nanmax(array)) if valid.any() else 0.0, units,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datetime", default="2021-06-26 12",
                   help='Sample hour in UTC, "YYYY-MM-DD HH" (default: 2021-06-26 12)')
    p.add_argument("--units", choices=["mm", "in"], default="mm",
                   help="Output units (default: mm, the Stage IV native unit)")
    p.add_argument("--res", type=float, default=0.03,
                   help="Output pixel size in degrees (default 0.03 ~ 3 km)")
    p.add_argument("--out", default=None,
                   help="Output GeoTIFF path (default: stage4_indiana_<stamp>.tif)")
    args = p.parse_args()

    dt = datetime.strptime(args.datetime, "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    out_path = Path(args.out) if args.out else Path(
        f"stage4_indiana_{dt:%Y%m%d%H}_{args.units}.tif")

    raw = fetch_stage4(dt)
    with tempfile.TemporaryDirectory() as tmp:
        grib_path = Path(tmp) / "st4.grb"
        grib_path.write_bytes(raw)
        array, transform = clip_to_indiana(grib_path, args.units, args.res)

    write_geotiff(array, transform, out_path, args.units)
    log.info("Done. Open %s in ArcGIS (EPSG:4326 / WGS84).", out_path)


if __name__ == "__main__":
    main()
