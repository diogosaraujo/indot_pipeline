"""extract_stage4_sample_tiff.py

Download a single NOAA Stage IV hourly QPE record, clip it to Indiana, and
write a GeoTIFF (to S3) that opens directly in ArcGIS.

This is a standalone inspection utility. It pulls one hour from the Iowa State
University / Iowa Environmental Mesonet Stage IV archive — the same source
script 05b uses:

    https://mesonet.agron.iastate.edu/archive/data/{YYYY}/{MM}/{DD}/stage4/
        ST4.{YYYYMMDDHH}.01h.grb[2]

Stage IV lives on a 4-km HRAP polar-stereographic grid in millimetres; the GRIB
message carries 2-D (curvilinear) latitude/longitude arrays. We read it with
cfgrib (same as pipeline script 05b — avoids depending on GDAL's GRIB plugin),
then resample the curvilinear field onto a regular EPSG:4326 (lon/lat) grid over
Indiana with a nearest-neighbour KDTree. The GeoTIFF is built in memory and
uploaded to S3 (bucket/prefix from config.yaml); only GeoTIFF *writing* uses
GDAL/rasterio (the always-present GTiff driver), not GRIB reading.

Output S3 key (default):
    s3://<output_bucket>/<output_prefix>samples/stage4_indiana_<stamp>_<units>.tif

Usage:
    python extract_stage4_sample_tiff.py                        # -> S3, default hour
    python extract_stage4_sample_tiff.py --datetime "2021-06-26 12" --units in
    python extract_stage4_sample_tiff.py --s3-key custom/key.tif
    python extract_stage4_sample_tiff.py --local ./st4.tif      # also drop a local copy

Requires: cfgrib, scipy, rasterio, numpy, requests, boto3  (all in the conda env).
Run from the repo root (loads config.yaml), or pass --config.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, write_bytes_to_s3  # noqa: E402

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
# Null out target cells with no Stage IV source pixel within this many degrees
# (~0.1 deg ~ 8-11 km, a bit over the 4-km grid spacing). Indiana is fully
# inside the CONUS domain, so this only guards the very edges.
MAX_NEIGHBOR_DEG = 0.10


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


def open_stage4_grib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a Stage IV GRIB with cfgrib; return (data_mm, lats_2d, lons_2d).

    cfgrib handles both GRIB1 and GRIB2. The HRAP grid lat/lon are 2-D arrays
    stored in the message; lons are normalized to -180..180. (Same logic as
    pipeline script 05b.)"""
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


def clip_to_indiana(grib_path: Path, units: str, res_deg: float) -> tuple:
    """Resample the curvilinear Stage IV grid onto a regular EPSG:4326 grid over
    Indiana via nearest-neighbour KDTree. Returns (array, transform). Nearest
    keeps the native pixel values (no smoothing of the precip field)."""
    from scipy.spatial import KDTree

    data_mm, lats_2d, lons_2d = open_stage4_grib(grib_path)
    log.info("Source Stage IV grid: %s points", data_mm.size)

    # Regular Indiana target grid (row 0 = north so the GeoTIFF is north-up).
    west, south0, east0, north = INDIANA_BOUNDS
    width = int(round((east0 - west) / res_deg))
    height = int(round((north - south0) / res_deg))
    east = west + width * res_deg
    south = north - height * res_deg
    tlons = west + (np.arange(width) + 0.5) * res_deg
    tlats = north - (np.arange(height) + 0.5) * res_deg
    glon, glat = np.meshgrid(tlons, tlats)

    tree = KDTree(np.column_stack([lats_2d.ravel(), lons_2d.ravel()]))
    dist, idx = tree.query(np.column_stack([glat.ravel(), glon.ravel()]), workers=-1)
    vals = data_mm.ravel()[idx].reshape(height, width)
    vals = np.where(dist.reshape(height, width) > MAX_NEIGHBOR_DEG, np.nan, vals)

    # Mask Stage IV missing/negative sentinels, then optionally convert mm -> in.
    vals = np.where(np.isfinite(vals) & (vals >= 0), vals, np.nan)
    if units == "in":
        vals = vals * MM_TO_IN

    transform = from_bounds(west, south, east, north, width, height)
    return vals.astype("float32"), transform


def geotiff_bytes(array: np.ndarray, transform, units: str) -> bytes:
    """Build a single-band GeoTIFF entirely in memory and return its bytes."""
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
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(filled, 1)
            dst.update_tags(1, UNITS=units, PRODUCT="NOAA Stage IV 1h QPE")
        data = mem.read()
    valid = np.isfinite(array)
    log.info(
        "Built GeoTIFF (%dx%d, EPSG:4326; valid pixels=%d, max=%.3f %s, %d bytes)",
        array.shape[1], array.shape[0], int(valid.sum()),
        float(np.nanmax(array)) if valid.any() else 0.0, units, len(data),
    )
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datetime", default="2021-06-26 12",
                   help='Sample hour in UTC, "YYYY-MM-DD HH" (default: 2021-06-26 12)')
    p.add_argument("--units", choices=["mm", "in"], default="mm",
                   help="Output units (default: mm, the Stage IV native unit)")
    p.add_argument("--res", type=float, default=0.03,
                   help="Output pixel size in degrees (default 0.03 ~ 3 km)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--s3-key", default=None,
                   help="Override S3 key (default: <prefix>samples/stage4_indiana_<stamp>_<units>.tif)")
    p.add_argument("--local", default=None,
                   help="Also write a local copy to this path")
    args = p.parse_args()

    dt = datetime.strptime(args.datetime, "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    fname = f"stage4_indiana_{dt:%Y%m%d%H}_{args.units}.tif"

    raw = fetch_stage4(dt)
    with tempfile.TemporaryDirectory() as tmp:
        grib_path = Path(tmp) / "st4.grb"
        grib_path.write_bytes(raw)
        array, transform = clip_to_indiana(grib_path, args.units, args.res)

    data = geotiff_bytes(array, transform, args.units)

    cfg = load_config(args.config)
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    key = args.s3_key or f"{prefix}samples/{fname}"
    write_bytes_to_s3(data, bucket, key)
    log.info("Uploaded to s3://%s/%s", bucket, key)

    if args.local:
        Path(args.local).write_bytes(data)
        log.info("Also wrote local copy: %s", args.local)

    log.info("Done. Open in ArcGIS (EPSG:4326 / WGS84).")


if __name__ == "__main__":
    main()
