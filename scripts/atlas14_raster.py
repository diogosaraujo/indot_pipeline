"""atlas14_raster.py

Export a NOAA Atlas 14 precipitation-frequency depth (default 50-yr, 24-h) as a
georeferenced GeoTIFF for use in ArcGIS.  Atlas 14 here is point data at the
gauge locations; the depths are interpolated (linear, with nearest-neighbour
fill so the whole state is covered) onto a regular WGS84 grid and, best-effort,
clipped to the Indiana boundary.  Bring it into ArcGIS and drape it over any
basemap.

Writes:
    s3://<bucket>/<prefix>analysis/atlas14/atlas14_p{rp}_{dur}h.tif

Usage:
    python scripts/atlas14_raster.py                 # 50-yr, 24-h
    python scripts/atlas14_raster.py --rp 100 --duration 24 --res 0.008
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request

import boto3
import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from shapely.geometry import shape
from shapely.ops import unary_union

from utils import load_config, write_bytes_to_s3

ATLAS_KEY = "atlas14/precipitation_frequency.parquet"
INV_KEY   = "stations/indiana_streamflow_sites.parquet"
NODATA    = -9999.0
STATE_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json",
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
]


def _pq(bucket, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def indiana_geom():
    """Fetch the Indiana polygon (best-effort) for clipping; None on failure."""
    for url in STATE_GEOJSON_URLS:
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
            geoms = []
            for f in data.get("features", []):
                props = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
                if str(props.get("name", "")).lower() == "indiana":
                    geoms.append(shape(f["geometry"]))
            if geoms:
                print(f"Indiana boundary from {url}")
                return unary_union(geoms)
        except Exception as e:                              # noqa: BLE001
            print(f"  boundary source failed ({e})")
    print("  no boundary — exporting the full interpolated rectangle (clip in ArcGIS)")
    return None


def _contains(geom, X, Y):
    try:
        from shapely import contains_xy
        return contains_xy(geom, X, Y)
    except Exception:                                       # shapely < 2.0
        from shapely.vectorized import contains
        return contains(geom, X, Y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rp", type=int, default=50, help="return period (yr)")
    ap.add_argument("--duration", type=int, default=24, help="duration (hours)")
    ap.add_argument("--res", type=float, default=0.01, help="grid resolution (deg, ~1.1 km)")
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    a = _pq(bucket, f"{prefix}{ATLAS_KEY}",
            ["site_no", "duration_hr", "return_period_yr", "depth_in"])
    a = a[(a["duration_hr"] == args.duration) & (a["return_period_yr"] == args.rp)]
    a["site_no"] = a["site_no"].astype(str)
    inv = _pq(bucket, f"{prefix}{INV_KEY}", ["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    df = a.merge(inv, on="site_no").dropna(subset=["dec_lat_va", "dec_long_va", "depth_in"])
    if df.empty:
        raise SystemExit(f"No Atlas14 rows for {args.rp}-yr / {args.duration}-h.")
    lon, lat, dep = (df["dec_long_va"].to_numpy(float), df["dec_lat_va"].to_numpy(float),
                     df["depth_in"].to_numpy(float))
    print(f"{len(df)} stations | depth {dep.min():.2f}–{dep.max():.2f} in")

    geom = indiana_geom()
    if geom is not None:
        minx, miny, maxx, maxy = geom.bounds
    else:
        pad = 0.2
        minx, miny, maxx, maxy = lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad

    res = args.res
    gx = np.arange(minx, maxx + res, res)
    gy = np.arange(miny, maxy + res, res)
    GX, GY = np.meshgrid(gx, gy)                            # GY ascends south -> north

    Zl = griddata((lon, lat), dep, (GX, GY), method="linear")
    Zn = griddata((lon, lat), dep, (GX, GY), method="nearest")
    Z = np.where(np.isnan(Zl), Zn, Zl)                     # fill hull gaps
    if geom is not None:
        Z = np.where(_contains(geom, GX, GY), Z, np.nan)   # clip to Indiana
    Z = np.where(np.isfinite(Z), Z, NODATA).astype("float32")

    Z_ras = np.flipud(Z)                                   # north-up for GeoTIFF
    transform = from_origin(minx, gy.max() + res, res, res)
    profile = dict(driver="GTiff", height=Z_ras.shape[0], width=Z_ras.shape[1],
                   count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                   nodata=NODATA, compress="deflate")

    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(Z_ras, 1)
            ds.update_tags(1, RETURN_PERIOD_YR=str(args.rp), DURATION_HR=str(args.duration),
                           UNITS="inches", SOURCE="NOAA Atlas 14 (interpolated from gauge points)")
        tif_bytes = mem.read()

    key = f"analysis/atlas14/atlas14_p{args.rp}_{args.duration}h.tif"
    write_bytes_to_s3(tif_bytes, bucket, f"{prefix}{key}")
    print(f"Wrote s3://{bucket}/{prefix}{key}  ({Z_ras.shape[1]}x{Z_ras.shape[0]} px, EPSG:4326)")
    print(f"Download:  aws s3 cp s3://{bucket}/{prefix}{key} .")


if __name__ == "__main__":
    main()
