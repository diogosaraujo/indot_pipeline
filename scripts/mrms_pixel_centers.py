"""mrms_pixel_centers.py

GeoJSON of the MRMS pixel CENTERS matched to the streamflow gauges used in 08c
(the "nearest MRMS pixel" source).  The nearest_pixel parquet stores only values,
so the centers are recomputed with the SAME nearest-neighbour snap 05 uses:
open one MRMS GRIB for the grid, then argmin |grid - gauge| per gauge.

Writes:
    s3://<bucket>/<prefix>analysis/mrms/mrms_pixel_centers.geojson

Usage:
    python scripts/mrms_pixel_centers.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pyarrow.parquet as pq
import s3fs

from utils import (
    canonicalize_mrms_grid,
    decompress_gz,
    list_mrms_keys_for_day,
    load_config,
    open_mrms_grib,
    s3_client,
)

TC_KEY  = "analysis/event_confusion_matrix_tc.parquet"
INV_KEY = "stations/indiana_streamflow_sites.parquet"
OUT_KEY = "analysis/mrms/mrms_pixel_centers.geojson"
EARTH_KM = 6371.0


def _pq(bucket, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def load_mrms_grid(cfg):
    """Open one MRMS GRIB and return (lats_grid, lons_grid) 1-D arrays."""
    fs = s3fs.S3FileSystem(anon=True)
    bucket = cfg["mrms"]["bucket"]
    folder = cfg["mrms"]["products"][0]["folder"]
    start = dt.date.fromisoformat(str(cfg["mrms"]["start_date"]))
    for i in range(0, 14):                       # walk forward until a day has files
        day = start + dt.timedelta(days=i)
        keys = list_mrms_keys_for_day(fs, bucket, folder, day)
        if keys:
            break
    else:
        raise SystemExit("No MRMS files found near start_date.")
    key = keys[0]
    raw = fs.cat(key)
    if key.endswith(".gz"):
        raw = decompress_gz(raw)
    with tempfile.TemporaryDirectory() as sc:
        p = Path(sc) / Path(key).name.replace(".gz", "")
        p.write_bytes(raw)
        ds = open_mrms_grib(p)
        _, lats_grid, lons_grid = canonicalize_mrms_grid(ds[list(ds.data_vars)[0]])
        ds.close()
    print(f"MRMS grid from {key}  ({lats_grid.size} lat x {lons_grid.size} lon)")
    return np.asarray(lats_grid, float), np.asarray(lons_grid, float)


def _haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    sites = set(_pq(bucket, f"{prefix}{TC_KEY}", ["site_no"])["site_no"].astype(str))
    inv = _pq(bucket, f"{prefix}{INV_KEY}", ["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    g = inv[inv["site_no"].isin(sites)].dropna(subset=["dec_lat_va", "dec_long_va"])
    print(f"{len(g)} of {len(sites)} 08c gauges have coordinates")

    lats_grid, lons_grid = load_mrms_grid(cfg)
    glat = g["dec_lat_va"].to_numpy(float)
    glon = g["dec_long_va"].to_numpy(float)
    lat_idx = np.abs(lats_grid[:, None] - glat[None, :]).argmin(axis=0)
    lon_idx = np.abs(lons_grid[:, None] - glon[None, :]).argmin(axis=0)
    plat, plon = lats_grid[lat_idx], lons_grid[lon_idx]
    dist = _haversine_km(glat, glon, plat, plon)

    features = []
    for sid, gla, glo, pla, plo, dk in zip(g["site_no"], glat, glon, plat, plon, dist):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(plo), 5), round(float(pla), 5)]},
            "properties": {
                "site_no": sid,
                "pixel_lat": round(float(pla), 5), "pixel_lon": round(float(plo), 5),
                "gauge_lat": round(float(gla), 5), "gauge_lon": round(float(glo), 5),
                "offset_km": round(float(dk), 3),
            },
        })
    body = json.dumps({"type": "FeatureCollection", "features": features}, indent=1).encode()
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{OUT_KEY}", Body=body,
                           ContentType="application/geo+json")
    print(f"Wrote s3://{bucket}/{prefix}{OUT_KEY}  ({len(features)} pixels; "
          f"median offset {np.median(dist):.2f} km, max {dist.max():.2f} km)")
    print(f"Download:  aws s3 cp s3://{bucket}/{prefix}{OUT_KEY} .")


if __name__ == "__main__":
    main()
