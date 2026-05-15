"""Shared utilities for the INDOT bridge pipeline."""
from __future__ import annotations

import io
import gzip
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import boto3
import requests
import yaml

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)

# cfgrib emits noisy DEBUG/INFO per-file messages; suppress them.
# Matches the convention in HNTB's existing notebook.
logging.getLogger("cfgrib").setLevel(logging.ERROR)


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    """Load and lightly validate the pipeline config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg["aws"]["output_bucket"].endswith("CHANGEME"):
        raise ValueError("Edit config.yaml: aws.output_bucket is still the placeholder.")
    return cfg


def s3_client(region: str = "us-east-1"):
    return boto3.client("s3", region_name=region)


def write_parquet_to_s3(df, bucket: str, key: str, region: str = "us-east-1") -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    buf.seek(0)
    s3_client(region).put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def write_bytes_to_s3(data: bytes, bucket: str, key: str, region: str = "us-east-1") -> None:
    s3_client(region).put_object(Bucket=bucket, Key=key, Body=data)


def s3_object_exists(bucket: str, key: str, region: str = "us-east-1") -> bool:
    """Check whether an S3 object exists.

    Uses list_objects_v2 (authorizes against s3:ListBucket) rather than
    head_object. s3:HeadObject is not a real IAM action — HeadObject calls
    authorize against s3:GetObject, which is misleading in policy documents.
    list_objects_v2 with an exact Prefix and MaxKeys=1 is equally fast and
    maps cleanly to the s3:ListBucket action already in the IAM policy.
    """
    resp = s3_client(region).list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return bool(resp.get("Contents"))


def daterange(start: date, end: date) -> Iterator[date]:
    """Yield each date inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0


def with_retries(fn, policy: RetryPolicy = RetryPolicy(), exceptions=(Exception,)):
    """Call `fn()` with exponential backoff. Returns the result or raises after max_attempts."""
    last: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except exceptions as e:
            last = e
            if attempt == policy.max_attempts:
                logging.warning("Attempt %d/%d failed: %s. No retries left",
                                attempt, policy.max_attempts, e)
                break
            delay = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
            logging.warning("Attempt %d/%d failed: %s. Retrying in %.1fs",
                            attempt, policy.max_attempts, e, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def decompress_gz(data: bytes) -> bytes:
    """Decompress a gzipped GRIB file."""
    return gzip.decompress(data)


def parse_iso_or_none(s: Optional[str]) -> Optional[date]:
    if s is None:
        return None
    return datetime.fromisoformat(s).date()


def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonicalize_mrms_grid(da):
    """Return (data_2d, lats_desc, lons_asc_-180_180).

    MRMS files can land with lats in either order and lons in 0..360 or
    -180..180. We standardize to lats descending and lons in -180..180 so
    that watershed masks built from a single reference grid are valid for
    every timestep.
    """
    import numpy as np
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lats = da[lat_name].values.copy()
    lons = da[lon_name].values.copy()
    arr = da.values

    if lons.max() > 180.0:
        lons = np.where(lons > 180, lons - 360.0, lons)
        # lons that were monotonic ascending in 0..360 remain monotonic
        # ascending in -180..180 as long as the dataset doesn't cross the
        # antimeridian (MRMS CONUS does not).
        order_lon = np.argsort(lons)
        if not np.array_equal(order_lon, np.arange(lons.size)):
            lons = lons[order_lon]
            arr = arr[..., order_lon]

    if lats[0] < lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]

    return arr, lats, lons


# ---------------------------------------------------------------------------
# MRMS helpers
# ---------------------------------------------------------------------------

# MRMS GRIB files do not carry a friendly variable name, so cfgrib stores
# the data array as `unknown`. We rely on `list(ds.data_vars)[0]` rather
# than this constant, but it's recorded here for clarity.
MRMS_VARIABLE_NAME = "unknown"


def open_mrms_grib(path) -> "xarray.Dataset":
    """Open an MRMS GRIB2 file with the right options for this pipeline.

    - `engine='cfgrib'`: GRIB backend.
    - `decode_timedelta=False`: silences the xarray FutureWarning that
      otherwise floods logs during long extraction runs.
    - `backend_kwargs={'indexpath': ''}`: tells cfgrib to keep its index
      in memory rather than writing a `.idx` sidecar next to the file
      (important when reading from `/tmp` on parallel workers).
    """
    import xarray as xr
    return xr.open_dataset(
        path,
        engine="cfgrib",
        decode_timedelta=False,
        backend_kwargs={"indexpath": ""},
    )


def apply_units(arr, kind: str, units: str):
    """Apply unit / sentinel handling for a 2-D MRMS data array.

    QPE products are native millimeters; convert to inches if requested.
    ARI is years, QPEFFG is %, RQI is dimensionless — no conversion needed
    but we still flag negative sentinel values as missing.
    """
    import numpy as np
    arr = np.where(arr < 0, np.nan, arr)
    if kind == "qpe" and units == "in":
        arr = arr / 25.4
    return arr


def resolve_hours(spec) -> list[str]:
    """Convert a `hours` config value into a list of two-digit hour strings.

    Accepts "all" (or None/missing) for every hour, or an explicit list
    such as ["23"] for once-per-day products like RQI_24H.
    """
    if spec is None or spec == "all":
        return [f"{i:02d}" for i in range(24)]
    if isinstance(spec, list):
        return [f"{int(h):02d}" for h in spec]
    raise ValueError(f"Unrecognized hours spec: {spec!r}")


def list_mrms_keys_for_day(fs, bucket: str, folder: str, day) -> list[str]:
    """List all GRIB files in s3://{bucket}/CONUS/{folder}/{YYYYMMDD}/."""
    prefix = f"{bucket}/CONUS/{folder}/{day:%Y%m%d}/"
    try:
        return sorted(fs.ls(prefix))
    except FileNotFoundError:
        return []


def build_watershed_union(bucket: str, prefix: str):
    """Union all per-gauge watershed GeoJSONs from S3 into one Shapely polygon.

    Files written by script 03 are GeoJSON Features saved under
    {prefix}watersheds/per_gauge/{site_no}.geojson.  Returns None if no
    files are found so callers can fall back to a bounding-box approach.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    ws_prefix = f"{prefix}watersheds/per_gauge/"
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=ws_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".geojson"):
                keys.append(obj["Key"])
    if not keys:
        logging.getLogger(__name__).warning("No watershed GeoJSONs found under %s", ws_prefix)
        return None

    polys = []
    for key in keys:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            feat = json.loads(body)
            candidates = (
                feat.get("features", [])
                if feat.get("type") == "FeatureCollection"
                else [feat]
            )
            for c in candidates:
                geom = c.get("geometry") if isinstance(c, dict) else None
                if geom:
                    polys.append(make_valid(shape(geom)))
        except Exception:
            pass
    return unary_union(polys) if polys else None


def filter_by_polygon(
    df,
    polygon,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    fallback_bbox: Optional[tuple] = None,
):
    """Return rows of df whose (lat_col, lon_col) fall inside polygon.

    If polygon is None, falls back to a (minlat, minlon, maxlat, maxlon) tuple.
    Applies a fast bounding-box pre-filter before the exact point-in-polygon test.
    """
    if polygon is None:
        if fallback_bbox is None:
            return df.copy()
        minlat, minlon, maxlat, maxlon = fallback_bbox
        mask = df[lat_col].between(minlat, maxlat) & df[lon_col].between(minlon, maxlon)
        return df[mask].reset_index(drop=True)

    from shapely.geometry import Point

    minx, miny, maxx, maxy = polygon.bounds
    bbox_mask = df[lat_col].between(miny, maxy) & df[lon_col].between(minx, maxx)
    cands = df[bbox_mask].copy()
    inside = cands.apply(
        lambda r: polygon.contains(Point(r[lon_col], r[lat_col])), axis=1
    )
    return cands[inside].reset_index(drop=True)


def parse_mrms_timestamp(key: str):
    """Parse the timestamp from an MRMS filename.

    Files look like:
      MRMS_MultiSensor_QPE_01H_Pass2_00.00_20240715-130000.grib2.gz
    """
    from datetime import datetime, timezone
    name = Path(key).name
    stem = name.split(".grib2")[0]
    ts = stem.split("_")[-1]
    return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
