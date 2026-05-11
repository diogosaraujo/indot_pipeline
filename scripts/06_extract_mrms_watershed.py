"""06_extract_mrms_watershed.py

For every MRMS product in `cfg.mrms.products` and every hourly timestamp
in [start_date, end_date]:
  - For each gauge's watershed polygon, extract every pixel inside it.
  - Compute the area-weighted watershed-mean using ALL pixels (no
    lower-limit filtering) so the mean is statistically faithful.
  - Optionally write a per-watershed Zarr store of pixel-level values,
    storing only pixels >= product.lower_limit (sparse). This is the
    same convention HNTB's notebook uses to keep storage tractable.

Why the sparse approach for the per-pixel store and dense for the mean:
  - "All pixels in watershed x all timestamps x all gauges x all products"
    gets very large very fast. For QPE_01H_Pass2 alone, dense storage of
    every pixel for every hour over 5+ years is ~100 GB per watershed.
    Storing only pixels above 0.1" reduces that by 1-2 orders of magnitude
    because most hours are dry over most of any watershed.
  - The watershed-mean is the daily-driver for trigger comparison and
    must be computed from the full pixel set. A sparse mean would bias
    high in dry periods.

Strategy:
  - Pre-compute one row/col index mask per watershed ONCE against the
    MRMS grid. The MRMS grid is identical hour-to-hour and product-to-
    product (all are 0.01-degree CONUS), so masks are reusable.
  - For each (product, day), loop over gauges using pre-computed indices.

Writes (one set per product key in config):
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/watershed_mean.parquet
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/per_watershed_zarr/{site_no}.zarr/
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import shape

from utils import (
    apply_units,
    canonicalize_mrms_grid,
    daterange,
    decompress_gz,
    ensure_dir,
    list_mrms_keys_for_day,
    load_config,
    open_mrms_grib,
    parse_iso_or_none,
    parse_mrms_timestamp,
    resolve_hours,
    s3_client,
)

log = logging.getLogger("06_mrms_watershed")


# ---------- Mask precomputation ----------

@dataclass
class WatershedMask:
    site_no: str
    row_idx: np.ndarray
    col_idx: np.ndarray
    weights: np.ndarray   # cos(lat)-area weights, normalized


def build_grid_reference(template_da: xr.DataArray) -> dict:
    """Capture the MRMS grid geometry from one timestep so we can build masks
    once and reuse them across all timesteps and products. The grid is
    canonicalized so lats are descending and lons are in -180..180 ascending."""
    _, lats, lons = canonicalize_mrms_grid(template_da)
    dlat = float(np.abs(lats[1] - lats[0]))
    dlon = float(np.abs(lons[1] - lons[0]))
    top = lats[0] + dlat / 2          # lats are descending => row 0 is max lat
    left = lons.min() - dlon / 2
    transform = from_origin(left, top, dlon, dlat)
    return {
        "lats": lats, "lons": lons, "dlat": dlat, "dlon": dlon,
        "transform": transform, "shape": (lats.size, lons.size),
    }


def build_mask_for_polygon(geom_geojson: dict, grid: dict) -> WatershedMask:
    """Rasterize the polygon to the MRMS grid -> 1-D row/col indices + weights."""
    geom = shape(geom_geojson["geometry"] if "geometry" in geom_geojson else geom_geojson)
    mask2d = geometry_mask(
        [geom],
        out_shape=grid["shape"],
        transform=grid["transform"],
        invert=True,           # True == inside polygon
        all_touched=True,
    )
    rows, cols = np.nonzero(mask2d)
    pixel_lats = grid["lats"][rows]
    w = np.cos(np.deg2rad(pixel_lats))
    w = w / w.sum() if w.sum() > 0 else w
    return WatershedMask(site_no="", row_idx=rows, col_idx=cols, weights=w)


def load_watershed_geojson(bucket: str, prefix: str, site_no: str):
    try:
        obj = s3_client().get_object(
            Bucket=bucket, Key=f"{prefix}watersheds/per_gauge/{site_no}.geojson"
        )
    except Exception:
        return None
    return json.loads(obj["Body"].read())


# ---------- Worker ----------

def process_day_watershed(args: tuple):
    """Process all hourly files in one (product, day) for ALL watersheds.

    Returns (means_path, list_of_pixel_paths). Pixel paths are absent
    when per_pixel_zarr is False or kind is not QPE-like.
    """
    (day_iso, product, mask_records, cfg, shard_dir, save_pixels) = args
    day = date.fromisoformat(day_iso)
    fs = s3fs.S3FileSystem(anon=True)
    keys = list_mrms_keys_for_day(fs, cfg["mrms"]["bucket"], product["folder"], day)
    if not keys:
        return "", []

    wanted_hours = set(resolve_hours(product.get("hours")))
    keys = [k for k in keys if parse_mrms_timestamp(k).strftime("%H") in wanted_hours]
    if not keys:
        return "", []

    units = cfg["mrms"].get("units", "in")
    lower_limit = float(product.get("lower_limit", 0.0))
    means_rows = []
    pixel_rows_by_site: dict[str, list] = {m["site_no"]: [] for m in mask_records}

    with tempfile.TemporaryDirectory() as scratch_str:
        scratch = Path(scratch_str)
        for key in keys:
            try:
                raw = fs.cat(key)
                if key.endswith(".gz"):
                    raw = decompress_gz(raw)
                tmp_path = scratch / Path(key).name.replace(".gz", "")
                tmp_path.write_bytes(raw)
                ds = open_mrms_grib(tmp_path)
            except Exception as e:
                log.warning("Failed to open %s: %s", key, e)
                continue

            data_vars = list(ds.data_vars)
            if not data_vars:
                ds.close()
                continue
            arr, _, _ = canonicalize_mrms_grid(ds[data_vars[0]])
            ds.close()
            arr = apply_units(arr, kind=product["kind"], units=units)

            ts = parse_mrms_timestamp(key)
            for m in mask_records:
                site_no = m["site_no"]
                rows = m["row_idx"]
                cols = m["col_idx"]
                weights = m["weights"]
                if rows.size == 0:
                    continue
                vals = arr[rows, cols]

                # Watershed mean: use ALL pixels (do not apply lower_limit)
                valid = ~np.isnan(vals)
                if valid.any():
                    mean_val = float(np.average(vals[valid], weights=weights[valid]))
                else:
                    mean_val = None
                means_rows.append((ts, site_no, mean_val))

                # Sparse per-pixel: only pixels at/above lower_limit
                if save_pixels:
                    above = valid & (vals >= lower_limit)
                    if above.any():
                        idxs = np.nonzero(above)[0]
                        for pidx in idxs:
                            pixel_rows_by_site[site_no].append(
                                (ts, int(pidx), float(vals[pidx]))
                            )

    means_path = ""
    if means_rows:
        df = pd.DataFrame(means_rows, columns=["datetime_utc", "site_no", "value_mean"])
        means_path = str(shard_dir / f"means_{product['key']}_{day:%Y%m%d}.parquet")
        df.to_parquet(means_path, compression="zstd")

    pixel_paths = []
    if save_pixels:
        for site_no, rows in pixel_rows_by_site.items():
            if not rows:
                continue
            pdf = pd.DataFrame(rows, columns=["datetime_utc", "pixel_idx", "value"])
            ppath = shard_dir / f"pixels_{product['key']}_{site_no}_{day:%Y%m%d}.parquet"
            pdf.to_parquet(ppath, compression="zstd")
            pixel_paths.append(str(ppath))

    return means_path, pixel_paths


# ---------- Driver ----------

def fetch_template_grid(cfg: dict) -> dict:
    """Open one MRMS file (any product) to read the grid geometry."""
    fs = s3fs.S3FileSystem(anon=True)
    # Use the first configured product as template
    product = cfg["mrms"]["products"][0]
    prefix = f"{cfg['mrms']['bucket']}/CONUS/{product['folder']}/"
    days = sorted(fs.ls(prefix))
    if not days:
        raise RuntimeError("No MRMS data found at " + prefix)
    files = sorted(fs.ls(days[0]))
    if not files:
        raise RuntimeError("Empty MRMS day: " + days[0])
    raw = fs.cat(files[0])
    if files[0].endswith(".gz"):
        raw = decompress_gz(raw)
    with tempfile.TemporaryDirectory() as scratch_str:
        tmp = Path(scratch_str) / "template.grib2"
        tmp.write_bytes(raw)
        ds = open_mrms_grib(tmp)
        grid = build_grid_reference(ds[list(ds.data_vars)[0]])
        ds.close()
    return grid


def build_all_masks(cfg: dict, grid: dict) -> list[dict]:
    """Read per-gauge watershed GeoJSONs for active stations and build masks.

    Only processes stations in indiana_streamflow_sites_active.parquet
    (end_date >= 2020-10-14) so MRMS extraction is limited to gauges with
    data in the MRMS era. All watersheds still exist in S3 for future use.
    """
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    s3 = s3_client()

    # Load active station list and use it as the allowed set
    obj = s3.get_object(
        Bucket=bucket,
        Key=f"{prefix}stations/indiana_streamflow_sites_active.parquet",
    )
    active_sites = set(
        pq.read_table(io.BytesIO(obj["Body"].read()))
        .to_pandas()["site_no"]
        .astype(str)
    )
    log.info("Restricting watershed masks to %d active stations (end_date >= 2020-10-14)", len(active_sites))

    paginator = s3.get_paginator("list_objects_v2")
    site_ids = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}watersheds/per_gauge/"):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.endswith(".geojson"):
                sid = name.replace(".geojson", "")
                if sid in active_sites:
                    site_ids.append(sid)

    masks = []
    for sid in site_ids:
        feat = load_watershed_geojson(bucket, prefix, sid)
        if not feat:
            continue
        try:
            m = build_mask_for_polygon(feat, grid)
        except Exception as e:
            log.warning("Mask build failed for %s: %s", sid, e)
            continue
        masks.append({
            "site_no": sid,
            "row_idx": m.row_idx,
            "col_idx": m.col_idx,
            "weights": m.weights,
        })
    log.info("Built %d watershed masks", len(masks))
    return masks


def upload_zarr_per_site(shard_dir: Path, product_key: str, cfg: dict) -> None:
    """Reduce per-site, per-day pixel parquets for one product into a single
    Zarr per site, then upload that Zarr to S3."""
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    site_files: dict[str, list[Path]] = {}
    for p in shard_dir.glob(f"pixels_{product_key}_*.parquet"):
        # name: pixels_{product_key}_{site_no}_{YYYYMMDD}.parquet
        # product_key may contain underscores so reconstruct from the suffix
        stem = p.stem
        # strip the prefix "pixels_{product_key}_" and the trailing "_YYYYMMDD"
        rest = stem[len(f"pixels_{product_key}_"):]
        site_no, _ = rest.rsplit("_", 1)
        site_files.setdefault(site_no, []).append(p)

    for site_no, files in site_files.items():
        files = sorted(files)
        dfs = [pd.read_parquet(f) for f in files]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True).sort_values(["datetime_utc", "pixel_idx"])
        times = pd.DatetimeIndex(df["datetime_utc"].unique()).sort_values().tz_localize(None)
        pixels = np.sort(df["pixel_idx"].unique())
        time_idx = {t: i for i, t in enumerate(times)}
        pix_idx = {p: i for i, p in enumerate(pixels)}
        out = np.full((len(times), len(pixels)), np.nan, dtype="float32")
        for t, p, v in zip(df["datetime_utc"], df["pixel_idx"], df["value"]):
            out[time_idx[t], pix_idx[p]] = v

        ds = xr.Dataset(
            data_vars={"value": (("time", "pixel"), out)},
            coords={"time": times, "pixel": pixels},
            attrs={
                "site_no": site_no,
                "product_key": product_key,
                "units": cfg["mrms"].get("units", "in"),
            },
        )
        local_zarr = shard_dir / f"{product_key}_{site_no}.zarr"
        ds.to_zarr(local_zarr, mode="w", consolidated=True)
        for f in local_zarr.rglob("*"):
            if f.is_file():
                key = (f"{prefix}mrms/{product_key}/per_watershed_zarr/{site_no}.zarr/"
                       f"{f.relative_to(local_zarr).as_posix()}")
                s3_client().put_object(Bucket=bucket, Key=key, Body=f.read_bytes())
        log.info("Uploaded Zarr [%s] for %s (%d times, %d pixels)",
                 product_key, site_no, len(times), len(pixels))


def reduce_means_to_s3(paths: list[str], bucket: str, key: str) -> int:
    tables = [pq.read_table(p) for p in paths if Path(p).exists()]
    if not tables:
        return 0
    combined = pa.concat_tables(tables).sort_by([("datetime_utc", "ascending"),
                                                  ("site_no", "ascending")])
    buf = io.BytesIO()
    pq.write_table(combined, buf, compression="zstd")
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return combined.num_rows


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Fetching MRMS grid template...")
    grid = fetch_template_grid(cfg)
    log.info("MRMS grid: %s, dlat=%.4f, dlon=%.4f", grid["shape"], grid["dlat"], grid["dlon"])

    log.info("Building watershed masks...")
    masks = build_all_masks(cfg, grid)
    if not masks:
        log.error("No masks built - check that watershed GeoJSONs exist in S3.")
        return

    start = parse_iso_or_none(cfg["mrms"]["start_date"]) or date(2020, 10, 14)
    end = parse_iso_or_none(cfg["mrms"]["end_date"]) or (date.today() - timedelta(days=1))
    days = list(daterange(start, end))
    log.info("Processing %d days", len(days))

    products = cfg["mrms"]["products"]
    log.info("Products to extract: %s", [p["key"] for p in products])
    save_pixels_global = bool(cfg["mrms"].get("per_pixel_zarr", False))

    shard_dir = ensure_dir("./mrms_shards_watershed")
    n_workers = cfg["execution"]["max_workers_grib"]

    for product in products:
        log.info("=== Product: %s (%s) ===", product["key"], product["folder"])
        # Per-pixel output is most useful for QPE; for ARI/RQI/QPEFFG the
        # watershed mean is usually enough, so save pixels only for QPE-like
        # products even when per_pixel_zarr is true globally.
        save_pixels = save_pixels_global and product["kind"] in ("qpe", "qpeffg")

        args = [(d.isoformat(), product, masks, cfg, shard_dir, save_pixels) for d in days]
        means_paths: list[str] = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(process_day_watershed, a) for a in args]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    mp, _ = fut.result()
                    if mp:
                        means_paths.append(mp)
                except Exception as e:
                    log.error("Day failed: %s", e)
                if i % 30 == 0:
                    log.info("[%s][%d/%d] days complete", product["key"], i, len(futs))

        out_key = f"{prefix}mrms/{product['key']}/watershed_mean.parquet"
        n = reduce_means_to_s3(means_paths, bucket, out_key)
        log.info("Wrote %s (%d rows)", out_key, n)

        if save_pixels:
            log.info("Building per-watershed Zarr stores for %s...", product["key"])
            upload_zarr_per_site(shard_dir, product["key"], cfg)


if __name__ == "__main__":
    main()
