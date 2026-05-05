"""05_extract_mrms_nearest.py

For every MRMS product in `cfg.mrms.products` and every hourly timestamp
in [start_date, end_date], extract the value at the nearest grid pixel
to each Indiana streamflow gauge.

Strategy: for each timestamp, open the GRIB once and pull values for ALL
gauges in a single vectorized lookup. This is O(N_files), not
O(N_files x N_gauges).

Parallelized across (product, day) pairs with a process pool. Each
worker writes its chunk to a temp Parquet on local disk; a final reducer
concatenates into one Parquet per product on S3.

Writes (one set per product key in config):
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/nearest_pixel.parquet

Output schema:
    datetime_utc, site_no, value
where `value` is in inches for QPE products (config-driven), years for
ARI, percent for QPEFFG, or dimensionless for RQI.
"""
from __future__ import annotations

import io
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

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

log = logging.getLogger("05_mrms_nearest")


# ---------- Worker ----------

def process_day(args: tuple) -> str:
    """Process all hourly files in one (product, day); return shard path."""
    (day_iso, product, gauges_records, cfg, shard_dir) = args
    day = date.fromisoformat(day_iso)
    fs = s3fs.S3FileSystem(anon=True)  # MRMS is public; no credentials
    keys = list_mrms_keys_for_day(fs, cfg["mrms"]["bucket"], product["folder"], day)
    if not keys:
        return ""

    # Filter to requested hours (e.g. ["23"] for RQI_24H)
    wanted_hours = set(resolve_hours(product.get("hours")))
    keys = [k for k in keys if parse_mrms_timestamp(k).strftime("%H") in wanted_hours]
    if not keys:
        return ""

    site_ids = np.array([str(g["site_no"]) for g in gauges_records])
    lats = np.array([float(g["dec_lat_va"]) for g in gauges_records])
    lons = np.array([float(g["dec_long_va"]) for g in gauges_records])

    units = cfg["mrms"].get("units", "in")
    rows = []
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
            arr, lats_grid, lons_grid = canonicalize_mrms_grid(ds[data_vars[0]])
            ds.close()
            arr = apply_units(arr, kind=product["kind"], units=units)

            # Nearest-neighbor index lookup, vectorized over gauges
            lat_idx = np.abs(lats_grid[:, None] - lats[None, :]).argmin(axis=0)
            lon_idx = np.abs(lons_grid[:, None] - lons[None, :]).argmin(axis=0)
            vals = arr[lat_idx, lon_idx]

            ts = parse_mrms_timestamp(key)
            for sid, v in zip(site_ids, vals):
                rows.append((ts, sid, float(v) if not np.isnan(v) else None))

    if not rows:
        return ""
    df = pd.DataFrame(rows, columns=["datetime_utc", "site_no", "value"])
    out_path = shard_dir / f"{product['key']}_{day:%Y%m%d}.parquet"
    df.to_parquet(out_path, compression="zstd")
    return str(out_path)


# ---------- Driver ----------

def read_gauges(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites_active.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def reduce_shards_to_s3(shards: list[str], bucket: str, key: str) -> int:
    """Concatenate local Parquet shards and upload one combined Parquet to S3."""
    tables = [pq.read_table(p) for p in shards if Path(p).exists()]
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

    gauges = read_gauges(bucket, prefix)
    gauges = gauges.dropna(subset=["dec_lat_va", "dec_long_va"]).reset_index(drop=True)
    gauges_records = gauges[["site_no", "dec_lat_va", "dec_long_va"]].to_dict("records")
    log.info("MRMS nearest-pixel extraction for %d gauges", len(gauges))

    start = parse_iso_or_none(cfg["mrms"]["start_date"]) or date(2020, 10, 14)
    end = parse_iso_or_none(cfg["mrms"]["end_date"]) or (date.today() - timedelta(days=1))
    days = list(daterange(start, end))
    log.info("Processing %d days from %s to %s", len(days), start, end)

    products = cfg["mrms"]["products"]
    log.info("Products to extract: %s", [p["key"] for p in products])

    shard_dir = ensure_dir("./mrms_shards_nearest")
    n_workers = cfg["execution"]["max_workers_grib"]

    for product in products:
        log.info("=== Product: %s (%s) ===", product["key"], product["folder"])
        args = [(d.isoformat(), product, gauges_records, cfg, shard_dir) for d in days]
        paths: list[str] = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(process_day, a) for a in args]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    p = fut.result()
                    if p:
                        paths.append(p)
                except Exception as e:
                    log.error("Day failed: %s", e)
                if i % 30 == 0:
                    log.info("[%s][%d/%d] days complete", product["key"], i, len(futs))

        out_key = f"{prefix}mrms/{product['key']}/nearest_pixel.parquet"
        n = reduce_shards_to_s3(paths, bucket, out_key)
        log.info("Wrote %s (%d rows)", out_key, n)


if __name__ == "__main__":
    main()
