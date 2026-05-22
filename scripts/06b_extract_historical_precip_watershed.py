"""06b_extract_historical_precip_watershed.py

Extends the MRMS watershed-mean record backwards using two sources:

  1. Iowa State University MRMS archive (~2015-01-01 onwards):
     Same QPE_01H_Pass2 product, same GRIB2 format and grid as noaa-mrms-pds.
     Uses the same rasterio geometry_mask approach as script 06.

  2. NOAA Stage IV Multi-Sensor QPE (2002-01-01 onwards, fallback):
     4-km HRAP polar-stereographic grid; units mm → inches.
     Per-watershed pixel masks are built on the first valid Stage IV file using
     shapely point-in-polygon tests; contributions are cos(lat)-weighted.
     Falls back to the nearest pixel when no pixel centre falls inside a polygon.

For each hour: ISU MRMS is tried first; Stage IV is the fallback.

Gap-filling: reads the existing watershed_mean.parquet to determine the
earliest available timestamp and only downloads dates before that.

Config key (config.yaml):
    historical_precip:
      start_date: "2002-01-01"   # earliest possible; Stage IV starts here
      end_date:   "2020-10-13"   # day before noaa-mrms-pds starts

Reads:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites_active.parquet
    s3://<bucket>/<prefix>watersheds/per_gauge/*.geojson
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/watershed_mean.parquet   (optional)

Writes (prepends historical rows, sorted, deduped):
    s3://<bucket>/<prefix>mrms/<PRODUCT_KEY>/watershed_mean.parquet

Output schema (identical to script 06):
    datetime_utc, site_no, value_mean
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import s3fs
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import Point
from shapely.geometry import shape as shapely_shape

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
log = logging.getLogger("06b_historical_watershed")

ISU_MRMS_BASE   = "https://mtarchive.geol.iastate.edu"
ISU_STAGE4_BASE = "https://mesonet.agron.iastate.edu/archive/data"

STAGE4_START   = date(2002,  1,  1)
ISU_MRMS_START = date(2015,  1,  1)
MRMS_PDS_START = date(2020, 10, 14)

# The ISU MRMS archive carries GaugeCorr_QPE_01H (the pre-2020 predecessor to
# MultiSensor_QPE_01H_Pass2).  Hardcoded here — independent of config folder.
ISU_MRMS_FOLDER = "GaugeCorr_QPE_01H"
ISU_MRMS_FSTEM  = "GaugeCorr_QPE_01H_00.00"

MM_TO_IN = 1.0 / 25.4

# Stage IV masks built once on the first valid Stage IV file, then cached
_stage4_masks_cache: list | None = None
_stage4_masks_lock  = threading.Lock()


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 60) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        return r.content if r.status_code == 200 else None
    except requests.RequestException:
        return None


def fetch_isu_mrms(dt: datetime) -> bytes | None:
    fname = f"{ISU_MRMS_FSTEM}_{dt.strftime('%Y%m%d')}-{dt.strftime('%H')}0000.grib2.gz"
    url = (f"{ISU_MRMS_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}"
           f"/mrms/ncep/{ISU_MRMS_FOLDER}/{fname}")
    return _get(url)


def fetch_stage4(dt: datetime) -> bytes | None:
    base = f"{ISU_STAGE4_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}/stage4"
    stem = f"ST4.{dt.strftime('%Y%m%d%H')}.01h"
    for ext in (".grib", ".grb2", ".grb"):
        raw = _get(f"{base}/{stem}{ext}")
        if raw is not None:
            return raw
    return None


# ── MRMS grid + mask helpers ──────────────────────────────────────────────────

def _grid_from_da(da) -> dict:
    _, lats, lons = canonicalize_mrms_grid(da)
    dlat = float(np.abs(lats[1] - lats[0]))
    dlon = float(np.abs(lons[1] - lons[0]))
    top  = lats[0] + dlat / 2
    left = lons.min() - dlon / 2
    return {
        "lats": lats, "lons": lons, "dlat": dlat, "dlon": dlon,
        "transform": from_origin(left, top, dlon, dlat),
        "shape": (lats.size, lons.size),
    }


def fetch_mrms_grid(cfg: dict) -> dict:
    """Download one MRMS file from noaa-mrms-pds to read the CONUS grid."""
    fs = s3fs.S3FileSystem(anon=True)
    product = cfg["mrms"]["products"][0]
    prefix = f"{cfg['mrms']['bucket']}/CONUS/{product['folder']}/"
    days  = sorted(fs.ls(prefix))
    files = sorted(fs.ls(days[0]))
    raw = fs.cat(files[0])
    if files[0].endswith(".gz"):
        raw = decompress_gz(raw)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str) / "template.grib2"
        tmp.write_bytes(raw)
        ds = open_mrms_grib(tmp)
        dvars = list(ds.data_vars)
        grid = _grid_from_da(ds[dvars[0]])
        ds.close()
    return grid


def build_mrms_mask(geom_geojson: dict, grid: dict) -> dict | None:
    """Rasterize a watershed polygon onto the MRMS 0.01° CONUS grid."""
    geom = shapely_shape(
        geom_geojson["geometry"] if "geometry" in geom_geojson else geom_geojson
    )
    mask2d = geometry_mask(
        [geom],
        out_shape=grid["shape"],
        transform=grid["transform"],
        invert=True,
        all_touched=True,
    )
    rows, cols = np.nonzero(mask2d)
    if rows.size == 0:
        return None
    w = np.cos(np.deg2rad(grid["lats"][rows]))
    w = w / w.sum()
    return {"row_idx": rows, "col_idx": cols, "weights": w}


# ── Stage IV grid + mask helpers ──────────────────────────────────────────────

def open_stage4_grib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a Stage IV GRIB file; return (data_mm, lats_2d, lons_2d)."""
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
    raise ValueError(f"No lat/lon variables in Stage IV file: {path}")


def build_stage4_masks(
    lats_2d: np.ndarray,
    lons_2d: np.ndarray,
    gauges_list: list[dict],
) -> list[dict]:
    """Build per-watershed pixel masks on the Stage IV 2D HRAP grid.

    Uses bounding-box pre-filter then shapely point-in-polygon.
    Falls back to the single nearest pixel when the polygon is smaller than
    one Stage IV cell (~4 km).
    """
    from scipy.spatial import KDTree

    flat_lats = lats_2d.ravel()
    flat_lons = lons_2d.ravel()
    kd = KDTree(np.column_stack([flat_lats, flat_lons]))

    masks = []
    for g in gauges_list:
        geom = shapely_shape(
            g["geojson"]["geometry"] if "geometry" in g["geojson"] else g["geojson"]
        )
        minlon, minlat, maxlon, maxlat = geom.bounds

        candidates = np.nonzero(
            (flat_lons >= minlon) & (flat_lons <= maxlon) &
            (flat_lats >= minlat) & (flat_lats <= maxlat)
        )[0]

        inside: np.ndarray
        if candidates.size > 0:
            try:
                inside = np.array(
                    [idx for idx in candidates
                     if geom.contains(Point(float(flat_lons[idx]), float(flat_lats[idx])))]
                )
            except Exception:
                inside = np.array([], dtype=int)
        else:
            inside = np.array([], dtype=int)

        if inside.size == 0:
            _, nn = kd.query([g["lat"], g["lon"]])
            inside = np.array([nn])

        w = np.cos(np.deg2rad(flat_lats[inside]))
        w = w / w.sum() if w.sum() > 0 else np.ones(len(inside)) / len(inside)
        masks.append({"site_no": g["site_no"], "idxs": inside, "weights": w})

    log.info("Built Stage IV watershed masks for %d sites", len(masks))
    return masks


# ── Per-day worker ────────────────────────────────────────────────────────────

def process_day(args: tuple) -> str:
    global _stage4_masks_cache

    (day_iso, mrms_masks, gauges_list, shard_dir) = args
    day  = date.fromisoformat(day_iso)
    rows: list[tuple] = []

    with tempfile.TemporaryDirectory() as scratch_str:
        scratch = Path(scratch_str)
        for hour in range(24):
            dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)

            raw    = None
            source = None

            if day >= ISU_MRMS_START:
                raw = fetch_isu_mrms(dt)
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
                    arr, _, _ = canonicalize_mrms_grid(ds[dvars[0]])
                    ds.close()
                    arr = apply_units(arr, kind="qpe", units="in")

                    for m in mrms_masks:
                        vals = arr[m["row_idx"], m["col_idx"]]
                        valid = ~np.isnan(vals)
                        if valid.any():
                            mean_val = float(
                                np.average(vals[valid], weights=m["weights"][valid])
                            )
                        else:
                            mean_val = None
                        rows.append((dt, m["site_no"], mean_val))

                else:  # stage4
                    tmp = scratch / f"stage4_{hour:02d}.grb"
                    tmp.write_bytes(raw)
                    data_mm, lats_2d, lons_2d = open_stage4_grib(tmp)

                    # Build Stage IV masks exactly once across all threads
                    if _stage4_masks_cache is None:
                        with _stage4_masks_lock:
                            if _stage4_masks_cache is None:
                                _stage4_masks_cache = build_stage4_masks(
                                    lats_2d, lons_2d, gauges_list
                                )

                    flat_mm = data_mm.ravel()
                    for m in _stage4_masks_cache:
                        vals_mm = flat_mm[m["idxs"]].astype(float)
                        valid   = vals_mm >= 0
                        if valid.any():
                            mean_in = float(
                                np.average(
                                    vals_mm[valid] * MM_TO_IN,
                                    weights=m["weights"][valid],
                                )
                            )
                        else:
                            mean_in = None
                        rows.append((dt, m["site_no"], mean_in))

            except Exception as e:
                log.debug("Error %s %s: %s", source, dt, e)
                continue

    if not rows:
        return ""

    df = pd.DataFrame(rows, columns=["datetime_utc", "site_no", "value_mean"])
    df["value_mean"] = df["value_mean"].astype("float64")
    out = shard_dir / f"hist_watershed_{day:%Y%m%d}.parquet"
    df.to_parquet(out, compression="zstd")
    return str(out)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_gauges_with_geojsons(bucket: str, prefix: str) -> list[dict]:
    """Load active gauges and their watershed GeoJSONs from S3."""
    s3 = s3_client()

    obj = s3.get_object(
        Bucket=bucket,
        Key=f"{prefix}stations/indiana_streamflow_sites_active.parquet",
    )
    gauges = (
        pq.read_table(io.BytesIO(obj["Body"].read()))
        .to_pandas()
        .dropna(subset=["dec_lat_va", "dec_long_va"])
        .reset_index(drop=True)
    )

    result = []
    for _, row in gauges.iterrows():
        sid = str(row["site_no"])
        try:
            gj_obj = s3.get_object(
                Bucket=bucket, Key=f"{prefix}watersheds/per_gauge/{sid}.geojson"
            )
            geojson = json.loads(gj_obj["Body"].read())
        except Exception:
            log.debug("No watershed GeoJSON for %s — skipping", sid)
            continue
        result.append({
            "site_no": sid,
            "lat":     float(row["dec_lat_va"]),
            "lon":     float(row["dec_long_va"]),
            "geojson": geojson,
        })

    log.info("Gauges with watershed GeoJSON: %d", len(result))
    return result


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

    hist_cfg  = cfg.get("historical_precip", {})
    cfg_start = parse_iso_or_none(hist_cfg.get("start_date", "")) or STAGE4_START
    cfg_end   = parse_iso_or_none(hist_cfg.get("end_date",   "")) or (MRMS_PDS_START - timedelta(days=1))

    product_key = cfg["mrms"]["products"][0]["key"]
    parquet_key = f"{prefix}mrms/{product_key}/watershed_mean.parquet"

    # ── Load gauges + watershed GeoJSONs ──────────────────────────────────────
    gauges_list = read_gauges_with_geojsons(bucket, prefix)
    if not gauges_list:
        log.error("No gauges with watershed GeoJSONs — aborting.")
        return

    # ── Build MRMS masks (same 0.01° grid as noaa-mrms-pds and ISU MRMS) ─────
    log.info("Fetching MRMS grid reference from noaa-mrms-pds...")
    grid = fetch_mrms_grid(cfg)
    log.info("MRMS grid: %s, dlat=%.4f°, dlon=%.4f°",
             grid["shape"], grid["dlat"], grid["dlon"])

    mrms_masks: list[dict] = []
    for g in gauges_list:
        m = build_mrms_mask(g["geojson"], grid)
        if m is None:
            log.debug("Empty MRMS mask for %s — skipping", g["site_no"])
            continue
        m["site_no"] = g["site_no"]
        mrms_masks.append(m)
    log.info("MRMS watershed masks built: %d", len(mrms_masks))

    # ── Determine date range ──────────────────────────────────────────────────
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

    shard_dir = ensure_dir("./hist_shards_watershed")
    n_workers = cfg["execution"].get("max_workers_io", 8)

    args  = [(d.isoformat(), mrms_masks, gauges_list, shard_dir) for d in days]
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
