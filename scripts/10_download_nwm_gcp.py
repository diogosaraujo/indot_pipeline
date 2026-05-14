"""10_download_nwm_gcp.py

GCP variant of 10_download_nwm.py — operational products only.

Extracts the two NWM operational products for every Indiana gauge using the
complete archive in gs://national-water-model (Sep 2018 → present).  The
retrospective (1979-2023) is handled by the AWS script (10_download_nwm.py)
which reads the public Zarr store on noaa-nwm-retrospective-3-0-pds.

Because the operational source data and the VM are both in GCP (us-central1)
all heavy reads stay in-region — no egress charges.

Products extracted:

  ┌────────────────────────────────────┬────────────────────────┬─────────────┐
  │ Product                            │ Coverage               │ Source      │
  ├────────────────────────────────────┼────────────────────────┼─────────────┤
  │ Standard Analysis & Assim (A&A)    │ Sep 2018 – present     │ GCS NetCDF  │
  │ Open-Loop A&A (no DA)              │ Sep 2018 – present     │ GCS NetCDF  │
  └────────────────────────────────────┴────────────────────────┴─────────────┘

Reads:
    gs://<gcp.output_bucket>/<gcp.output_prefix>stations/indiana_streamflow_sites.parquet

Writes:
    gs://<gcp.output_bucket>/<gcp.output_prefix>nwm/comid_locations.parquet
    gs://<gcp.output_bucket>/<gcp.output_prefix>nwm/analysis_assim.parquet
    gs://<gcp.output_bucket>/<gcp.output_prefix>nwm/open_loop.parquet

Optionally (if aws.output_bucket is set in config_gcp.yaml):
    s3://<aws.output_bucket>/<aws.output_prefix>nwm/comid_locations.parquet
    s3://<aws.output_bucket>/<aws.output_prefix>nwm/analysis_assim.parquet
    s3://<aws.output_bucket>/<aws.output_prefix>nwm/open_loop.parquet
"""
from __future__ import annotations

import io
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import gcsfs
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import xarray as xr
import yaml
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("10_nwm_gcp")

# ── NWM product constants ─────────────────────────────────────────────────────

GCS_BUCKET  = "national-water-model"
OPS_START   = pd.Timestamp("2018-09-17", tz="UTC")

NLDI_BASE   = "https://api.water.usgs.gov/nldi/linked-data"


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ["config_gcp.yaml", os.path.join(script_dir, "..", "config_gcp.yaml")]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config_gcp.yaml not found; run from the project root directory.")


# ── GCS helpers ───────────────────────────────────────────────────────────────

_gcs: storage.Client | None = None


def _gcs_client() -> storage.Client:
    global _gcs
    if _gcs is None:
        _gcs = storage.Client()
    return _gcs


def _gcs_object_exists(bucket_name: str, blob_name: str) -> bool:
    return _gcs_client().bucket(bucket_name).blob(blob_name).exists()


def _read_parquet_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    data = _gcs_client().bucket(bucket_name).blob(blob_name).download_as_bytes()
    return pq.read_table(io.BytesIO(data)).to_pandas()


def _write_parquet_gcs(df: pd.DataFrame, bucket_name: str, blob_name: str) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    buf.seek(0)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_file(
        buf, content_type="application/octet-stream"
    )
    log.info("Wrote gs://%s/%s (%d rows)", bucket_name, blob_name, len(df))


# ── Optional S3 push ──────────────────────────────────────────────────────────

def _push_parquet_to_s3(df: pd.DataFrame, aws_cfg: dict, key: str) -> None:
    import boto3

    key_id  = aws_cfg.get("access_key_id")  or os.environ.get("AWS_ACCESS_KEY_ID")
    secret  = aws_cfg.get("secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    region  = aws_cfg.get("region", "us-east-1")
    s3_bkt  = aws_cfg.get("output_bucket", "")
    if not s3_bkt:
        return

    session = boto3.Session(
        aws_access_key_id=key_id or None,
        aws_secret_access_key=secret or None,
        region_name=region,
    )
    s3 = session.client("s3")
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    buf.seek(0)
    s3.put_object(Bucket=s3_bkt, Key=key, Body=buf.getvalue())
    log.info("Pushed to s3://%s/%s (%d rows)", s3_bkt, key, len(df))


# ── Shared helpers ────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ── COMID lookup ──────────────────────────────────────────────────────────────

def _fetch_comid_info(site_no: str, timeout: int) -> Optional[dict]:
    r = requests.get(f"{NLDI_BASE}/nwissite/USGS-{site_no}", timeout=timeout)
    if r.status_code != 200:
        return None
    features = r.json().get("features", [])
    if not features:
        return None
    comid = features[0].get("properties", {}).get("comid")
    if not comid:
        return None
    comid = int(comid)

    comid_lat = comid_lon = None
    r2 = requests.get(f"{NLDI_BASE}/comid/{comid}", timeout=timeout)
    if r2.status_code == 200:
        fl_feats = r2.json().get("features", [])
        if fl_feats:
            geom = fl_feats[0].get("geometry", {})
            coords = geom.get("coordinates", [])
            if coords:
                mid = coords[len(coords) // 2]
                comid_lon, comid_lat = float(mid[0]), float(mid[1])

    return {"comid": comid, "comid_lat": comid_lat, "comid_lon": comid_lon}


def build_comid_table(inv: pd.DataFrame, timeout: int, max_workers: int) -> pd.DataFrame:
    rows = list(inv.itertuples(index=False))
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_fetch_comid_info, str(row.site_no), timeout): row
            for row in rows
        }
        for i, (fut, row) in enumerate(futs.items(), 1):
            try:
                info = fut.result()
            except Exception as e:
                log.warning("%s: COMID lookup failed: %s", row.site_no, e)
                info = None
            if info is None:
                continue
            dist_km = None
            if (
                info["comid_lat"] is not None
                and not pd.isna(row.dec_lat_va)
                and not pd.isna(row.dec_long_va)
            ):
                dist_km = _haversine_km(
                    float(row.dec_lat_va), float(row.dec_long_va),
                    info["comid_lat"], info["comid_lon"],
                )
            results.append({
                "site_no":     str(row.site_no),
                "comid":       info["comid"],
                "usgs_lat":    float(row.dec_lat_va)  if not pd.isna(row.dec_lat_va)  else None,
                "usgs_lon":    float(row.dec_long_va) if not pd.isna(row.dec_long_va) else None,
                "comid_lat":   info["comid_lat"],
                "comid_lon":   info["comid_lon"],
                "distance_km": round(dist_km, 4) if dist_km is not None else None,
            })
            if i % 50 == 0:
                log.info("COMID lookup: %d / %d done", i, len(futs))

    return pd.DataFrame(results)


# ── HAND-based Synthetic Rating Curves ───────────────────────────────────────

def load_src_interpolators(comids: list[int], nwm_bucket: str, key: str) -> dict:
    """Load HYDRO_TBL_1D.nc from the public NWM GCS bucket via gcsfs."""
    if not key:
        log.warning("gcp.nwm_src_key not set — stage_m will be NaN for A&A / Open-Loop")
        return {}
    try:
        gcs = gcsfs.GCSFileSystem(token="anon")
        with gcs.open(f"{nwm_bucket}/{key}", "rb") as f:
            data = f.read()
        ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    except Exception as e:
        log.warning(
            "Could not load SRC from gs://%s/%s: %s — stage_m will be NaN", nwm_bucket, key, e
        )
        return {}

    comid_set   = set(comids)
    feature_ids = ds["feature_id"].values.astype(int)

    if "Stage_1" in ds:
        n_pts     = sum(1 for v in ds.data_vars if v.startswith("Stage_"))
        stage_arr = np.stack([ds[f"Stage_{i}"].values    for i in range(1, n_pts + 1)], axis=1)
        q_arr     = np.stack([ds[f"Discharge_{i}"].values for i in range(1, n_pts + 1)], axis=1)
    elif "stage_ht_NQ" in ds:
        stage_arr = ds["stage_ht_NQ"].values
        q_arr     = ds["discharge_ht_NQ"].values
    else:
        log.error("SRC file has unrecognised variable layout; stage_m will be NaN")
        return {}

    interpolators: dict = {}
    for idx, fid in enumerate(feature_ids):
        if int(fid) not in comid_set:
            continue
        q_pts = q_arr[idx]
        h_pts = stage_arr[idx]
        valid = np.isfinite(q_pts) & np.isfinite(h_pts)
        if valid.sum() < 2:
            continue
        q_v, h_v = q_pts[valid], h_pts[valid]
        order    = np.argsort(q_v)
        q_s, h_s = q_v[order], h_v[order]
        interpolators[int(fid)] = lambda q, _q=q_s, _h=h_s: float(
            np.interp(q, _q, _h, left=_h[0], right=_h[-1])
        )

    log.info("SRC: built interpolators for %d / %d COMIDs", len(interpolators), len(comids))
    return interpolators


def _apply_stage(df: pd.DataFrame, interp: dict) -> pd.DataFrame:
    if not interp:
        df["stage_m"] = np.nan
        return df
    df["stage_m"] = [
        interp[c](q) if c in interp else np.nan
        for c, q in zip(df["comid"], df["streamflow_cms"])
    ]
    return df


# ── Operational data — hourly NetCDF/HDF5 extraction ─────────────────────────

def _ops_key(ts: pd.Timestamp, product: str) -> str:
    d = ts.strftime("%Y%m%d")
    h = ts.hour
    if product == "analysis_assim":
        return (
            f"nwm.{d}/analysis_assim/"
            f"nwm.t{h:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
        )
    return (
        f"nwm.{d}/analysis_assim_no_da/"
        f"nwm.t{h:02d}z.analysis_assim_no_da.channel_rt.tm00.conus.nc"
    )


def _extract_one_hour(
    fs,
    bucket: str,
    key: str,
    target_comids: np.ndarray,
    product: str,
    ts: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    try:
        with fs.open(f"{bucket}/{key}", "rb") as fobj:
            with h5py.File(fobj, "r") as h:
                all_ids = h["feature_id"][:]
                idx     = np.where(np.isin(all_ids, target_comids))[0]
                if len(idx) == 0:
                    return None
                idx = np.sort(idx)
                df  = pd.DataFrame({
                    "comid":          all_ids[idx].astype(int),
                    "streamflow_cms": h["streamflow"][idx].astype(float),
                    "velocity_ms":    h["velocity"][idx].astype(float),
                })
                if product == "analysis_assim" and "nudge" in h:
                    df["nudge_cms"] = h["nudge"][idx].astype(float)
                df["datetime_utc"] = ts
                return df
    except Exception as e:
        log.debug("%s %s: %s", product, ts.isoformat(), e)
        return None


def extract_operational(
    product: str,
    comid_table: pd.DataFrame,
    station_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    src_interp: dict,
    max_workers: int,
) -> pd.DataFrame:
    now = pd.Timestamp.utcnow()

    active_windows = []
    for site_no, (begin, end) in station_periods.items():
        t0 = max(_to_utc(begin), OPS_START)
        t1 = min(_to_utc(end) if pd.notna(end) else now, now)
        if t0 < t1:
            active_windows.append((t0, t1))
    if not active_windows:
        log.warning("%s: no stations overlap operational period", product)
        return pd.DataFrame()

    global_start = min(w[0] for w in active_windows)
    global_end   = max(w[1] for w in active_windows)

    comid_to_sites: dict[int, list[str]] = {}
    for _, row in comid_table.iterrows():
        comid_to_sites.setdefault(int(row["comid"]), []).append(str(row["site_no"]))
    target_comids = np.array(list(comid_to_sites.keys()), dtype=np.int64)

    hours = pd.date_range(
        global_start.floor("h"), global_end.floor("h"), freq="1h", tz="UTC"
    )
    log.info(
        "%s: %d hourly files to process (%s → %s)",
        product, len(hours), global_start.date(), global_end.date(),
    )

    fs = gcsfs.GCSFileSystem(token="anon")
    raw_parts: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _extract_one_hour, fs, GCS_BUCKET,
                _ops_key(ts, product), target_comids, product, ts,
            ): ts
            for ts in hours
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                df = fut.result()
            except Exception as e:
                log.debug("%s hour %d: %s", product, i, e)
                df = None
            if df is not None:
                raw_parts.append(df)
            if i % 720 == 0:
                log.info(
                    "%s: %d / %d hours processed (%d rows so far)",
                    product, i, len(hours), sum(len(p) for p in raw_parts),
                )

    if not raw_parts:
        log.error("%s: no data extracted", product)
        return pd.DataFrame()

    combined = pd.concat(raw_parts, ignore_index=True)

    mapping = pd.DataFrame([
        {"comid": c, "site_no": s}
        for c, sites in comid_to_sites.items()
        for s in sites
    ])
    out = combined.merge(mapping, on="comid", how="inner")

    period_map = {
        s: (max(_to_utc(b), OPS_START), min(_to_utc(e) if pd.notna(e) else now, now))
        for s, (b, e) in station_periods.items()
    }
    mask = out.apply(
        lambda r: (
            r["site_no"] in period_map
            and period_map[r["site_no"]][0] <= r["datetime_utc"] <= period_map[r["site_no"]][1]
        ),
        axis=1,
    )
    out = out[mask].copy()
    out = _apply_stage(out, src_interp)

    cols = ["site_no", "comid", "datetime_utc", "streamflow_cms", "velocity_ms"]
    if product == "analysis_assim":
        cols.append("nudge_cms")
    cols.append("stage_m")
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg        = _load_config()
    gcp_cfg    = cfg["gcp"]
    aws_cfg    = cfg.get("aws", {})
    bucket     = gcp_cfg["output_bucket"]
    prefix     = gcp_cfg["output_prefix"]
    nwm_bucket = gcp_cfg.get("nwm_bucket", GCS_BUCKET)
    src_key    = gcp_cfg.get("nwm_src_key", "")
    max_io     = cfg.get("execution", {}).get("max_workers_io", 8)
    timeout    = 30  # seconds for NLDI requests

    # ── Station inventory ──────────────────────────────────────────────────────
    log.info("Loading station inventory from GCS...")
    inv = _read_parquet_gcs(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    inv["site_no"] = inv["site_no"].astype(str)
    inv = inv[["site_no", "dec_lat_va", "dec_long_va", "begin_date", "end_date"]].copy()

    def _parse_date(v) -> pd.Timestamp:
        if pd.isna(v) or v is None:
            return pd.Timestamp.now()
        return pd.Timestamp(v)

    station_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        row["site_no"]: (_parse_date(row["begin_date"]), _parse_date(row["end_date"]))
        for _, row in inv.iterrows()
    }

    # ── COMID lookup ───────────────────────────────────────────────────────────
    log.info("Fetching NHDPlus COMIDs via NLDI (%d stations)...", len(inv))
    comid_table = build_comid_table(inv, timeout=timeout, max_workers=max_io)
    log.info("COMIDs resolved: %d / %d", len(comid_table), len(inv))

    comid_blob = f"{prefix}nwm/comid_locations.parquet"
    _write_parquet_gcs(comid_table, bucket, comid_blob)
    if aws_cfg.get("output_bucket"):
        _push_parquet_to_s3(comid_table, aws_cfg, f"{aws_cfg.get('output_prefix', '')}nwm/comid_locations.parquet")

    if comid_table.empty:
        log.error("No COMIDs resolved — aborting.")
        return

    # ── SRC interpolators ──────────────────────────────────────────────────────
    comids = comid_table["comid"].astype(int).tolist()
    src_interp = load_src_interpolators(comids, nwm_bucket, src_key)

    # ── Operational products ───────────────────────────────────────────────────
    product_blobs = {
        "analysis_assim":       f"{prefix}nwm/analysis_assim.parquet",
        "analysis_assim_no_da": f"{prefix}nwm/open_loop.parquet",
    }
    aws_keys = {
        "analysis_assim":       f"{aws_cfg.get('output_prefix', '')}nwm/analysis_assim.parquet",
        "analysis_assim_no_da": f"{aws_cfg.get('output_prefix', '')}nwm/open_loop.parquet",
    }

    for product, blob in product_blobs.items():
        log.info("Starting %s extraction...", product)
        df = extract_operational(product, comid_table, station_periods, src_interp, max_io)
        if not df.empty:
            _write_parquet_gcs(df, bucket, blob)
            if aws_cfg.get("output_bucket"):
                _push_parquet_to_s3(df, aws_cfg, aws_keys[product])
        else:
            log.error("%s extraction returned no data", product)

    log.info("Done.")


if __name__ == "__main__":
    main()
