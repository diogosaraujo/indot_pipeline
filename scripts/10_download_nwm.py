"""10_download_nwm.py

Download National Water Model (NWM) channel outputs for Indiana USGS gauges,
matched to NHDPlus COMIDs via the NLDI service.  Three products are extracted:

  ┌─────────────────────────────────┬────────────────────────┬─────────┐
  │ Product                         │ Coverage               │ Source  │
  ├─────────────────────────────────┼────────────────────────┼─────────┤
  │ NWM Retrospective v3.0          │ Feb 1979 – Dec 2023    │ Zarr    │
  │ Standard Analysis & Assim (A&A) │ ~Sep 2018 – present    │ NetCDF  │
  │ Open-Loop A&A (no DA)           │ ~Sep 2018 – present    │ NetCDF  │
  └─────────────────────────────────┴────────────────────────┴─────────┘

Variables extracted per product:

  streamflow  (m³/s)   — all three products
  velocity    (m/s)    — all three products
  nudge       (m³/s)   — A&A only; DA correction added to model streamflow
  head_m      (m)      — Retrospective only; water surface elevation (NAVD88)
  stage_m     (m)      — all products; for Retro = head_m directly; for A&A
                         and Open-Loop derived by interpolating the HAND-based
                         Synthetic Rating Curves (SRC) stored in HYDRO_TBL_1D.nc

Each station's download is clipped to its USGS begin_date / end_date, further
bounded by the product's available period.

COMID matching
--------------
USGS gauges are matched to NHDPlus COMIDs through the NLDI service
(api.water.usgs.gov/nldi).  A separate table records the NHDPlus flowline
midpoint so the snap distance can be inspected.

SRC note
--------
The HAND-based SRC file (HYDRO_TBL_1D.nc) path is set in config.yaml under
nwm.src_s3_bucket / nwm.src_s3_key.  Update those values once the exact path
in noaa-nwm-pds is confirmed.  If the file is unreachable, stage_m is written
as NaN for A&A and Open-Loop (Retrospective is unaffected — head_m is direct).

Reads:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet

Writes:
    s3://<bucket>/<prefix>nwm/comid_locations.parquet
    s3://<bucket>/<prefix>nwm/retrospective.parquet
    s3://<bucket>/<prefix>nwm/analysis_assim.parquet
    s3://<bucket>/<prefix>nwm/open_loop.parquet

comid_locations schema:
    site_no, comid, usgs_lat, usgs_lon, comid_lat, comid_lon, distance_km

retrospective schema:
    site_no, comid, datetime_utc,
    streamflow_cms, velocity_ms, head_m, stage_m

analysis_assim schema:
    site_no, comid, datetime_utc,
    streamflow_cms, velocity_ms, nudge_cms, stage_m

open_loop schema:
    site_no, comid, datetime_utc,
    streamflow_cms, velocity_ms, stage_m
"""
from __future__ import annotations

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import s3fs
import xarray as xr

from utils import RetryPolicy, load_config, s3_client, with_retries, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("10_nwm")

# ── NWM product constants ─────────────────────────────────────────────────────

# Retrospective v3.0 — public Zarr store (AWS Open Data)
RETRO_ZARR   = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETRO_START  = pd.Timestamp("1979-02-01", tz="UTC")
RETRO_END    = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")

# Operational products — public NetCDF/HDF5 files (AWS Open Data)
OPS_BUCKET   = "noaa-nwm-pds"
OPS_START    = pd.Timestamp("2018-09-17", tz="UTC")   # NWM v2.0 archive start

NLDI_BASE    = "https://api.water.usgs.gov/nldi/linked-data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ── COMID lookup ──────────────────────────────────────────────────────────────

def _fetch_comid_info(site_no: str, timeout: int) -> Optional[dict]:
    """Return COMID and NHDPlus flowline midpoint for one USGS station."""
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

    # Get flowline geometry; use midpoint as the representative COMID location
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
    """Fetch COMIDs for all stations and compute snap distances."""
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
            if (info["comid_lat"] is not None
                    and not pd.isna(row.dec_lat_va)
                    and not pd.isna(row.dec_long_va)):
                dist_km = _haversine_km(
                    float(row.dec_lat_va), float(row.dec_long_va),
                    info["comid_lat"], info["comid_lon"],
                )
            results.append({
                "site_no":    str(row.site_no),
                "comid":      info["comid"],
                "usgs_lat":   float(row.dec_lat_va)  if not pd.isna(row.dec_lat_va)  else None,
                "usgs_lon":   float(row.dec_long_va) if not pd.isna(row.dec_long_va) else None,
                "comid_lat":  info["comid_lat"],
                "comid_lon":  info["comid_lon"],
                "distance_km": round(dist_km, 4) if dist_km is not None else None,
            })
            if i % 50 == 0:
                log.info("COMID lookup: %d / %d done", i, len(futs))

    return pd.DataFrame(results)


# ── HAND-based Synthetic Rating Curves ───────────────────────────────────────

def load_src_interpolators(comids: list[int], bucket: str, key: str) -> dict:
    """Load HYDRO_TBL_1D.nc from S3 and build per-COMID stage interpolators.

    The SRC file maps NHDPlus COMID → (discharge_pts, stage_pts) rating curve.
    Returns dict: comid (int) → callable(streamflow_cms) → stage_m.
    COMIDs not in the SRC file are silently omitted.

    Two variable-name conventions are handled:
      • NWM v2.x : Stage_1 … Stage_N  /  Discharge_1 … Discharge_N
      • NWM v3.0 : stage_ht_NQ  /  discharge_ht_NQ  (2-D arrays)
    """
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        ds = xr.open_dataset(io.BytesIO(obj["Body"].read()), engine="h5netcdf")
    except Exception as e:
        log.warning("Could not load SRC from s3://%s/%s: %s — stage_m will be NaN", bucket, key, e)
        return {}

    comid_set = set(comids)
    feature_ids = ds["feature_id"].values.astype(int)

    if "Stage_1" in ds:
        n_pts = sum(1 for v in ds.data_vars if v.startswith("Stage_"))
        stage_arr = np.stack([ds[f"Stage_{i}"].values    for i in range(1, n_pts + 1)], axis=1)
        q_arr     = np.stack([ds[f"Discharge_{i}"].values for i in range(1, n_pts + 1)], axis=1)
    elif "stage_ht_NQ" in ds:
        stage_arr = ds["stage_ht_NQ"].values       # (n_comids, n_pts)
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
        q_v = q_pts[valid]
        h_v = h_pts[valid]
        order = np.argsort(q_v)
        q_s, h_s = q_v[order], h_v[order]
        # Capture by value to avoid late-binding closure bug
        interpolators[int(fid)] = lambda q, _q=q_s, _h=h_s: float(
            np.interp(q, _q, _h, left=_h[0], right=_h[-1])
        )

    log.info("SRC: built interpolators for %d / %d COMIDs", len(interpolators), len(comids))
    return interpolators


def _apply_stage(df: pd.DataFrame, interp: dict) -> pd.DataFrame:
    """Vectorised stage derivation; NaN for COMIDs missing from SRC."""
    if not interp:
        df["stage_m"] = np.nan
        return df
    df["stage_m"] = [
        interp[c](q) if c in interp else np.nan
        for c, q in zip(df["comid"], df["streamflow_cms"])
    ]
    return df


# ── Retrospective v3.0 — Zarr extraction ─────────────────────────────────────

def extract_retrospective(
    comid_table: pd.DataFrame,
    station_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Open the retrospective Zarr store once and extract per-station windows.

    The store is opened with anonymous credentials (public AWS Open Data bucket).
    Only the COMIDs present in comid_table are selected; time is sliced per
    station to avoid materialising the full 44-year CONUS array.
    """
    comid_to_sites: dict[int, list[str]] = {}
    for _, row in comid_table.iterrows():
        comid_to_sites.setdefault(int(row["comid"]), []).append(str(row["site_no"]))

    comids_needed = list(comid_to_sites.keys())
    log.info("Retrospective: opening Zarr store, selecting %d COMIDs...", len(comids_needed))

    fs    = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(RETRO_ZARR, s3=fs)
    ds    = xr.open_zarr(store, consolidated=True)

    # Drop COMIDs that don't exist in this NWM domain (small intermittent reaches)
    available = set(ds["feature_id"].values.astype(int))
    comids_ok  = [c for c in comids_needed if c in available]
    missing    = [c for c in comids_needed if c not in available]
    if missing:
        log.warning("Retrospective: %d COMIDs not in NWM domain: %s", len(missing), missing[:10])

    # Detect the actual variable names — log them for transparency
    ds_vars = list(ds.data_vars)
    log.info("Retrospective Zarr variables: %s", ds_vars)

    # Stage/head variable name varies by NWM version in the Zarr store.
    # NWM v3.0 Zarr does not include Head; only streamflow and velocity are available.
    HEAD_CANDIDATES = ["Head", "head", "qlink"]   # q_lateral is lateral inflow, not stage
    head_var = next((v for v in HEAD_CANDIDATES if v in ds_vars), None)
    if head_var is None:
        log.warning("No head/stage variable found in Zarr store (tried %s); head_m will be NaN",
                    HEAD_CANDIDATES)

    retro_vars = ["streamflow", "velocity"]
    if head_var:
        retro_vars.append(head_var)

    sub = ds[retro_vars].sel(feature_id=comids_ok)

    # Compute the global time window needed across all stations so we can
    # bulk-load all COMIDs in one shot instead of 265 separate S3 requests.
    t_starts, t_ends = [], []
    for site_no, (begin, end) in station_periods.items():
        t0 = max(_to_utc(begin), RETRO_START).tz_localize(None)
        t1 = min(_to_utc(end),   RETRO_END).tz_localize(None)
        if t0 < t1:
            t_starts.append(t0)
            t_ends.append(t1)
    if not t_starts:
        return pd.DataFrame()

    global_t0 = min(t_starts)
    global_t1 = max(t_ends)
    log.info(
        "Loading Zarr subset into memory (%s → %s, %d COMIDs, vars=%s) — "
        "this may take 10–30 min depending on network...",
        global_t0.date(), global_t1.date(), len(comids_ok), retro_vars,
    )
    sub_loaded = sub.sel(time=slice(global_t0, global_t1)).load()
    log.info("Zarr data loaded. Slicing per-station windows...")

    parts: list[pd.DataFrame] = []
    for comid in comids_ok:
        site_nos = comid_to_sites[comid]
        comid_slice = sub_loaded.sel(feature_id=comid)

        for site_no in site_nos:
            begin, end = station_periods.get(site_no, (RETRO_START, RETRO_END))
            t0 = max(_to_utc(begin), RETRO_START).tz_localize(None)
            t1 = min(_to_utc(end),   RETRO_END).tz_localize(None)
            if t0 >= t1:
                continue

            base_cols = ["streamflow", "velocity"] + ([head_var] if head_var else [])
            df = (
                comid_slice.sel(time=slice(t0, t1))
                .to_dataframe()[base_cols]
                .reset_index()
                .rename(columns={
                    "time":       "datetime_utc",
                    "streamflow": "streamflow_cms",
                    "velocity":   "velocity_ms",
                    **({head_var: "head_m"} if head_var else {}),
                })
            )
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize("UTC")
            df["site_no"] = site_no
            df["comid"]   = comid
            if "head_m" not in df.columns:
                df["head_m"] = np.nan
            df["stage_m"] = df["head_m"]
            parts.append(df[["site_no", "comid", "datetime_utc",
                              "streamflow_cms", "velocity_ms", "head_m", "stage_m"]])

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── Operational data — hourly NetCDF/HDF5 extraction ─────────────────────────

def _ops_key(ts: pd.Timestamp, product: str) -> str:
    d = ts.strftime("%Y%m%d")
    h = ts.hour
    if product == "analysis_assim":
        return (f"nwm.{d}/analysis_assim/"
                f"nwm.t{h:02d}z.analysis_assim.channel_rt.tm00.conus.nc")
    else:
        return (f"nwm.{d}/analysis_assim_no_da/"
                f"nwm.t{h:02d}z.analysis_assim_no_da.channel_rt.tm00.conus.nc")


def _extract_one_hour(
    fs: s3fs.S3FileSystem,
    bucket: str,
    key: str,
    target_comids: np.ndarray,
    product: str,
    ts: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Open one hourly NWM NetCDF and return rows for the target COMIDs.

    Uses h5py with s3fs to read only the HDF5 dataset chunks that contain
    the requested feature_ids, avoiding a full CONUS download.
    """
    try:
        with fs.open(f"{bucket}/{key}", "rb") as fobj:
            with h5py.File(fobj, "r") as h:
                all_ids = h["feature_id"][:]
                idx = np.where(np.isin(all_ids, target_comids))[0]
                if len(idx) == 0:
                    return None
                # Sort indices — required for efficient HDF5 reads
                idx = np.sort(idx)
                df = pd.DataFrame({
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
    """Download all hourly NWM operational files that overlap any station's active period."""
    now = pd.Timestamp.utcnow()

    # Union window across all stations, clipped to operational archive
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

    hours = pd.date_range(global_start.floor("h"), global_end.floor("h"), freq="1h", tz="UTC")
    log.info("%s: %d hourly files to process (%s → %s)",
             product, len(hours), global_start.date(), global_end.date())

    fs = s3fs.S3FileSystem(anon=True)
    raw_parts: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _extract_one_hour, fs, OPS_BUCKET,
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
                log.info("%s: %d / %d hours processed (%d rows so far)",
                         product, i, len(hours),
                         sum(len(p) for p in raw_parts))

    if not raw_parts:
        log.error("%s: no data extracted", product)
        return pd.DataFrame()

    combined = pd.concat(raw_parts, ignore_index=True)

    # Expand comid → site_no (a COMID can match multiple sites if they share a reach)
    mapping = pd.DataFrame([
        {"comid": c, "site_no": s}
        for c, sites in comid_to_sites.items()
        for s in sites
    ])
    out = combined.merge(mapping, on="comid", how="inner")

    # Clip each row to the station's active window
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
    cfg        = load_config()
    bucket     = cfg["aws"]["output_bucket"]
    prefix     = cfg["aws"]["output_prefix"]
    timeout    = cfg["streamstats"]["request_timeout_sec"]
    max_io     = cfg["execution"]["max_workers_io"]
    nwm_cfg    = cfg.get("nwm", {})
    src_bucket = nwm_cfg.get("src_s3_bucket", OPS_BUCKET)
    src_key    = nwm_cfg.get("src_s3_key", "")

    # ── Station inventory ──────────────────────────────────────────────
    log.info("Loading station inventory...")
    inv = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    inv["site_no"] = inv["site_no"].astype(str)
    inv = inv[["site_no", "dec_lat_va", "dec_long_va", "begin_date", "end_date"]].copy()

    # Build station active-period dict (tz-naive; converted to UTC per-call)
    def _parse_date(v) -> pd.Timestamp:
        if pd.isna(v) or v is None:
            return pd.Timestamp.now()
        return pd.Timestamp(v)

    station_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        row["site_no"]: (_parse_date(row["begin_date"]), _parse_date(row["end_date"]))
        for _, row in inv.iterrows()
    }

    # ── COMID lookup ───────────────────────────────────────────────────
    log.info("Fetching NHDPlus COMIDs via NLDI (%d stations)...", len(inv))
    comid_table = build_comid_table(inv, timeout=timeout, max_workers=max_io)
    log.info("COMIDs resolved: %d / %d", len(comid_table), len(inv))

    write_parquet_to_s3(comid_table, bucket, f"{prefix}nwm/comid_locations.parquet")
    log.info("Wrote nwm/comid_locations.parquet (%d rows)", len(comid_table))

    if comid_table.empty:
        log.error("No COMIDs resolved — aborting.")
        return

    # ── SRC interpolators (for A&A and Open-Loop stage) ───────────────
    comids = comid_table["comid"].astype(int).tolist()
    src_interp: dict = {}
    if src_key:
        log.info("Loading HAND SRC from s3://%s/%s ...", src_bucket, src_key)
        src_interp = load_src_interpolators(comids, src_bucket, src_key)
    else:
        log.warning("nwm.src_s3_key not set in config — stage_m will be NaN for A&A / Open-Loop")

    # ── Retrospective v3.0 ─────────────────────────────────────────────
    log.info("Starting retrospective extraction...")
    retro = extract_retrospective(comid_table, station_periods)
    if not retro.empty:
        write_parquet_to_s3(retro, bucket, f"{prefix}nwm/retrospective.parquet")
        log.info("Wrote nwm/retrospective.parquet (%d rows, %d stations)",
                 len(retro), retro["site_no"].nunique())
    else:
        log.error("Retrospective extraction returned no data")

    # ── Operational products ───────────────────────────────────────────
    for product, out_key in [
        ("analysis_assim",    f"{prefix}nwm/analysis_assim.parquet"),
        ("analysis_assim_no_da", f"{prefix}nwm/open_loop.parquet"),
    ]:
        log.info("Starting %s extraction...", product)
        df = extract_operational(
            product, comid_table, station_periods, src_interp, max_workers=max_io,
        )
        if not df.empty:
            write_parquet_to_s3(df, bucket, out_key)
            log.info("Wrote %s (%d rows, %d stations)", out_key, len(df), df["site_no"].nunique())
        else:
            log.error("%s extraction returned no data", product)

    log.info("Done.")


if __name__ == "__main__":
    main()
