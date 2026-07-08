"""visualize_bridge_event.py — per-bridge event precip + streamflow vs ARI

Two stacked panels for a single (scour-critical) bridge during a storm event:

  Top   — hourly MRMS QPE (1-h Pass2) sampled at the bridge grid cell over the event
          window, with a horizontal reference line at the Atlas-14 1-HOUR depth for the
          return period NEAREST the peak-hour's 1-h ARI (e.g. peak-hour ARI 205 yr -> P200).
  Bottom— NWM operational A&A (analysis_assim, with data assimilation) streamflow for the
          COMID crossing the bridge, with a horizontal reference line at the flow return
          period NEAREST the peak flow's ARI.  The flow-ARI curve is a Bulletin-17C
          Log-Pearson III fit on the NWM RETROSPECTIVE v3.0 (open-loop, no-DA) annual
          maxima at that COMID — the same method as 04c, computed on the fly because the
          bridge's reach is not one of the 106 gauge reaches 04c pre-fits.

Both panels share the event-window time axis.

COMID resolution: the NHDPlus reach the bridge point sits on, via the NLDI
`comid/position` service (same source script 10 uses); override with --comid.

Caching (skip with --refresh):
    events/bridge_<asset>_<event>/series.parquet          (MRMS + A&A point series)
    flow_stats/bridge_nwm_lp3/<comid>.parquet             (LP3 quantiles + annual maxima)

Reads (public, no creds): noaa-mrms-pds, noaa-nwm-pds, noaa-nwm-retrospective-3-0-pds
Reads (pipeline bucket):  bridge_coverage_flags, atlas14, indiana_streamflow_sites
Writes: results/bridge_<asset>_<event>.png  and  s3://<bucket>/<prefix>events/bridge_.../

Usage (on EC2):
    python scripts/visualize_bridge_event.py                       # 062-31-07183, 2026-06-09
    python scripts/visualize_bridge_event.py --asset 062-31-07183 --event-date 2026-06-09
    python scripts/visualize_bridge_event.py --comid 10357264 --refresh
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import s3fs
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).parent))
from utils import (apply_units, canonicalize_mrms_grid, decompress_gz, load_config,
                   open_mrms_grib, s3_object_exists, write_bytes_to_s3, write_parquet_to_s3)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bridge_event")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_ASSET   = "062-31-07183"
DEFAULT_EVENT   = date(2026, 6, 9)
WIN_DAYS_BEFORE = 1                 # window = [event-1d 00Z, event+2d 00Z)  (72 hourly steps)
WIN_DAYS_AFTER  = 2

MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_1H_FOLDER  = "MultiSensor_QPE_01H_Pass2_00.00"
NWM_BUCKET      = "noaa-nwm-pds"

BRIDGE_KEY      = "analysis/bridge_coverage/bridge_coverage_flags.parquet"
ASSET_COL       = "Asset Name"
BLAT_COL        = "(16) Latitude:"
BLON_COL        = "(17) Longitude:"
SCOUR_COL       = "scour_critical"
ATLAS14_KEY     = "atlas14/precipitation_frequency.parquet"
INV_KEY         = "stations/indiana_streamflow_sites.parquet"

# Standard return periods for snapping the "nearest ARI".
PRECIP_RPS      = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]   # Atlas-14 ladder (1-h)
FLOW_RPS        = [2, 5, 10, 25, 50, 100, 200, 500]            # LP3 ladder (04c max = 500)
ATLAS14_DUR_HR  = 1                 # user choice: hourly MRMS -> 1-h Atlas-14
CFS_PER_CMS     = 35.3146667

OUT_DIR         = "results"
PRECIP_BAR_C    = "#1f78b4"
PRECIP_LINE_C   = "#e31a1c"
FLOW_LINE_C     = "#0f7d7d"
FLOW_ARI_C      = "#b30000"

os.makedirs(OUT_DIR, exist_ok=True)


# ── Small helpers ─────────────────────────────────────────────────────────────

def _anon_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True)


def _safe(asset: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", asset).strip("_")


def nearest_standard_rp(ari: float, rps: list[int]) -> int:
    """Standard RP closest to `ari` in log10 space (205.2 -> 200)."""
    la = np.log10(max(ari, 1.0001))
    return min(rps, key=lambda r: abs(np.log10(r) - la))


def ari_from_value(value: float, rps: np.ndarray, vals: np.ndarray) -> tuple[float, bool]:
    """Return period of `value` by log-log interpolation of the (RP -> value) curve.

    Returns (ari_years, capped).  `capped` is True when `value` exceeds the largest
    tabulated quantile (ARI clamped to the max RP) or falls below the smallest.
    """
    order = np.argsort(vals)
    v = np.asarray(vals)[order]
    r = np.asarray(rps)[order]
    good = np.isfinite(v) & (v > 0)
    v, r = v[good], r[good]
    if v.size < 2 or not np.isfinite(value) or value <= 0:
        return float("nan"), True
    lv, lr = np.log10(v), np.log10(r)
    capped = value >= v[-1] or value <= v[0]
    ari = 10.0 ** float(np.interp(np.log10(value), lv, lr))
    return ari, capped


# ── Bridge + COMID ────────────────────────────────────────────────────────────

def load_bridge(bucket: str, prefix: str, b04b, asset: str) -> dict:
    df = b04b._read_parquet_s3(bucket, f"{prefix}{BRIDGE_KEY}",
                               columns=[ASSET_COL, BLAT_COL, BLON_COL, SCOUR_COL])
    df[ASSET_COL] = df[ASSET_COL].astype(str).str.strip()
    row = df[df[ASSET_COL] == asset]
    if row.empty:
        raise SystemExit(f"Asset '{asset}' not found in {BRIDGE_KEY}")
    r = row.iloc[0]
    return {"asset": asset, "lat": float(r[BLAT_COL]), "lon": float(r[BLON_COL]),
            "scour": bool(r[SCOUR_COL])}


def resolve_comid(lat: float, lon: float) -> int:
    r = requests.get("https://api.water.usgs.gov/nldi/linked-data/comid/position",
                     params={"coords": f"POINT({lon} {lat})"}, timeout=60)
    r.raise_for_status()
    return int(r.json()["features"][0]["properties"]["comid"])


# ── MRMS point series ─────────────────────────────────────────────────────────

def _mrms_key(ts: pd.Timestamp) -> str:
    d = ts.strftime("%Y%m%d")
    fname = f"MRMS_{MRMS_1H_FOLDER}_{d}-{ts.hour:02d}0000.grib2.gz"
    return f"{MRMS_BUCKET}/CONUS/{MRMS_1H_FOLDER}/{d}/{fname}"


def mrms_at_point(fs, hours: list[pd.Timestamp], lat: float, lon: float) -> np.ndarray:
    """Hourly MRMS 1-h QPE (inches) at the grid cell nearest the bridge."""
    out = np.full(len(hours), np.nan)
    ii = jj = None
    for k, ts in enumerate(hours):
        key = _mrms_key(ts)
        try:
            with fs.open(key, "rb") as f:
                raw = f.read()
        except Exception:
            continue
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(decompress_gz(raw)); tmp_path = tmp.name
        try:
            ds = open_mrms_grib(tmp_path)
            var = list(ds.data_vars)[0]
            arr, lats, lons = canonicalize_mrms_grid(ds[var])
            arr = apply_units(arr, kind="qpe", units="in")
        finally:
            os.unlink(tmp_path)
        if ii is None:
            ii = int(np.argmin(np.abs(lats - lat)))
            jj = int(np.argmin(np.abs(lons - lon)))
            log.info("MRMS cell for bridge: lat %.4f lon %.4f", lats[ii], lons[jj])
        v = arr[ii, jj]
        out[k] = 0.0 if not np.isfinite(v) else float(v)
    log.info("MRMS: %d/%d hours retrieved", int(np.isfinite(out).sum()), len(hours))
    return out


# ── NWM A&A operational point series ──────────────────────────────────────────

def _nwm_aa_key(ts: pd.Timestamp) -> str:
    return (f"{NWM_BUCKET}/nwm.{ts:%Y%m%d}/analysis_assim/"
            f"nwm.t{ts.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc")


def nwm_aa_at_comid(fs, hours: list[pd.Timestamp], comid: int) -> np.ndarray:
    """Hourly NWM A&A streamflow (m3/s) at `comid` (tm00 nowcast per hour).

    NWM channel_rt `streamflow` is stored as PACKED INTEGERS with a scale_factor
    (0.01) and add_offset.  xarray applies these automatically (so the retrospective
    is already in m3/s), but h5py returns the RAW ints — they must be unpacked by
    hand: value = raw * scale_factor + add_offset.  Skipping this inflates flow ~100x.
    """
    import h5py
    out = np.full(len(hours), np.nan)
    idx = None
    sf, ao = 1.0, 0.0
    for k, ts in enumerate(hours):
        key = _nwm_aa_key(ts)
        try:
            with fs.open(key, "rb") as fobj, h5py.File(fobj, "r") as h:
                dset = h["streamflow"]
                if idx is None:
                    ids = h["feature_id"][:].astype(np.int64)
                    hit = np.where(ids == comid)[0]
                    if hit.size == 0:
                        log.warning("COMID %d absent from NWM feature_id list.", comid)
                        return out
                    idx = int(hit[0])
                    # netCDF attrs come back as 1-element arrays via h5py — ravel to scalar.
                    sf = float(np.asarray(dset.attrs.get("scale_factor", 1.0)).ravel()[0])
                    ao = float(np.asarray(dset.attrs.get("add_offset", 0.0)).ravel()[0])
                    log.info("NWM streamflow packing: scale_factor=%s add_offset=%s", sf, ao)
                raw = float(dset[idx])
                q = raw * sf + ao
                out[k] = np.nan if q < 0 else q
        except Exception:
            continue
    peak = float(np.nanmax(out)) if np.isfinite(out).any() else float("nan")
    log.info("NWM A&A: %d/%d hours retrieved; peak=%.1f m3/s",
             int(np.isfinite(out).sum()), len(hours), peak)
    return out


# ── Atlas-14 1-h depth ladder at the bridge point ─────────────────────────────

def atlas14_1h_depths(bucket: str, prefix: str, b04b, lat: float, lon: float) -> dict[int, float]:
    """Atlas-14 ATLAS14_DUR_HR-hour depth (in) interpolated to (lat,lon) for each RP."""
    a14 = b04b._read_parquet_s3(bucket, f"{prefix}{ATLAS14_KEY}")
    a14["site_no"] = a14["site_no"].astype(str)
    a14 = a14[a14["duration_hr"] == ATLAS14_DUR_HR]
    inv = b04b._read_parquet_s3(bucket, f"{prefix}{INV_KEY}",
                                columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    a14 = a14.merge(inv, on="site_no", how="left").dropna(
        subset=["dec_lat_va", "dec_long_va", "depth_in"])
    depths: dict[int, float] = {}
    for rp in PRECIP_RPS:
        g = a14[a14["return_period_yr"] == rp]
        if len(g) < 3:
            continue
        src = g[["dec_lat_va", "dec_long_va"]].to_numpy(float)
        val = g["depth_in"].to_numpy(float)
        d = griddata(src, val, [[lat, lon]], method="linear")[0]
        if not np.isfinite(d):
            d = griddata(src, val, [[lat, lon]], method="nearest")[0]
        if np.isfinite(d):
            depths[rp] = float(d)
    return depths


# ── Retrospective (open-loop) LP3 flow ladder at the COMID (04c method) ────────

def retro_lp3_flows(bucket: str, prefix: str, m04c, comid: int, lat: float,
                    refresh: bool) -> Optional[dict]:
    """LP3 flow quantiles (cfs) at `comid` from NWM RETROSPECTIVE v3.0 annual maxima,
    using 04c's exact Bulletin-17C machinery.  Cached per COMID."""
    cache = f"{prefix}flow_stats/bridge_nwm_lp3/{comid}.parquet"
    b04b = m04c.b04b
    if not refresh and s3_object_exists(bucket, cache):
        log.info("Reading cached LP3 flows: s3://%s/%s", bucket, cache)
        c = b04b._read_parquet_s3(bucket, cache)
        q = {int(r.return_period_yr): float(r.q_cfs) for r in c.itertuples()}
        meta = {"n_years": int(c["n_years"].iloc[0]),
                "wy_start": int(c["wy_start"].iloc[0]),
                "wy_end": int(c["wy_end"].iloc[0]),
                "method": str(c["method"].iloc[0])}
        return {"q_cfs": q, **meta}

    nwm10 = m04c._load_nwm10()
    comid_table = pd.DataFrame({"site_no": [f"bridge_{comid}"], "comid": [comid]})
    periods = {f"bridge_{comid}": (nwm10.RETRO_START, nwm10.RETRO_END)}
    log.info("Extracting NWM retrospective for COMID %d (opens Zarr; a few min)...", comid)
    retro = nwm10.extract_retrospective(comid_table, periods)
    if retro is None or retro.empty:
        log.error("No retrospective data for COMID %d.", comid)
        return None
    ann = m04c.annual_max_series(retro)          # water-year maxima in cfs
    if len(ann) < b04b.MIN_YEARS:
        log.error("Only %d water years at COMID %d (< %d) — no LP3.",
                  len(ann), comid, b04b.MIN_YEARS)
        return None
    params = b04b.fit_lp3(np.log10(ann.values), lat)
    q = {rp: float(b04b.lp3_quantile(rp, params["mean_log"], params["std_log"],
                                     params["skew"])) for rp in FLOW_RPS}
    meta = {"n_years": int(len(ann)), "wy_start": int(ann.index.min()),
            "wy_end": int(ann.index.max()), "method": params.get("fitting_method", "MOM")}
    cache_df = pd.DataFrame({"return_period_yr": list(q.keys()),
                             "q_cfs": list(q.values())})
    for kx, vx in meta.items():
        cache_df[kx] = vx
    write_parquet_to_s3(cache_df, bucket, cache)
    log.info("Cached LP3 flows -> s3://%s/%s", bucket, cache)
    return {"q_cfs": q, **meta}


# ── Series cache (MRMS + A&A) ─────────────────────────────────────────────────

def load_or_build_series(bucket, prefix, b04b, fs, hours, lat, lon, comid,
                         asset, event, refresh):
    # _v2: v1 stored raw (unscaled) NWM streamflow — bump so stale caches are ignored.
    key = f"{prefix}events/bridge_{_safe(asset)}_{event:%Y%m%d}/series_v2.parquet"
    if not refresh and s3_object_exists(bucket, key):
        log.info("Reading cached series: s3://%s/%s", bucket, key)
        df = b04b._read_parquet_s3(bucket, key)
        if len(df) == len(hours):
            return df["mrms_in"].to_numpy(float), df["nwm_aa_cms"].to_numpy(float)
        log.warning("Cached series length %d != window %d — rebuilding.", len(df), len(hours))
    mrms = mrms_at_point(fs, hours, lat, lon)
    aa   = nwm_aa_at_comid(fs, hours, comid)
    df = pd.DataFrame({"datetime_utc": [h.tz_convert("UTC") for h in hours],
                       "mrms_in": mrms, "nwm_aa_cms": aa})
    write_parquet_to_s3(df, bucket, key)
    log.info("Cached series -> s3://%s/%s", bucket, key)
    return mrms, aa


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(bridge, comid, hours, mrms_in, aa_cms, a14_depths, lp3, out_png):
    t = [h.to_pydatetime() for h in hours]
    fig, (axp, axf) = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True,
                                   gridspec_kw={"hspace": 0.13})

    # ── Top: MRMS hourly precip + Atlas-14 1-h ARI line ───────────────────────
    width = (mdates.date2num(t[1]) - mdates.date2num(t[0])) * 0.9 if len(t) > 1 else 0.03
    axp.bar(t, np.nan_to_num(mrms_in), width=width, color=PRECIP_BAR_C,
            alpha=0.75, label="MRMS 1-h QPE")
    peak_p = float(np.nanmax(mrms_in)) if np.isfinite(mrms_in).any() else 0.0
    kpk = int(np.nanargmax(mrms_in)) if np.isfinite(mrms_in).any() else 0
    top_ref = 0.0
    if a14_depths and peak_p > 0:
        rps = np.array(sorted(a14_depths)); dv = np.array([a14_depths[r] for r in rps])
        ari_p, cap_p = ari_from_value(peak_p, rps, dv)
        rp_p = nearest_standard_rp(ari_p, list(rps)) if np.isfinite(ari_p) else int(rps[0])
        top_ref = a14_depths[rp_p]
        axp.axhline(top_ref, color=PRECIP_LINE_C, ls="--", lw=2.0,
                    label=f"Atlas-14 P{rp_p} · 1-h = {top_ref:.2f} in")
        axp.plot([t[kpk]], [peak_p], "v", color=PRECIP_LINE_C, ms=11, zorder=6)
        cmp = "≥" if cap_p and peak_p >= dv[-1] else ("≤" if cap_p else "≈")
        axp.text(0.015, 0.95,
                 f"Peak 1-h precip = {peak_p:.2f} in\nAtlas-14 ARI {cmp} {ari_p:.0f} yr  ->  P{rp_p}",
                 transform=axp.transAxes, va="top", ha="left", fontsize=11,
                 fontweight="bold", color=PRECIP_LINE_C,
                 bbox=dict(boxstyle="round", fc="white", ec=PRECIP_LINE_C, alpha=0.9))
    axp.set_ylabel("MRMS 1-h precip (in)", fontsize=12)
    axp.set_ylim(0, max(peak_p, top_ref) * 1.35 if max(peak_p, top_ref) > 0 else 0.3)
    axp.grid(axis="y", ls=":", alpha=0.4)
    axp.legend(loc="upper right", fontsize=10, framealpha=0.92)
    sc = "SCOUR-CRITICAL" if bridge["scour"] else "bridge"
    axp.set_title(f"Bridge {bridge['asset']} ({sc}) · COMID {comid} · "
                  f"event {hours[0]:%Y-%m-%d}–{hours[-1]:%m-%d} UTC", fontsize=12.5)

    # ── Bottom: NWM A&A streamflow + retrospective open-loop LP3 flow-ARI line ─
    axf.plot(t, aa_cms, color=FLOW_LINE_C, lw=2.2, label="NWM A&A (with DA)")
    peak_q = float(np.nanmax(aa_cms)) if np.isfinite(aa_cms).any() else 0.0
    kqk = int(np.nanargmax(aa_cms)) if np.isfinite(aa_cms).any() else 0
    if lp3 and peak_q > 0:
        rps = np.array(FLOW_RPS)
        q_cms = np.array([lp3["q_cfs"][r] / CFS_PER_CMS for r in FLOW_RPS])
        ari_q, cap_q = ari_from_value(peak_q, rps, q_cms)
        rp_q = nearest_standard_rp(ari_q, FLOW_RPS) if np.isfinite(ari_q) else FLOW_RPS[0]
        qline = lp3["q_cfs"][rp_q] / CFS_PER_CMS
        axf.axhline(qline, color=FLOW_ARI_C, ls="--", lw=2.0,
                    label=f"Retro open-loop LP3 Q{rp_q} = {qline:,.0f} m³/s")
        axf.plot([t[kqk]], [peak_q], "v", color=FLOW_ARI_C, ms=11, zorder=6)
        cmp = "≥" if cap_q and peak_q >= q_cms[-1] else ("≤" if cap_q else "≈")
        axf.text(0.015, 0.95,
                 f"Peak A&A flow = {peak_q:,.0f} m³/s\nRetro LP3 ARI {cmp} {ari_q:.0f} yr  ->  Q{rp_q}",
                 transform=axf.transAxes, va="top", ha="left", fontsize=11,
                 fontweight="bold", color=FLOW_ARI_C,
                 bbox=dict(boxstyle="round", fc="white", ec=FLOW_ARI_C, alpha=0.9))
        axf.set_ylim(0, max(peak_q, qline) * 1.35)
    axf.set_ylabel("NWM streamflow (m³/s)", fontsize=12)
    axf.set_xlabel("Time (UTC)", fontsize=12)
    axf.grid(axis="y", ls=":", alpha=0.4)
    axf.legend(loc="upper right", fontsize=10, framealpha=0.92)
    if lp3:
        axf.text(0.015, 0.03,
                 f"LP3 on NWM Retrospective v3.0 open-loop annual maxima "
                 f"({lp3['wy_start']}–{lp3['wy_end']}, {lp3['n_years']} WY, B17C {lp3['method']})",
                 transform=axf.transAxes, va="bottom", ha="left", fontsize=8, color="0.35")

    axf.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    axf.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_png)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--event-date", default=DEFAULT_EVENT.isoformat(),
                    help="YYYY-MM-DD (default 2026-06-09)")
    ap.add_argument("--comid", type=int, default=None,
                    help="override the NLDI-resolved crossing COMID")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore all caches (re-download series + re-fit LP3)")
    ap.add_argument("--refresh-series", action="store_true",
                    help="rebuild only the MRMS+A&A series cache (keeps the LP3 fit)")
    ap.add_argument("--refresh-lp3", action="store_true",
                    help="re-fit only the retrospective LP3 (re-opens the Zarr; slow)")
    args = ap.parse_args()
    event = date.fromisoformat(args.event_date)
    refresh_series = args.refresh or args.refresh_series
    refresh_lp3    = args.refresh or args.refresh_lp3

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    m04c = _load("nwm_lp3_04c", "04c_nwm_regression_flows.py")
    b04b = m04c.b04b
    fs = _anon_fs()

    bridge = load_bridge(bucket, prefix, b04b, args.asset)
    log.info("Bridge %s at (%.5f, %.5f)  scour_critical=%s",
             bridge["asset"], bridge["lat"], bridge["lon"], bridge["scour"])

    comid = args.comid or resolve_comid(bridge["lat"], bridge["lon"])
    log.info("Crossing COMID: %d", comid)

    start = pd.Timestamp(event, tz="UTC") - pd.Timedelta(days=WIN_DAYS_BEFORE)
    end   = pd.Timestamp(event, tz="UTC") + pd.Timedelta(days=WIN_DAYS_AFTER)
    hours = list(pd.date_range(start, end, freq="1h", inclusive="left"))
    log.info("Event window: %s → %s (%d hourly steps)",
             start, end, len(hours))

    mrms_in, aa_cms = load_or_build_series(bucket, prefix, b04b, fs, hours,
                                           bridge["lat"], bridge["lon"], comid,
                                           args.asset, event, refresh_series)
    a14 = atlas14_1h_depths(bucket, prefix, b04b, bridge["lat"], bridge["lon"])
    log.info("Atlas-14 1-h depths (in): %s",
             {r: round(v, 2) for r, v in sorted(a14.items())})
    lp3 = retro_lp3_flows(bucket, prefix, m04c, comid, bridge["lat"], refresh_lp3)
    if lp3:
        log.info("Retro LP3 Q (cfs): %s | %d WY %d-%d",
                 {r: round(v) for r, v in lp3["q_cfs"].items()},
                 lp3["n_years"], lp3["wy_start"], lp3["wy_end"])

    out_png = os.path.join(OUT_DIR, f"bridge_{_safe(args.asset)}_{event:%Y%m%d}.png")
    make_figure(bridge, comid, hours, mrms_in, aa_cms, a14, lp3, out_png)

    with open(out_png, "rb") as f:
        s3key = (f"{prefix}events/bridge_{_safe(args.asset)}_{event:%Y%m%d}/"
                 f"bridge_{_safe(args.asset)}_{event:%Y%m%d}.png")
        write_bytes_to_s3(f.read(), bucket, s3key)
    log.info("Uploaded s3://%s/%s", bucket, s3key)
    log.info("Done.")


if __name__ == "__main__":
    main()
