"""04c_nwm_regression_flows.py — LP3 flood frequency on NWM Retrospective v3.0

Fits the SAME Bulletin 17C Log-Pearson III workflow as 04b_regression_flows.py,
but to the annual-maximum streamflow series derived from the National Water Model
Retrospective v3.0 (1979-2023) at the nearest-reach COMID of each of the 106 USGS
gauges that survived the 08c funnel (the set station_funnel.py labels "ACTUALLY
USED (08c output)").

Why NWM instead of the gauge record
───────────────────────────────────
The observational records are short (USGS IV starts ~2002, MRMS ~2020), so a
short-record flood-frequency fit is noisy.  The NWM retrospective is a continuous,
gap-free 44-year model reanalysis — a long, homogeneous basis for Q10/Q25/Q50/Q100
at every analysis reach, independent of how long the USGS gauge has been running.

Pipeline
────────
  1. Station universe = distinct site_no in
       analysis/event_confusion_matrix_tc.parquet          (the 106 gauges in 08c).
  2. COMID per gauge from nwm/comid_locations.parquet       (NLDI nearest reach, 10).
  3. NWM Retrospective v3.0 hourly streamflow at each COMID over the FULL
     1979-2023 period, re-extracted from the public AWS Zarr store via script 10's
     extract_retrospective (bypasses the USGS-period clipping the stored
     nwm/retrospective.parquet was built with).  Cached to
     nwm/retrospective_full.parquet so re-runs skip the ~10-30 min Zarr read.
  4. Water-year (Oct-Sep) annual maxima, m³/s → cfs.  Water years with < 90 % hourly
     coverage — i.e. the partial boundary years at each end of the record — are
     dropped so a clipped year cannot understate the annual maximum.
  5. Bulletin 17C LP3 fit (Grubbs-Beck low-outlier test → EMA/MOM → weighted
     regional skew) reused verbatim from 04b so the two methods stay identical.

No Rule-B regulation filter is applied here: the 106 gauges are already the
non-regulated, LP3-fitted set from 04b/08c, and NWM annual maxima carry no USGS
peak qualification codes to test.

Outputs (kept SEPARATE from the USGS 04b table so the flow-stats 08c depends on
are never overwritten)
──────────────────────
    s3://<bucket>/<prefix>flow_stats/nwm_per_gauge_flow_stats.parquet
    s3://<bucket>/<prefix>analysis/nwm_lp3_frequency_curves/<site>_nwm_lp3.png
    s3://<bucket>/<prefix>analysis/nwm_lp3_frequency_curves/nwm_lp3_summary.csv
    s3://<bucket>/<prefix>nwm/retrospective_full.parquet   (extraction cache)

Memory note: the full 44-yr hourly record for ~106 COMIDs is ~40 M rows; the
extraction loads a ~1 GB Zarr subset and the cached frame is a few GB in pandas.
Run on a machine with ≥ 16 GB RAM (the pipeline EC2 host is fine).

Usage:
    python scripts/04c_nwm_regression_flows.py [--refresh] [--no-plots]
        --refresh   force re-extraction of the full retrospective (ignore cache)
        --no-plots  skip the per-station LP3 frequency-curve PNGs
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config, s3_object_exists, write_parquet_to_s3

# ── Reuse 04b's Bulletin 17C LP3 machinery verbatim (single-sourced) ──────────
# Same importlib pattern 08c uses to reuse 08 — the flood-frequency code lives in
# exactly one place so 04b and 04c can never drift apart.
_spec = importlib.util.spec_from_file_location(
    "regression_flows_04b", Path(__file__).with_name("04b_regression_flows.py"))
b04b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b04b)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("04c_nwm_lp3")

# ── Configuration ─────────────────────────────────────────────────────────────
TC_KEY          = "analysis/event_confusion_matrix_tc.parquet"   # 08c output → 106 stations
COMID_KEY       = "nwm/comid_locations.parquet"
INV_KEY         = "stations/indiana_streamflow_sites.parquet"
RETRO_FULL_KEY  = "nwm/retrospective_full.parquet"               # full-record cache
OUT_KEY         = "flow_stats/nwm_per_gauge_flow_stats.parquet"
S3_PLOTS_PREFIX = "analysis/nwm_lp3_frequency_curves/"
SUMMARY_KEY     = S3_PLOTS_PREFIX + "nwm_lp3_summary.csv"

MIN_YEARS       = b04b.MIN_YEARS          # ≥ 10 clean water-year maxima for an LP3 fit
CFS_PER_CMS     = 35.3146667              # 1 m³/s = 35.3146667 ft³/s

# Water-year coverage: NWM retrospective is hourly, so a complete water year holds
# ~8760 rows.  Requiring ≥ 90 % of them drops only the partial boundary years at
# each end of the record (a clipped year could otherwise understate its maximum).
HOURS_PER_WY    = 8760
MIN_WY_COVERAGE = 0.90
MIN_WY_HOURS    = int(MIN_WY_COVERAGE * HOURS_PER_WY)   # 7884


# ── Lazy import of script 10 (heavy Zarr/HDF5 deps — only when re-extracting) ──

def _load_nwm10():
    """Import 10_download_nwm.py on demand (pulls in s3fs/xarray/h5py)."""
    spec = importlib.util.spec_from_file_location(
        "download_nwm_10", Path(__file__).with_name("10_download_nwm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Full-record retrospective (cache or extract) ──────────────────────────────

def get_full_retrospective(
    bucket: str, prefix: str, comid_table: pd.DataFrame, refresh: bool
) -> pd.DataFrame:
    """Return hourly NWM retrospective streamflow for the analysis COMIDs over the
    FULL 1979-2023 period, reading the cache when present.

    Columns: site_no, comid, datetime_utc (tz-aware UTC), streamflow_cms.
    """
    cache_key = f"{prefix}{RETRO_FULL_KEY}"

    if not refresh and s3_object_exists(bucket, cache_key):
        log.info("Reading cached full retrospective from s3://%s/%s ...", bucket, cache_key)
        retro = b04b._read_parquet_s3(bucket, cache_key)
        retro["site_no"] = retro["site_no"].astype(str)
        retro["datetime_utc"] = pd.to_datetime(retro["datetime_utc"], utc=True)
        want = set(comid_table["site_no"])
        missing = want - set(retro["site_no"].unique())
        if missing:
            log.warning("Cached full retrospective is missing %d requested station(s): "
                        "%s — run with --refresh to rebuild.",
                        len(missing), sorted(missing)[:10])
        return retro

    # ── Full re-extraction from the public Zarr store (script 10) ─────────────
    nwm10 = _load_nwm10()
    station_periods = {
        s: (nwm10.RETRO_START, nwm10.RETRO_END) for s in comid_table["site_no"]
    }
    log.info("Extracting FULL NWM retrospective (%s → %s) for %d COMID(s) — opens the "
             "public Zarr store, may take 10-30 min...",
             nwm10.RETRO_START.date(), nwm10.RETRO_END.date(),
             comid_table["comid"].nunique())

    retro = nwm10.extract_retrospective(
        comid_table[["site_no", "comid"]].copy(), station_periods)
    if retro.empty:
        raise RuntimeError("NWM retrospective extraction returned no data.")

    # Velocity is not needed for annual maxima — drop it to keep the cache lean.
    keep_cols = ["site_no", "comid", "datetime_utc", "streamflow_cms"]
    retro = retro[[c for c in keep_cols if c in retro.columns]].copy()

    write_parquet_to_s3(retro, bucket, cache_key)
    log.info("Cached full retrospective → s3://%s/%s (%d rows, %d stations)",
             bucket, cache_key, len(retro), retro["site_no"].nunique())

    retro["site_no"] = retro["site_no"].astype(str)
    retro["datetime_utc"] = pd.to_datetime(retro["datetime_utc"], utc=True)
    return retro


# ── Annual-maximum series ─────────────────────────────────────────────────────

def annual_max_series(site_df: pd.DataFrame) -> pd.Series:
    """Water-year (Oct-Sep) annual-maximum discharge in cfs from hourly NWM cms.

    Water years with < MIN_WY_HOURS of data (the partial boundary years) are
    dropped.  Returns a Series of peak cfs indexed by water_year, sorted.
    """
    df = site_df.dropna(subset=["streamflow_cms"])
    if df.empty:
        return pd.Series(dtype=float)

    dt   = pd.to_datetime(df["datetime_utc"], utc=True)
    cfs  = df["streamflow_cms"].to_numpy(dtype=float) * CFS_PER_CMS
    # Water year labels by the calendar year in which it ends (matches 04a/04b).
    wy   = dt.dt.year.to_numpy() + (dt.dt.month.to_numpy() >= 10).astype(int)

    g     = pd.DataFrame({"wy": wy, "cfs": cfs})
    grp   = g.groupby("wy")["cfs"]
    ann   = grp.max()
    hours = grp.size()

    ann = ann[hours >= MIN_WY_HOURS]
    ann = ann[ann > 0]
    return ann.sort_index()


# ── Frequency-curve plot (reuse 04b's, relabel for NWM) ───────────────────────

def plot_nwm_curve(site_no, comid, ann, params, q_design, gof) -> plt.Figure:
    fig = b04b.plot_frequency_curve(site_no, ann, params, q_design, gof)
    ax  = fig.axes[0]
    method = params.get("fitting_method", "MOM")
    ax.set_ylabel("Annual-Max Discharge (cfs)")
    ax.set_title(
        f"NWM Retrospective v3.0 — LP3 Frequency Curve\n"
        f"Station {site_no} · COMID {comid} · Bulletin 17C {method}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ── Output-row helpers ────────────────────────────────────────────────────────

def _blank_row(site_no, comid, dist, source, n_years=0, wy_start=None, wy_end=None) -> dict:
    row = {
        "site_no":     site_no,
        "comid":       comid,
        "distance_km": dist,
        "source":      source,
        "n_years":     n_years,
        "wy_start":    wy_start,
        "wy_end":      wy_end,
    }
    for col in b04b.ALL_Q_COLS.values():
        row[col] = np.nan
    return row


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="force re-extraction of the full retrospective (ignore cache)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the per-station LP3 frequency-curve PNGs")
    args = ap.parse_args()

    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    # 1. Station universe — the 106 gauges actually scored in 08c ──────────────
    tc = b04b._read_parquet_s3(bucket, f"{prefix}{TC_KEY}", columns=["site_no"])
    stations = sorted(tc["site_no"].astype(str).unique())
    log.info("08c analysis stations (event_confusion_matrix_tc): %d", len(stations))
    if len(stations) != 106:
        log.warning("Expected 106 stations from 08c, found %d — proceeding with the "
                    "actual set.", len(stations))

    # 2. COMID per gauge (NLDI nearest reach, from script 10) ──────────────────
    cl = b04b._read_parquet_s3(bucket, f"{prefix}{COMID_KEY}")
    cl["site_no"] = cl["site_no"].astype(str)
    cl["comid"]   = cl["comid"].astype(int)
    cl106 = cl[cl["site_no"].isin(stations)].drop_duplicates("site_no").copy()
    comid_map = {
        r.site_no: (int(r.comid),
                    round(float(r.distance_km), 4) if pd.notna(r.distance_km) else None)
        for r in cl106.itertuples(index=False)
    }
    no_comid = [s for s in stations if s not in comid_map]
    log.info("Stations with a resolved COMID: %d / %d%s",
             len(comid_map), len(stations),
             f"  (no COMID: {no_comid})" if no_comid else "")

    # 3. Full-record NWM retrospective (cache or re-extract) ───────────────────
    retro = get_full_retrospective(bucket, prefix, cl106, refresh=args.refresh)
    retro_by_site = {s: g for s, g in retro.groupby("site_no")}

    # 4. Station coordinates for the B17C regional-skew lookup (same as 04b) ────
    inv = b04b._read_parquet_s3(bucket, f"{prefix}{INV_KEY}",
                                columns=["site_no", "dec_lat_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    lat_map = inv.set_index("site_no")["dec_lat_va"].to_dict()

    # 5. Per-station LP3 fit on the NWM annual maxima ──────────────────────────
    out_rows: list[dict] = []
    summary_rows: list[dict] = []
    counts = {"fit": 0, "short": 0, "no_data": 0}
    n_total = len(stations)

    for i, site_no in enumerate(stations, 1):
        comid, dist = comid_map.get(site_no, (None, None))
        site_df = retro_by_site.get(site_no)

        if comid is None or site_df is None or site_df.empty:
            out_rows.append(_blank_row(site_no, comid, dist, "no_nwm_data"))
            summary_rows.append({"site_no": site_no, "comid": comid, "status": "no_nwm_data"})
            counts["no_data"] += 1
            log.info("[%d/%d] %s — no NWM retrospective data", i, n_total, site_no)
            continue

        ann = annual_max_series(site_df)
        n_years = len(ann)

        if n_years < MIN_YEARS:
            out_rows.append(_blank_row(site_no, comid, dist, "insufficient_record",
                                       n_years=n_years,
                                       wy_start=int(ann.index.min()) if n_years else None,
                                       wy_end=int(ann.index.max()) if n_years else None))
            summary_rows.append({"site_no": site_no, "comid": comid,
                                 "status": "insufficient_record", "n_years": n_years})
            counts["short"] += 1
            log.info("[%d/%d] %s — only %d water years (< %d), insufficient",
                     i, n_total, site_no, n_years, MIN_YEARS)
            continue

        wy_start, wy_end = int(ann.index.min()), int(ann.index.max())
        lat    = float(lat_map.get(site_no, 39.8))     # Indiana centroid fallback
        log_q  = np.log10(ann.values)
        params = b04b.fit_lp3(log_q, lat)
        gof    = b04b.compute_gof(log_q, params)
        log.info("[%d/%d] %s — COMID %d — %d WY (%d–%d) — %s",
                 i, n_total, site_no, comid, n_years, wy_start, wy_end,
                 params.get("fitting_method", "MOM"))

        # Populate every standard return-period column (LP3 gives any quantile).
        row = _blank_row(site_no, comid, dist, "nwm_retro_lp3",
                         n_years=n_years, wy_start=wy_start, wy_end=wy_end)
        q_design: dict[int, float] = {}
        for rp in b04b.ALL_RETURN_PERIODS:
            q_t = b04b.lp3_quantile(rp, params["mean_log"], params["std_log"], params["skew"])
            row[b04b.ALL_Q_COLS[rp]] = q_t
            if rp in b04b.RETURN_PERIODS:
                q_design[rp] = round(q_t, 1)
        out_rows.append(row)
        counts["fit"] += 1

        summary_rows.append({
            "site_no":         site_no,
            "comid":           comid,
            "distance_km":     dist,
            "n_years":         n_years,
            "wy_start":        wy_start,
            "wy_end":          wy_end,
            "status":          "nwm_retro_lp3",
            "fitting_method":  params.get("fitting_method", "MOM"),
            "n_censored":      params.get("n_censored", 0),
            "threshold_cfs":   (round(10 ** params["threshold_log"], 1)
                                if params.get("threshold_log") is not None else None),
            "mean_log":        round(params["mean_log"], 4),
            "std_log":         round(params["std_log"], 4),
            "skew_at_site":    round(params["skew_at_site"], 4),
            "skew_regional":   round(params["skew_regional"], 4),
            "weight_at_site":  params["weight_at_site"],
            "weight_regional": params["weight_regional"],
            "skew_weighted":   round(params["skew"], 4),
            **{f"Q{rp}": q_design.get(rp) for rp in b04b.RETURN_PERIODS},
            **gof,
        })

        if not args.no_plots:
            try:
                fig = plot_nwm_curve(site_no, comid, ann, params, q_design, gof)
                b04b._upload_png(fig, bucket, f"{prefix}{S3_PLOTS_PREFIX}{site_no}_nwm_lp3.png")
                plt.close(fig)
            except Exception as exc:                       # noqa: BLE001
                log.warning("    Plot failed: %s", exc)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out = pd.DataFrame(out_rows)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUT_KEY}")
    log.info("Wrote %s%s (%d stations)", prefix, OUT_KEY, len(out))

    b04b._upload_csv(pd.DataFrame(summary_rows), bucket, f"{prefix}{SUMMARY_KEY}")

    log.info("NWM LP3 fitted: %d  |  insufficient (< %d WY): %d  |  "
             "no NWM data/COMID: %d  |  total: %d",
             counts["fit"], MIN_YEARS, counts["short"], counts["no_data"], n_total)


if __name__ == "__main__":
    main()
