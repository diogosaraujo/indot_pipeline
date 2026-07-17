#!/usr/bin/env python3
"""08h_station_tc_trigger_analysis.py

Station-gauge version of the Tc-fixed event-overlap confusion matrix. Identical
to 08c except the precipitation trigger uses the nearest ISD/GHCNh gauge that
COVERS the analysis window — resampled to hourly with 08's resample_station_precip
(per-hour max, which recovers the single hourly report / largest trailing 1-h
accumulation) — compared against Atlas 14 POINT DDF. Point gauges are point
measurements, so the Atlas 14 point estimates apply directly (same frequency
source as 08c; only the precip series changes).

Common analysis window = flow ∩ MRMS-nearest coverage — IDENTICAL to 08c/08g, so
the point-MRMS, areal-MRMS and station analyses all share one window; the
covering precip station must SPAN it (assign_covering_station).

Universe = valid Q ∩ Kirpich Tc ∩ Atlas14 ∩ streamflow ∩ window ∩ a covering
precip gauge. Sweeps PRECIP_RPS × FLOW_RPS at D = round(Tc).

Output: analysis/event_confusion_matrix_tc_station.parquet  (schema matches 08c;
        source tag = 'station_nearest')
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pandas as pd

from utils import load_config, write_parquet_to_s3

_spec = importlib.util.spec_from_file_location(
    "tc08c", Path(__file__).with_name("08c_tc_trigger_analysis.py"))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
m = c.m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08h_station_tc")

OUTPUT_KEY = "analysis/event_confusion_matrix_tc_station.parquet"
SOURCE     = "station_nearest"


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    log.info("Loading inputs (Atlas 14, flow stats, streamflow, Tc)...")
    atlas14    = m.load_atlas14(bucket, prefix)
    flow_stats = m.load_flow_stats(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)
    tc_by_site = c.load_tc(bucket, prefix)
    try:
        clusters = m.load_clusters(bucket, prefix)
    except Exception:
        clusters = {}

    q_cols = [f"Q{rp}" for rp in c.FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q  = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])
    stations_all = sorted(has_q & set(tc_by_site)
                          & set(atlas14["site_no"]) & set(streamflow["site_no"]))
    log.info("Universe (valid Q ∩ Tc ∩ Atlas14 ∩ streamflow): %d", len(stations_all))

    flow_start = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end   = streamflow.groupby("site_no")["datetime_utc"].max()

    # Common window = flow ∩ MRMS-nearest coverage (matches 08c/08g exactly).
    log.info("Loading MRMS nearest-pixel (window reference only)...")
    mrms = m.load_mrms_nearest(bucket, prefix, product_key)
    mrms_start = mrms.groupby("site_no")["datetime_utc"].min() if not mrms.empty else {}
    mrms_end   = mrms.groupby("site_no")["datetime_utc"].max() if not mrms.empty else {}

    window_by_site: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in stations_all:
        fs_, fe_ = flow_start.get(s, pd.NaT), flow_end.get(s, pd.NaT)
        ms_ = mrms_start.get(s, pd.NaT) if len(mrms_start) else pd.NaT
        me_ = mrms_end.get(s, pd.NaT) if len(mrms_end) else pd.NaT
        if any(pd.isna(x) for x in (fs_, fe_, ms_, me_)):
            continue
        ws, we = max(fs_, ms_), min(fe_, me_)
        if ws < we:
            window_by_site[s] = (ws, we)
    sites_win = [s for s in stations_all if s in window_by_site]
    log.info("Gauges with a valid flow∩MRMS window: %d / %d", len(sites_win), len(stations_all))

    # Assign each gauge the nearest ISD/GHCNh station that COVERS its window.
    log.info("Loading station coverage (ISD + GHCNh)...")
    coverage = m.load_station_coverage(bucket, prefix)
    if coverage.empty:
        log.error("No station precip coverage — nothing to do.")
        return
    gauges = m.load_gauges(bucket, prefix)
    assign = m.assign_covering_station(gauges, coverage, window_by_site)
    sites_st = [s for s in sites_win if s in assign]
    log.info("Covering precip station found for %d / %d gauges", len(sites_st), len(sites_win))
    if not sites_st:
        log.error("No gauge has a covering precip station — nothing to do.")
        return

    needed = set(assign[s] for s in sites_st)
    log.info("Loading precip for %d covering stations...", len(needed))
    precip_hourly = m.load_precip_for_stations(bucket, prefix, needed)
    if precip_hourly.empty:
        log.error("No station precip rows loaded.")
        return

    existing, complete_keys, stored_end = m.load_existing(
        bucket, prefix, OUTPUT_KEY, c.TC_COMBINATIONS)
    all_records: list[dict] = []

    def _resume_skip(site_no, source, we):
        if (site_no, source) in complete_keys:
            if stored_end.get((site_no, source), pd.NaT) >= we:
                return True
            complete_keys.discard((site_no, source))
        return False

    resamp_cache: dict[str, tuple[pd.Series, str]] = {}   # station_id → (hourly, agg)
    n = len(sites_st)
    for i, site_no in enumerate(sites_st, 1):
        ws, we = window_by_site[site_no]
        if _resume_skip(site_no, SOURCE, we):
            log.info("[%s][%d/%d] %s: complete, skipping", SOURCE, i, n, site_no)
            continue
        sid = assign[site_no]
        if sid not in resamp_cache:
            sub = precip_hourly[precip_hourly["station_id"] == sid]
            if sub.empty:
                resamp_cache[sid] = (pd.Series(dtype=float), "none")
            else:
                resamp_cache[sid] = m.resample_station_precip(
                    sub.set_index("datetime_utc")["precip_in"].sort_index())
        precip_site, agg_method = resamp_cache[sid]
        if precip_site.empty:
            log.warning("[%s][%d/%d] %s: no precip, skipping", SOURCE, i, n, site_no)
            continue
        flow_site = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
        a14_site  = atlas14[atlas14["site_no"] == site_no]
        fs_rows   = flow_stats[flow_stats["site_no"] == site_no]
        if flow_site.empty or a14_site.empty or fs_rows.empty:
            continue
        recs = c.analyse_station_tc(site_no, clusters.get(site_no, -1), precip_site, flow_site,
                                    a14_site, fs_rows.iloc[0], SOURCE, ws, we,
                                    tc_by_site[site_no], precip_agg=agg_method)
        all_records.extend(recs)
        log.info("[%s][%d/%d] %s: Tc=%dh, %d combos (agg=%s)", SOURCE, i, n, site_no,
                 max(1, round(tc_by_site[site_no])), len(recs), agg_method)

    parts: list[pd.DataFrame] = []
    if existing is not None and complete_keys:
        kept = existing[existing[["site_no", "source"]].apply(
            lambda r: (r["site_no"], r["source"]) in complete_keys, axis=1)]
        parts.append(kept)
        log.info("Retaining %d rows from previous run", len(kept))
    if all_records:
        parts.append(pd.DataFrame(all_records))
    if not parts:
        log.error("No results produced.")
        return

    out = pd.concat(parts, ignore_index=True)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s%s (%d rows, %d stations)",
             prefix, OUTPUT_KEY, len(out), out["site_no"].nunique())


if __name__ == "__main__":
    main()
