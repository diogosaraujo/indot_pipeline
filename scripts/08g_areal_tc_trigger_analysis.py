#!/usr/bin/env python3
"""08g_areal_tc_trigger_analysis.py

Watershed-mean (AREAL) version of the Tc-fixed event-overlap confusion matrix.
Identical to 08c except the precipitation trigger uses the MRMS WATERSHED-MEAN
series (06/06b) compared against the AREAL precipitation-frequency DDF from 07a
(mrms/<product>/areal_precip_frequency.parquet) — instead of nearest-pixel MRMS
vs Atlas 14.

It reuses 08c.analyse_station_tc UNCHANGED: the areal DDF carries the same schema
as Atlas 14 (site_no, duration_hr, return_period_yr, depth_in), so it drops
straight into depth_at_duration (log-log interp to each station's Kirpich Tc).

Universe = valid Q ∩ Kirpich Tc ∩ areal DDF ∩ streamflow ∩ (flow∩watershed window).
Sweeps PRECIP_RPS × FLOW_RPS at D = round(Tc). NOTE: the areal DDF has no P1
(AMS-GEV is undefined below T=2), so precip_rp=1 is silently skipped.

Output: analysis/event_confusion_matrix_tc_areal.parquet   (schema matches 08c;
        source tag = 'watershed')
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pandas as pd

from utils import load_config, write_parquet_to_s3

# Reuse 08c (Tc analyzer) and, through it, the base 08 loaders/helpers (c.m).
_spec = importlib.util.spec_from_file_location(
    "tc08c", Path(__file__).with_name("08c_tc_trigger_analysis.py"))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
m = c.m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08g_areal_tc")

OUTPUT_KEY = "analysis/event_confusion_matrix_tc_areal.parquet"
SOURCE     = "watershed"


def load_watershed_precip(bucket, prefix, product) -> pd.DataFrame:
    df = m._read_parquet_s3(bucket, f"{prefix}mrms/{product}/watershed_mean.parquet")
    df["site_no"] = df["site_no"].astype(str)
    df = df.rename(columns={"value_mean": "precip_in"})
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df[["datetime_utc", "site_no", "precip_in"]]


def load_areal_ddf(bucket, prefix, product) -> pd.DataFrame:
    df = m._read_parquet_s3(bucket, f"{prefix}mrms/{product}/areal_precip_frequency.parquet")
    df["site_no"] = df["site_no"].astype(str)
    return df


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    log.info("Loading inputs (areal DDF, flow stats, streamflow, Tc)...")
    areal_ddf  = load_areal_ddf(bucket, prefix, product_key)
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
                          & set(areal_ddf["site_no"]) & set(streamflow["site_no"]))
    log.info("Universe (valid Q ∩ Tc ∩ areal DDF ∩ streamflow): %d", len(stations_all))

    flow_start = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end   = streamflow.groupby("site_no")["datetime_utc"].max()

    log.info("Loading MRMS watershed-mean precipitation...")
    wshed   = load_watershed_precip(bucket, prefix, product_key)
    w_start = wshed.groupby("site_no")["datetime_utc"].min()
    w_end   = wshed.groupby("site_no")["datetime_utc"].max()

    window_by_site: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in stations_all:
        fs_, fe_ = flow_start.get(s, pd.NaT), flow_end.get(s, pd.NaT)
        ps_, pe_ = w_start.get(s, pd.NaT), w_end.get(s, pd.NaT)
        if any(pd.isna(x) for x in (fs_, fe_, ps_, pe_)):
            continue
        ws, we = max(fs_, ps_), min(fe_, pe_)
        if ws < we:
            window_by_site[s] = (ws, we)
    sites_win = [s for s in stations_all if s in window_by_site]
    log.info("Gauges with a valid flow∩watershed window: %d / %d", len(sites_win), len(stations_all))

    existing, complete_keys, stored_end = m.load_existing(
        bucket, prefix, OUTPUT_KEY, c.TC_COMBINATIONS)
    all_records: list[dict] = []

    def _resume_skip(site_no, source, we):
        if (site_no, source) in complete_keys:
            if stored_end.get((site_no, source), pd.NaT) >= we:
                return True
            complete_keys.discard((site_no, source))
        return False

    n = len(sites_win)
    for i, site_no in enumerate(sites_win, 1):
        ws, we = window_by_site[site_no]
        if _resume_skip(site_no, SOURCE, we):
            log.info("[%s][%d/%d] %s: complete, skipping", SOURCE, i, n, site_no)
            continue
        precip_site = wshed[wshed["site_no"] == site_no].set_index("datetime_utc")["precip_in"].sort_index()
        flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
        ddf_site    = areal_ddf[areal_ddf["site_no"] == site_no]
        fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
        if precip_site.empty or flow_site.empty or ddf_site.empty or fs_rows.empty:
            log.warning("[%s][%d/%d] %s: missing data, skipping", SOURCE, i, n, site_no)
            continue
        recs = c.analyse_station_tc(site_no, clusters.get(site_no, -1), precip_site, flow_site,
                                    ddf_site, fs_rows.iloc[0], SOURCE, ws, we,
                                    tc_by_site[site_no], precip_agg="mrms")
        all_records.extend(recs)
        log.info("[%s][%d/%d] %s: Tc=%dh, %d combos", SOURCE, i, n, site_no,
                 max(1, round(tc_by_site[site_no])), len(recs))

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
