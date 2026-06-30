"""08c_tc_trigger_analysis.py

Fixed-duration (Kirpich Tc) version of the event-overlap confusion matrix.

Instead of sweeping accumulation duration × precip return period, the
accumulation duration is FIXED per station at its Kirpich time of concentration,
rounded to the nearest hour (validated against the per-cluster CSI ridge — see
tc_vs_csi_ridge.py).  Only the precipitation threshold (ARI / return period) is
swept, so results can be pooled GLOBALLY (all stations together) rather than per
cluster.

For each station:
    D = round(Tc)  hours                               (MRMS is hourly → exact)
    precip wet hour : trailing D-hour accumulation ≥ Atlas14(D, precip_rp)
                      where Atlas14(D, ·) is log-log interpolated across the
                      published DDF durations to the station's exact D.
    flow   wet hour : hourly streamflow ≥ Q(flow_rp)
Event grouping, ±24 h linking, episodes and TP/FN/FP/TN are IDENTICAL to 08
(the helpers are imported from it, not reimplemented).

Output schema (one row per station × precip_rp × flow_rp):
    site_no, cluster, source, tc_hr, duration_hr, precip_rp_yr, precip_depth_in,
    flow_rp_yr, tp, fp, fn, tn, n_precip_events, n_flow_events,
    pct_precip_missing, precip_agg, common_start, common_end, n_common_hours

Writes:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_tc.parquet

Usage:
    python scripts/08c_tc_trigger_analysis.py
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, write_parquet_to_s3

# Import the digit-prefixed module 08_trigger_analysis.py and reuse its loaders
# and event-overlap helpers (group_wet_events, classify_overlap, window logic …).
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py")
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08c_tc")

PRECIP_RPS = m.PRECIP_RPS                     # ARI thresholds to sweep
FLOW_RPS   = m.FLOW_RPS                       # Q10 / Q50 / Q100
OUTPUT_KEY = "analysis/event_confusion_matrix_tc.parquet"
TC_COMBINATIONS = len(PRECIP_RPS) * len(FLOW_RPS)   # per station (single duration)


# ---------- Tc + DDF-at-Tc ----------

def load_tc(bucket: str, prefix: str) -> dict[str, float]:
    """site_no → Kirpich Tc (hours) from basin characteristics."""
    df = m._read_parquet_s3(
        bucket, f"{prefix}watersheds/basin_characteristics.parquet",
        ["site_no", "tc_hr"])
    df["site_no"] = df["site_no"].astype(str)
    df = df.dropna(subset=["tc_hr"])
    return dict(zip(df["site_no"], df["tc_hr"].astype(float)))


def depth_at_duration(a14_site: pd.DataFrame, precip_rp: int, d_hr: float) -> float:
    """Atlas 14 depth at the exact duration d_hr for one return period, by
    log-log interpolation across the published DDF durations (clamped)."""
    rows = a14_site[a14_site["return_period_yr"] == precip_rp].dropna(subset=["depth_in"])
    if len(rows) < 2:
        return float("nan")
    durs   = rows["duration_hr"].to_numpy(float)
    depths = rows["depth_in"].to_numpy(float)
    return m.loglog_depth_at_rp(durs, depths, float(d_hr))   # generic log-log interp


# ---------- Per-station analysis (single Tc duration) ----------

def analyse_station_tc(
    site_no: str,
    cluster: int,
    precip_hourly: pd.Series,
    flow_hourly: pd.Series,
    atlas14_site: pd.DataFrame,
    flow_stats_row: pd.Series,
    source: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    tc_hr: float,
    precip_agg: str = "mrms",
) -> list[dict]:
    if pd.isna(window_start) or pd.isna(window_end) or window_start >= window_end:
        log.warning("[%s] %s: empty window — skipping", source, site_no)
        return []
    d_tc = max(1, int(round(float(tc_hr))))                  # accumulation hours
    common_start, common_end = window_start, window_end

    grid = pd.date_range(common_start, common_end, freq="1h", tz="UTC")
    precip = precip_hourly.reindex(grid)
    pct_precip_missing = round(100.0 * float(precip.isna().mean()), 2)
    precip = precip.fillna(0.0)
    flow   = flow_hourly.reindex(grid)
    n_common = len(grid)

    # flow-wet events per return period (depend only on Q_rp)
    flow_events_by_rp: dict[int, list] = {}
    for flow_rp in FLOW_RPS:
        q_col = f"Q{flow_rp}"
        if q_col not in flow_stats_row or pd.isna(flow_stats_row[q_col]):
            continue
        wet = (flow >= float(flow_stats_row[q_col])).fillna(False)
        flow_events_by_rp[flow_rp] = m.group_wet_events(wet)
    if not flow_events_by_rp:
        return []

    rolling = precip.rolling(window=d_tc, min_periods=d_tc).sum()

    records: list[dict] = []
    for precip_rp in PRECIP_RPS:
        depth = depth_at_duration(atlas14_site, precip_rp, d_tc)
        if not np.isfinite(depth) or depth <= 0:
            continue
        precip_wet = (rolling >= depth).fillna(False)
        precip_events = m.group_wet_events(precip_wet)
        for flow_rp, flow_events in flow_events_by_rp.items():
            tp, fp, fn, tn = m.classify_overlap(
                precip_events, flow_events, common_start, common_end)
            records.append({
                "site_no":         site_no,
                "cluster":         cluster,
                "source":          source,
                "tc_hr":           round(float(tc_hr), 2),
                "duration_hr":     d_tc,
                "precip_rp_yr":    precip_rp,
                "precip_depth_in": round(float(depth), 4),
                "flow_rp_yr":      flow_rp,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_precip_events": len(precip_events),
                "n_flow_events":   len(flow_events),
                "pct_precip_missing": pct_precip_missing,
                "precip_agg":      precip_agg,
                "common_start":    common_start,
                "common_end":      common_end,
                "n_common_hours":  n_common,
            })
    return records


# ---------- Main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    log.info("Loading inputs...")
    atlas14    = m.load_atlas14(bucket, prefix)
    flow_stats = m.load_flow_stats(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)
    clusters   = m.load_clusters(bucket, prefix)
    tc_by_site = load_tc(bucket, prefix)
    log.info("Stations with a Kirpich Tc: %d", len(tc_by_site))

    q_cols = [f"Q{rp}" for rp in FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q  = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])

    stations_all = sorted(
        set(clusters) & has_q & set(tc_by_site)
        & set(atlas14["site_no"]) & set(streamflow["site_no"])
    )
    log.info("Stations with all inputs (incl. Tc): %d", len(stations_all))

    flow_start_by_site = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end_by_site   = streamflow.groupby("site_no")["datetime_utc"].max()

    # Per-gauge window = streamflow span ∩ MRMS coverage (identical for both sources)
    log.info("Loading MRMS nearest-pixel precipitation...")
    try:
        mrms = m.load_mrms_nearest(bucket, prefix, product_key)
    except Exception as e:
        log.error("Could not load MRMS nearest: %s", e)
        mrms = pd.DataFrame()
    mrms_start = mrms.groupby("site_no")["datetime_utc"].min() if not mrms.empty else {}
    mrms_end   = mrms.groupby("site_no")["datetime_utc"].max() if not mrms.empty else {}

    window_by_site: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in stations_all:
        fs_, fe_ = flow_start_by_site.get(s, pd.NaT), flow_end_by_site.get(s, pd.NaT)
        ms_ = mrms_start.get(s, pd.NaT) if len(mrms_start) else pd.NaT
        me_ = mrms_end.get(s, pd.NaT) if len(mrms_end) else pd.NaT
        if any(pd.isna(x) for x in (fs_, fe_, ms_, me_)):
            continue
        ws, we = max(fs_, ms_), min(fe_, me_)
        if ws < we:
            window_by_site[s] = (ws, we)
    sites_win = [s for s in stations_all if s in window_by_site]
    log.info("Gauges with a valid flow∩MRMS window: %d / %d", len(sites_win), len(stations_all))

    existing, complete_keys, stored_end = m.load_existing(
        bucket, prefix, OUTPUT_KEY, TC_COMBINATIONS)
    all_records: list[dict] = []

    def _resume_skip(site_no: str, source: str, we: pd.Timestamp) -> bool:
        if (site_no, source) in complete_keys:
            if stored_end.get((site_no, source), pd.NaT) >= we:
                return True
            complete_keys.discard((site_no, source))
        return False

    # ── Source 1: nearest MRMS pixel ──────────────────────────────────────────
    if not mrms.empty:
        n = len(sites_win)
        for i, site_no in enumerate(sites_win, 1):
            ws, we = window_by_site[site_no]
            if _resume_skip(site_no, "nearest", we):
                log.info("[nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                continue
            precip_site = mrms[mrms["site_no"] == site_no].set_index("datetime_utc")["precip_in"].sort_index()
            flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
            a14_site    = atlas14[atlas14["site_no"] == site_no]
            fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
            if precip_site.empty or flow_site.empty or a14_site.empty or fs_rows.empty:
                log.warning("[nearest][%d/%d] %s: missing data, skipping", i, n, site_no)
                continue
            recs = analyse_station_tc(site_no, clusters[site_no], precip_site, flow_site,
                                      a14_site, fs_rows.iloc[0], "nearest", ws, we,
                                      tc_by_site[site_no])
            all_records.extend(recs)
            log.info("[nearest][%d/%d] %s: Tc=%dh, %d combos",
                     i, n, site_no, max(1, round(tc_by_site[site_no])), len(recs))

    # ── Source 2: nearest covering ISD/GHCNh station ──────────────────────────
    log.info("Loading station coverage (ISD + GHCNh)...")
    coverage = m.load_station_coverage(bucket, prefix)
    if coverage.empty:
        log.warning("No station precip — skipping station_nearest source.")
    else:
        gauges = m.load_gauges(bucket, prefix)
        assign = m.assign_covering_station(gauges, coverage, window_by_site)
        needed = set(assign.values())
        log.info("Covering precip station for %d / %d gauges; loading %d stations...",
                 len(assign), len(sites_win), len(needed))
        precip_hourly = m.load_precip_for_stations(bucket, prefix, needed)
        if precip_hourly.empty:
            log.warning("No station precip rows — skipping station_nearest.")
        else:
            sites_st = [s for s in sites_win if s in assign]
            n = len(sites_st)
            resamp_cache: dict[str, tuple[pd.Series, str]] = {}
            for i, site_no in enumerate(sites_st, 1):
                ws, we = window_by_site[site_no]
                if _resume_skip(site_no, "station_nearest", we):
                    log.info("[station_nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                    continue
                sid = assign[site_no]
                if sid not in resamp_cache:
                    sub = precip_hourly[precip_hourly["station_id"] == sid]
                    resamp_cache[sid] = (
                        m.resample_station_precip(
                            sub.set_index("datetime_utc")["precip_in"].sort_index())
                        if not sub.empty else (pd.Series(dtype=float), "none"))
                precip_site, agg_method = resamp_cache[sid]
                if precip_site.empty:
                    continue
                flow_site = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
                a14_site  = atlas14[atlas14["site_no"] == site_no]
                fs_rows   = flow_stats[flow_stats["site_no"] == site_no]
                if flow_site.empty or a14_site.empty or fs_rows.empty:
                    continue
                recs = analyse_station_tc(site_no, clusters[site_no], precip_site, flow_site,
                                          a14_site, fs_rows.iloc[0], "station_nearest", ws, we,
                                          tc_by_site[site_no], precip_agg=agg_method)
                all_records.extend(recs)
                log.info("[station_nearest][%d/%d] %s: %d combos (agg=%s)",
                         i, n, site_no, len(recs), agg_method)

    # ── Combine with retained complete pairs and write ────────────────────────
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
