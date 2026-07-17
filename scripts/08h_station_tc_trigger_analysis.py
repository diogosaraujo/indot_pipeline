#!/usr/bin/env python3
"""08h_station_tc_trigger_analysis.py

Station-gauge version of the Tc-fixed event-overlap confusion matrix. Identical
to 08c except the precipitation trigger uses a WEATHER-STATION gauge (NOAA
ISD/LCD + GHCNh) compared against Atlas 14 POINT DDF (station gauges are point
measurements, so the point estimates apply — same frequency source as 08c; only
the precip series changes).

Station pairing (from 08d, NOT the 'covering' pairing of 08/08c): each gauge is
paired to its geographically NEAREST qualifying precip station, walking outward
(2nd, 3rd, …) until one carries a real hourly value for >= MIN_PERIOD_COVERAGE of
the reference window. **Distance is not gated**, so every gauge gets a station.
The analysis window is that reference period, [max(flow_start, 2002), flow_end].
08d's loader also handles GHCNh's mm→inch conversion and per-hour-max resampling.

Universe = valid Q ∩ Kirpich Tc ∩ Atlas14 ∩ streamflow, each paired to a station.
Sweeps PRECIP_RPS × FLOW_RPS at D = round(Tc).

Output: analysis/event_confusion_matrix_tc_station.parquet  (08c schema + dist_mi
        and precip_station_id; source tag = 'station_nearest')
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pandas as pd

from utils import load_config, write_parquet_to_s3

# Reuse 08c (Tc analyzer + base-08 loaders via c.m) and 08d (station pairing).
def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  Path(__file__).with_name(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

c = _load("08c_tc_trigger_analysis.py")
d = _load("08d_indot_trigger_analysis.py")
m = c.m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08h_station_tc")

OUTPUT_KEY   = "analysis/event_confusion_matrix_tc_station.parquet"
SOURCE       = "station_nearest"
INV_KEY      = "stations/indiana_streamflow_sites.parquet"
UNIVERSE_KEY = "analysis/event_confusion_matrix_tc.parquet"   # 08c's retained 106 gauges


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    log.info("Loading inputs (station Atlas 14, flow stats, streamflow, Tc)...")
    # Atlas 14 keyed by PRECIP STATION (07c): the station's rain is judged against
    # its OWN climatology, not the streamgage's (climatology-mismatch fix).
    atlas14_st = m._read_parquet_s3(bucket, f"{prefix}atlas14/precipitation_frequency_stations.parquet")
    atlas14_st["station_id"] = atlas14_st["station_id"].astype(str)
    flow_stats = m.load_flow_stats(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)
    tc_by_site = c.load_tc(bucket, prefix)
    try:
        clusters = m.load_clusters(bucket, prefix)
    except Exception:
        clusters = {}

    # Pin to 08c's retained 106 gauges (flow∩MRMS window already applied there).
    retained = set(m._read_parquet_s3(bucket, f"{prefix}{UNIVERSE_KEY}",
                                      ["site_no"])["site_no"].astype(str))
    q_cols = [f"Q{rp}" for rp in c.FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q  = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])
    stations_all = sorted(retained & has_q & set(tc_by_site) & set(streamflow["site_no"]))
    log.info("Universe (08c-retained 106 ∩ valid Q ∩ Tc ∩ streamflow): %d", len(stations_all))

    flow_start = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end   = streamflow.groupby("site_no")["datetime_utc"].max()

    # Gauge coordinates for the nearest-station search.
    inv = m._read_parquet_s3(bucket, f"{prefix}{INV_KEY}",
                             ["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    latlon = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).set_index("site_no")

    # Qualifying precip stations + their per-hour-max hourly series (08d loader).
    log.info("Loading + qualifying precip stations (ISD + GHCNh)...")
    qual, hourly_by_sid = d.load_and_qualify(bucket, prefix)

    existing, complete_keys, stored_end = m.load_existing(
        bucket, prefix, OUTPUT_KEY, c.TC_COMBINATIONS)
    all_records: list[dict] = []

    def _resume_skip(site_no, source, we):
        if (site_no, source) in complete_keys:
            if stored_end.get((site_no, source), pd.NaT) >= we:
                return True
            complete_keys.discard((site_no, source))
        return False

    n = len(stations_all)
    n_no_coord = n_no_station = 0
    for i, site_no in enumerate(stations_all, 1):
        if site_no not in latlon.index:
            n_no_coord += 1
            continue
        fs_, fe_ = flow_start.get(site_no, pd.NaT), flow_end.get(site_no, pd.NaT)
        if pd.isna(fs_) or pd.isna(fe_):
            continue
        # Window end = flow_end (08d's reference window), so resume can be checked
        # before the nearest-station search.
        if _resume_skip(site_no, SOURCE, fe_):
            log.info("[%s][%d/%d] %s: complete, skipping", SOURCE, i, n, site_no)
            continue
        lat = float(latlon.at[site_no, "dec_lat_va"])
        lon = float(latlon.at[site_no, "dec_long_va"])

        res = d.assign_station(lat, lon, fs_, fe_, qual, hourly_by_sid)
        if res is None:                                   # no qualifying station reached coverage
            n_no_station += 1
            log.warning("[%s][%d/%d] %s: no qualifying precip station", SOURCE, i, n, site_no)
            continue
        sid, dist_mi, ws, we, _grid, miss = res

        precip_site = hourly_by_sid[sid]                  # raw hourly-max (unfilled)
        flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
        a14_site    = atlas14_st[atlas14_st["station_id"] == sid]   # station's own climatology
        fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
        if flow_site.empty or a14_site.empty or fs_rows.empty:
            log.warning("[%s][%d/%d] %s: no station Atlas 14 for %s, skipping", SOURCE, i, n, site_no, sid)
            continue
        recs = c.analyse_station_tc(site_no, clusters.get(site_no, -1), precip_site, flow_site,
                                    a14_site, fs_rows.iloc[0], SOURCE, ws, we,
                                    tc_by_site[site_no], precip_agg="max")
        for r in recs:                                    # QC tags for the ungated pairing
            r["precip_station_id"] = sid
            r["dist_mi"] = round(dist_mi, 2)
        all_records.extend(recs)
        log.info("[%s][%d/%d] %s: Tc=%dh, station=%s (%.1f mi), %d combos",
                 SOURCE, i, n, site_no, max(1, round(tc_by_site[site_no])), sid, dist_mi, len(recs))

    log.info("Paired: %d gauges | no coord: %d | no qualifying station: %d",
             len(stations_all) - n_no_coord - n_no_station, n_no_coord, n_no_station)

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
