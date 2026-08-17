"""e00 — assemble the episode's alert history and freeze the zoom-region catalog.

Alert history comes from two places, because the digest mechanism landed
mid-episode:
  * Aug 12 predates monitor/alerts/pending/, so that run's alert_state snapshot
    was archived to episode/aug12_alert_state.parquet.
  * Aug 13 onward: one pending/{hour}.parquet per firing run.

Writes  episode/episode_events.parquet   one row per (bridge, trigger, run)
        episode/regions.json (in-repo)   the FIXED zoom catalog

The catalog is frozen to a file on purpose: every downstream product reads it,
so a region keeps identical extent across days and across reruns. Regenerate
only when the episode definition itself changes.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from common import (DAYS, REGIONS_JSON, TZ, bucket, derive_regions, ep_key,
                    load_config)
from monitor_common import config
from monitor_common.s3io import list_keys, read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e00")

COLS = ["bridge_id", "trigger_type", "severity_rp", "valid_hour", "map_class"]


def collect_events() -> pd.DataFrame:
    frames = []
    try:                                    # the Aug 12 run, archived separately
        a = read_parquet(bucket(), ep_key("aug12_alert_state.parquet"))
        a = a.rename(columns={"last_severity_rp": "severity_rp"})
        if "valid_hour" not in a.columns:
            a["valid_hour"] = pd.Timestamp("2026-08-12 15:00", tz="UTC")
        a["map_class"] = np.where(a["trigger_type"] == "precip", "precip", "unknown")
        frames.append(a[COLS])
        log.info("Aug 12 snapshot: %d events", len(a))
    except Exception as e:  # noqa: BLE001
        log.warning("No Aug 12 snapshot (%s) — that day will be missing.", e)

    k = config.keys()
    keys = [x for x in list_keys(k["bucket"], k["pending"]) if x.endswith(".parquet")]
    log.info("pending/ runs: %d", len(keys))
    for key in sorted(keys):
        d = read_parquet(k["bucket"], key)
        if d.empty:
            continue
        d["valid_hour"] = pd.to_datetime(d["valid_hour"], utc=True)
        for c in COLS:
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d[COLS])

    ev = pd.concat(frames, ignore_index=True)
    ev["valid_hour"] = pd.to_datetime(ev["valid_hour"], utc=True)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-mi", type=float, default=6.0)
    ap.add_argument("--min-bridges", type=int, default=5)
    args = ap.parse_args()

    ev = collect_events()
    cfg = load_config().set_index("bridge_id")
    for c in ("lat", "lon", "comid"):
        ev[c] = ev["bridge_id"].map(cfg[c])
    ev["scour"] = ev["bridge_id"].map(cfg[config.SCOUR_COL]).fillna(False).astype(bool)
    ev["asset"] = ev["bridge_id"].map(cfg[config.ASSET_COL]).fillna(ev["bridge_id"])
    ev["day"] = ev["valid_hour"].dt.tz_convert(TZ).dt.date.astype(str)
    ev = ev[ev["day"].isin(DAYS)].dropna(subset=["lat", "lon"]).reset_index(drop=True)

    log.info("Episode: %d events, %d bridges, days %s",
             len(ev), ev["bridge_id"].nunique(), sorted(ev["day"].unique()))
    per_day = ev.groupby("day").agg(events=("bridge_id", "size"),
                                    bridges=("bridge_id", "nunique"))
    log.info("per day:\n%s", per_day.to_string())

    write_parquet(ev, bucket(), ep_key("episode_events.parquet"))

    regions = derive_regions(ev, link_mi=args.link_mi, min_bridges=args.min_bridges)
    blob = json.dumps(regions, indent=1)
    REGIONS_JSON.write_text(blob)
    from monitor_common.s3io import write_bytes
    write_bytes(blob.encode(), bucket(), ep_key("regions.json"),
                content_type="application/json")
    log.info("Froze %d regions -> %s (+ S3 mirror)", len(regions), REGIONS_JSON)
    covered = 0
    for rid, r in regions.items():
        covered += r["bridges"]
        log.info("  %-4s %3d bridges  %2dx%2d mi  centroid %s  days %s",
                 rid, r["bridges"], r["span_mi"][0], r["span_mi"][1],
                 r["centroid"], ",".join(sorted(r["per_day"])))
    log.info("Regions cover %d/%d alerting bridges", covered, ev["bridge_id"].nunique())


if __name__ == "__main__":
    main()
