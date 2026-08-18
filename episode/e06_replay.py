"""e06 — replay the monitor over hours it never evaluated, and rebuild the
notification that would have gone out.

The poller is a poller, not a watcher: evaluate() only ever tests the newest
hour, so an hour missed while the function was down is missed permanently even
though the source data still exists. This replays those hours from the episode
store that e01 already fetched.

It deliberately does NOT touch production state:
  * alert_state is seeded from a snapshot and kept IN MEMORY. Replaying into the
    live table would both corrupt it (writing stale last_wet_hour values) and
    silently suppress everything, since a past valid_hour minus a newer
    last_wet_hour is negative and never exceeds the 24-h declustering gap.
  * output goes to episode/replay/, not monitor/alerts/.

Precip caveat: the trailing-Tc windows are filled from the episode store, so
this shows what a monitor with a WARM 48-h buffer would have caught. The live
system on Aug 13 had a partially filled buffer, so its real precip yield would
have been lower. Flow is unaffected — it is instantaneous.

Usage:
    python episode/e06_replay.py                       # the Aug 12-13 gap
    python episode/e06_replay.py --from 2026-08-13T00:00Z --to 2026-08-13T18:00Z
    python episode/e06_replay.py --email               # actually send the digest
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import bucket, ep_key, load_config, load_mrms_hour, load_nwm_hour
from monitor_common import config, figure
from monitor_common.s3io import read_parquet, write_bytes, write_parquet
from monitor_common.triggers import evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e06")

CFS = 35.3146667
GAP_START = pd.Timestamp("2026-08-12 16:00", tz="UTC")
GAP_END = pd.Timestamp("2026-08-13 18:00", tz="UTC")


def sample_bridges(cfg: pd.DataFrame, hours: list[pd.Timestamp]) -> dict:
    """{hour: DataFrame(bridge_id, precip_in)} from the stored MRMS subsets."""
    lats = cfg["lat"].to_numpy(float)
    lons = cfg["lon"].to_numpy(float)
    out = {}
    for ts in hours:
        got = load_mrms_hour(ts)
        if got is None:
            continue
        arr, glat_desc, glon_asc = got
        ilat = np.abs(glat_desc[None, :] - lats[:, None]).argmin(axis=1)
        ilon = np.abs(glon_asc[None, :] - lons[:, None]).argmin(axis=1)
        v = arr[ilat, ilon]
        out[ts] = pd.DataFrame({"bridge_id": cfg["bridge_id"].to_numpy(),
                                "precip_in": np.where(np.isfinite(v), v, 0.0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="t0", default=GAP_START.isoformat())
    ap.add_argument("--to", dest="t1", default=GAP_END.isoformat())
    ap.add_argument("--seed", default="aug12_alert_state.parquet",
                    help="alert_state snapshot to start declustering from; "
                         "'none' starts empty")
    ap.add_argument("--email", action="store_true", help="send the digest via SES")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    t0 = pd.Timestamp(args.t0).tz_convert("UTC")
    t1 = pd.Timestamp(args.t1).tz_convert("UTC")
    hours = list(pd.date_range(t0, t1, freq="1h"))
    cfg = load_config()
    log.info("Replaying %d hour(s) %s -> %s over %d bridges",
             len(hours), t0, t1, len(cfg))

    if args.seed and args.seed.lower() != "none":
        alert_state = read_parquet(bucket(), ep_key(args.seed))
        for c in ("last_wet_hour", "last_alert_hour"):
            if c in alert_state.columns:
                alert_state[c] = pd.to_datetime(alert_state[c], utc=True)
        log.info("Seeded declustering from %s (%d rows)", args.seed, len(alert_state))
    else:
        alert_state = pd.DataFrame(columns=["bridge_id", "trigger_type", "last_wet_hour",
                                            "last_alert_hour", "last_severity_rp"])

    # MRMS for the trailing window as well as the replay window
    lookback = list(pd.date_range(t0 - pd.Timedelta(hours=config.STATE_HOURS), t1, freq="1h"))
    log.info("Sampling MRMS at bridge points for %d hour(s)...", len(lookback))
    mrms_all = sample_bridges(cfg, lookback)
    log.info("  %d hour(s) available", len(mrms_all))

    comids = cfg["comid"].dropna().astype(np.int64).unique()
    comid_by_bridge = cfg.set_index("bridge_id")["comid"]
    all_fires = []
    for ts in hours:
        nw = load_nwm_hour(ts)
        if nw is None:
            log.warning("no NWM for %s — hour skipped", ts)
            continue
        latest = nw.reindex(comids)
        latest["streamflow_cms"] = latest["q_ol_cms"]
        window = [h for h in pd.date_range(ts - pd.Timedelta(hours=config.STATE_HOURS - 1),
                                           ts, freq="1h") if h in mrms_all]
        slices = {h: mrms_all[h] for h in window}
        fires, alert_state = evaluate(cfg, slices, latest, ts if slices else None,
                                      ts, alert_state)
        if fires:
            log.info("  %s -> %d new event(s)", ts, len(fires))
            for f in fires:                       # carry A&A so the digest can flag it
                c = comid_by_bridge.get(f["bridge_id"], np.nan)
                q = nw["q_aa_cms"].get(c, np.nan) if pd.notna(c) else np.nan
                f["q_aa_cfs"] = float(q) * CFS if pd.notna(q) else np.nan
            all_fires += fires

    if not all_fires:
        log.info("Replay produced no new events — nothing was missed in this window.")
        return

    ev = pd.DataFrame(all_fires)
    meta = cfg.set_index("bridge_id")
    ev["asset"] = ev["bridge_id"].map(meta[config.ASSET_COL]).fillna(ev["bridge_id"])
    for c in ("lat", "lon", "comid", "tc_dur_hr"):
        ev[c] = ev["bridge_id"].map(meta[c])
    ev["scour"] = ev["bridge_id"].map(meta[config.SCOUR_COL]).fillna(False).astype(bool)
    ev["aa_confirms"] = (ev["trigger_type"].eq("flow")
                         & (ev["q_aa_cfs"] >= ev["threshold"])).fillna(False)
    ev["map_class"] = np.where(ev["trigger_type"].eq("precip"), "precip",
                        np.where(ev["aa_confirms"], "flow_conf", "flow_open"))

    log.info("REPLAY RESULT: %d event(s), %d bridge(s), %d scour-critical",
             len(ev), ev["bridge_id"].nunique(), int(ev["scour"].sum()))
    log.info("  by trigger:\n%s", ev["trigger_type"].value_counts().to_string())
    log.info("  by corroboration:\n%s", ev["map_class"].value_counts().to_string())

    stamp = f"{t0:%Y%m%d%H}_{t1:%Y%m%d%H}"
    counties = None
    try:
        counties = read_parquet(bucket(), config.keys()["counties"])
    except Exception:  # noqa: BLE001
        pass
    pdf = figure.build_digest_pdf(ev, cfg, counties, t1, t1)
    if not args.no_upload:
        write_parquet(ev, bucket(), ep_key(f"replay/events_{stamp}.parquet"))
        write_bytes(pdf, bucket(), ep_key(f"replay/digest_{stamp}.pdf"),
                    content_type="application/pdf")
        log.info("Wrote episode/replay/digest_%s.pdf (%.0f KB)", stamp, len(pdf) / 1024)

    if args.email:
        import lambda_alerter
        body = ("*** REPLAY — reconstructed from archived data, not a live alert ***\n"
                f"Hours {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M} UTC were never "
                "evaluated by the poller.\n\n"
                + lambda_alerter._digest_body(ev, t1, t1))
        sent = lambda_alerter._send_email(
            f"[REPLAY] {ev['bridge_id'].nunique()} bridge(s) missed "
            f"({t0:%m-%d %H:%MZ} to {t1:%m-%d %H:%MZ})", body, pdf,
            f"replay_{stamp}.pdf")
        log.info("Email sent: %s", sent)


if __name__ == "__main__":
    main()
