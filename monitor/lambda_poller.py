"""Poller Lambda — the hourly fan-in.

Runs once an hour (EventBridge Scheduler). Fetches the single CONUS MRMS grib
and NWM channel_rt files ONCE, samples every bridge, appends to rolling state,
evaluates both triggers for all bridges, and async-invokes the alerter Lambda
once per firing bridge. This is deliberately NOT one Lambda per bridge.

Env: MONITOR_BUCKET, MONITOR_PREFIX, MONITOR_ALERTER_FUNCTION, and the tuning
knobs in monitor_common/config.py.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from monitor_common import catalog, config, mrms, nwm, state
from monitor_common.triggers import evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("monitor.poller")


def _window_hours(latest: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(latest - pd.Timedelta(hours=config.STATE_HOURS - 1),
                              latest, freq="1h"))


def _ingest_mrms(cfg: pd.DataFrame, latest: pd.Timestamp) -> None:
    lats = cfg["lat"].to_numpy(float)
    lons = cfg["lon"].to_numpy(float)
    have = set(state.existing_hours("mrms"))
    want = [h for h in _window_hours(latest) if h not in have]
    want = sorted(want)[-config.BACKFILL_MAX_HOURS:]
    if latest not in want and latest not in have:
        want.append(latest)
    for ts in want:
        vals = mrms.sample_points(ts, lats, lons)
        if vals is None:
            continue
        state.write_slice("mrms", ts, pd.DataFrame({"bridge_id": cfg["bridge_id"].to_numpy(),
                                                     "precip_in": vals}))
        log.info("MRMS slice written %s (mean=%.3f in, wet cells=%d)",
                 state.stamp(ts), float(np.nanmean(vals)), int((vals > 0).sum()))


def _ingest_nwm(comids: np.ndarray, ol_hour, aa_hour) -> pd.DataFrame | None:
    """Write NWM slices for the newest hour; return the latest merged slice."""
    latest = ol_hour or aa_hour
    if latest is None:
        return None
    have = set(state.existing_hours("nwm"))
    want = [latest] if latest not in have else []
    # light backfill so the alerter's 24-h series has recent hours
    for h in _window_hours(latest):
        if h not in have and h != latest:
            want.append(h)
    want = sorted(set(want))[-config.BACKFILL_MAX_HOURS:]
    if latest not in want:
        want.append(latest)
    latest_df = None
    for ts in sorted(set(want)):
        ol = nwm.read_comids(ts, config.NWM_PRODUCT_TRIGGER, comids)
        aa = nwm.read_comids(ts, config.NWM_PRODUCT_DISPLAY, comids)
        if ol is None and aa is None:
            continue
        base = pd.DataFrame(index=pd.Index(comids, name="comid"))
        if ol is not None:
            base["q_ol_cms"] = ol["streamflow_cms"]
            base["v_ol_ms"] = ol["velocity_ms"]
        if aa is not None:
            base["q_aa_cms"] = aa["streamflow_cms"]
            base["v_aa_ms"] = aa["velocity_ms"]
        slice_df = base.reset_index()
        state.write_slice("nwm", ts, slice_df)
        if ts == latest:
            latest_df = base.copy()
            latest_df["streamflow_cms"] = base["q_ol_cms"] if "q_ol_cms" in base.columns else np.nan
        log.info("NWM slice written %s (%d comids)", state.stamp(ts), len(slice_df))
    if latest_df is None:
        # reconstruct from the freshly written latest slice
        recent = state.read_recent("nwm", [latest])
        if latest in recent:
            latest_df = recent[latest].set_index("comid")
            latest_df["streamflow_cms"] = (latest_df["q_ol_cms"] if "q_ol_cms" in latest_df.columns
                                           else np.nan)
    return latest_df


def _fan_out(fires: list[dict], cfg: pd.DataFrame) -> int:
    """Group fires by bridge; one async alerter invocation per firing bridge."""
    if not fires:
        return 0
    by_bridge: dict[str, list[dict]] = {}
    for f in fires:
        by_bridge.setdefault(f["bridge_id"], []).append(f)
    import boto3
    lam = boto3.client("lambda", region_name=config.REGION)
    sent = 0
    idx = cfg.set_index("bridge_id")
    for bid, evs in by_bridge.items():
        row = idx.loc[bid]
        valid_hour = max(e["valid_hour"] for e in evs)
        payload = {
            "bridge_id": bid,
            "valid_hour": valid_hour.isoformat(),
            "triggers": [{"type": e["trigger_type"], "observed": e["observed"],
                          "threshold": e["threshold"], "severity_rp": e["severity_rp"]}
                         for e in evs],
        }
        try:
            lam.invoke(FunctionName=config.ALERTER_FUNCTION, InvocationType="Event",
                       Payload=json.dumps(payload).encode())
            sent += 1
        except Exception as e:  # noqa: BLE001
            log.error("Alerter invoke failed for %s: %s", bid, e)
    return sent


def handler(event=None, context=None):
    cfg = catalog.load()
    now = pd.Timestamp.now(tz="UTC")

    mrms_hour = mrms.latest_available_hour(now)
    ol_hour = nwm.latest_available_hour(now, config.NWM_PRODUCT_TRIGGER)
    aa_hour = nwm.latest_available_hour(now, config.NWM_PRODUCT_DISPLAY)
    log.info("Latest available -> MRMS=%s  NWM_ol=%s  NWM_aa=%s", mrms_hour, ol_hour, aa_hour)

    if mrms_hour is not None:
        _ingest_mrms(cfg, mrms_hour)

    comids = cfg["comid"].dropna().astype(np.int64).unique()
    nwm_latest = _ingest_nwm(comids, ol_hour, aa_hour) if len(comids) else None
    nwm_hour = ol_hour or aa_hour

    # trailing precip series
    keep_after = (mrms_hour or now) - pd.Timedelta(hours=config.STATE_HOURS)
    mrms_slices = state.read_recent("mrms", _window_hours(mrms_hour)) if mrms_hour is not None else {}

    alert_state = state.read_alert_state()
    fires, new_state = evaluate(cfg, mrms_slices, nwm_latest, mrms_hour, nwm_hour, alert_state)
    state.write_alert_state(new_state)

    # housekeeping
    for kind in ("mrms", "nwm"):
        removed = state.prune(kind, keep_after)
        if removed:
            log.info("Pruned %d stale %s slices", removed, kind)

    sent = _fan_out(fires, cfg)
    log.info("Evaluated %d bridges: %d new firing events -> %d alert(s) dispatched",
             len(cfg), len(fires), sent)
    return {"bridges": len(cfg), "fires": len(fires), "alerts": sent,
            "mrms_hour": None if mrms_hour is None else mrms_hour.isoformat(),
            "nwm_hour": None if nwm_hour is None else nwm_hour.isoformat()}
