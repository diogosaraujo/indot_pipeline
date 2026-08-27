"""Trigger evaluation + 24-h event separation for the monitor.

Precip trigger : trailing round(Tc)-hour MRMS accumulation >= Atlas-14 depth at
                 that duration (per scripts/08c_tc_trigger_analysis.py).
Flow trigger   : NWM open-loop hourly streamflow (cfs) >= retro-LP3 Q
                 (per scripts/08f_nwm_trigger_analysis.py, gauge-free operating point).

Firing floor   : scour-critical bridges fire at >= FIRE_RP_SCOUR (10-yr);
                 all other over-water bridges fire at >= FIRE_RP_OTHER (50-yr).
Severity       : the highest tier in SEVERITY_RPS whose threshold is exceeded.

Event separation mirrors group_wet_events (08_trigger_analysis.py): a new alert
is raised only when the current wet hour is > DECLUSTER_GAP_HOURS (24 h) after
the previous wet hour for that (bridge, trigger); wet hours within the gap are
one ongoing event and are suppressed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("monitor.triggers")


def build_wide(slices: dict[pd.Timestamp, pd.DataFrame], value_col: str,
               key_col: str) -> pd.DataFrame:
    """[hours x keys] matrix from a {ts: slice_df} mapping."""
    cols = []
    for ts, df in sorted(slices.items()):
        s = df.set_index(key_col)[value_col]
        s.name = ts
        cols.append(s)
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).T.sort_index()   # index=ts, columns=key


def trailing_precip(mrms_wide: pd.DataFrame, cfg: pd.DataFrame,
                    current_hour: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Per-bridge trailing round(Tc)-hour precip sum and hour-count, aligned to cfg rows.

    Returns (sum_in, count) as arrays in cfg order. count < tc_dur means the
    window is incomplete (missing hours) and firing is suppressed by the caller.
    """
    n = len(cfg)
    sums = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    if mrms_wide.empty:
        return sums, counts
    w = mrms_wide.loc[mrms_wide.index <= current_hour]
    bridge_ids = cfg["bridge_id"].to_numpy()
    tc = cfg["tc_dur_hr"].to_numpy()
    pos = {bid: i for i, bid in enumerate(bridge_ids)}
    for d in np.unique(tc):
        d = int(d)
        members = bridge_ids[tc == d]
        members = [m for m in members if m in w.columns]
        if not members:
            continue
        block = w[members].tail(d)
        bsum = block.sum(axis=0, skipna=True)
        bcnt = block.notna().sum(axis=0)
        for m in members:
            i = pos[m]
            sums[i] = float(bsum.get(m, 0.0))
            counts[i] = int(bcnt.get(m, 0))
    return sums, counts


def _severity(values: np.ndarray, cfg: pd.DataFrame, col_fmt: str) -> np.ndarray:
    """Highest RP in SEVERITY_RPS whose threshold column is exceeded (0 = none)."""
    sev = np.zeros(len(cfg), dtype=int)
    for rp in config.SEVERITY_RPS:            # ascending -> higher tiers overwrite
        thr = pd.to_numeric(cfg[col_fmt.format(rp=rp)], errors="coerce").to_numpy()
        hit = np.isfinite(thr) & (values >= thr)
        sev = np.where(hit, rp, sev)
    return sev


def evaluate(cfg: pd.DataFrame,
             mrms_slices: dict[pd.Timestamp, pd.DataFrame],
             nwm_latest: pd.DataFrame | None,
             mrms_hour: pd.Timestamp | None,
             nwm_hour: pd.Timestamp | None,
             alert_state: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Return (firing_events, updated_alert_state).

    firing_events: list of dicts with bridge_id, trigger_type, valid_hour,
    observed, threshold, severity_rp — one per NEW event (post-declustering).
    """
    fire_floor = np.where(cfg[config.SCOUR_COL].to_numpy(bool),
                          config.FIRE_RP_SCOUR, config.FIRE_RP_OTHER)

    rows: list[dict] = []   # candidate wet rows (bridge, trigger) this hour

    # ── Precip ───────────────────────────────────────────────────────────────
    if mrms_hour is not None and mrms_slices:
        wide = build_wide(mrms_slices, "precip_in", "bridge_id")
        psum, pcnt = trailing_precip(wide, cfg, mrms_hour)
        sev = _severity(psum, cfg, "P{rp}")
        complete = pcnt >= cfg["tc_dur_hr"].to_numpy()
        wet = (sev > 0) & (sev >= fire_floor) & complete
        for i in np.nonzero(wet)[0]:
            rp = int(sev[i])
            rows.append({"bridge_id": cfg["bridge_id"].iat[i], "trigger_type": "precip",
                         "valid_hour": mrms_hour, "observed": float(psum[i]),
                         "threshold": float(cfg[f"P{rp}"].iat[i]), "severity_rp": rp})

    # ── Flow ─────────────────────────────────────────────────────────────────
    if (nwm_hour is not None and nwm_latest is not None and not nwm_latest.empty
            and "streamflow_cms" in nwm_latest.columns):
        comid = pd.to_numeric(cfg["comid"], errors="coerce")
        q_cms = comid.map(nwm_latest["streamflow_cms"]).to_numpy()
        q_cfs = q_cms * config.CFS_PER_CMS
        sev = _severity(q_cfs, cfg, "Q{rp}_cfs")
        wet = (sev > 0) & (sev >= fire_floor) & np.isfinite(q_cfs)
        for i in np.nonzero(wet)[0]:
            rp = int(sev[i])
            rows.append({"bridge_id": cfg["bridge_id"].iat[i], "trigger_type": "flow",
                         "valid_hour": nwm_hour, "observed": float(q_cfs[i]),
                         "threshold": float(cfg[f"Q{rp}_cfs"].iat[i]), "severity_rp": rp})

    if not rows:
        return [], alert_state

    # ── Declustering, and re-alerting while an event persists ────────────────
    # A NEW event needs a gap of more than DECLUSTER_GAP_HOURS since the bridge
    # was last wet (matching group_wet_events in the study). But a bridge that
    # stays wet refreshes last_wet_hour every hour, so the gap never opens and a
    # multi-day flood would notify exactly once. So also re-fire when the event
    # is still running and REALERT_HOURS have passed since the last alert, and
    # carry the event's start so the digest can say how long it has been going.
    gap = pd.Timedelta(hours=config.DECLUSTER_GAP_HOURS)
    regap = pd.Timedelta(hours=config.REALERT_HOURS)
    st = alert_state.set_index(["bridge_id", "trigger_type"]).to_dict("index")

    fires: list[dict] = []
    for r in rows:
        keyt = (r["bridge_id"], r["trigger_type"])
        rec = dict(st.get(keyt, {}))
        now_h = r["valid_hour"]
        last_wet = rec.get("last_wet_hour")
        last_alert = rec.get("last_alert_hour")
        start = rec.get("event_start_hour")

        is_new = last_wet is None or pd.isna(last_wet) or (now_h - last_wet) > gap
        if is_new:
            start = now_h
            fire, reason = True, "new"
        else:
            due = (last_alert is None or pd.isna(last_alert)
                   or (now_h - last_alert) >= regap)
            fire, reason = due, ("ongoing" if due else "")

        rec["last_wet_hour"] = now_h
        rec["event_start_hour"] = start
        if fire:
            rec["last_alert_hour"] = now_h
            rec["last_severity_rp"] = r["severity_rp"]
            hours = 0.0 if pd.isna(start) else (now_h - start).total_seconds() / 3600.0
            fires.append({**r, "event_start_hour": start,
                          "event_hours": round(hours, 1), "alert_reason": reason})
        else:
            rec.setdefault("last_alert_hour", pd.NaT)
            rec.setdefault("last_severity_rp", r["severity_rp"])
        st[keyt] = rec

    new_state = (pd.DataFrame([{"bridge_id": k[0], "trigger_type": k[1], **v}
                               for k, v in st.items()])
                 if st else alert_state)
    return fires, new_state
