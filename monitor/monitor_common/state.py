"""Rolling state for the monitor.

Design: append-only per-hour slice files (tiny parquets) rather than one big
rewritten table, so hourly writes never race and reads are cheap.

    monitor/state/mrms/{YYYYMMDDHH}.parquet   bridge_id, precip_in
    monitor/state/nwm/{YYYYMMDDHH}.parquet    comid, q_ol_cms, v_ol_ms,
                                              q_aa_cms, v_aa_ms

Alert-dedup state (one small rewritten table) enforces the 24-h event
separation from scripts/08_trigger_analysis.py:

    monitor/alert_state.parquet   bridge_id, trigger_type, last_wet_hour,
                                  last_alert_hour, last_severity_rp
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config
from .s3io import list_keys, read_parquet, write_parquet

log = logging.getLogger("monitor.state")

STAMP = "%Y%m%d%H"


def stamp(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime(STAMP)


def _parse_stamp(key: str) -> pd.Timestamp | None:
    base = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]   # .parquet or .npz
    try:
        return pd.Timestamp(pd.to_datetime(base, format=STAMP)).tz_localize("UTC")
    except Exception:  # noqa: BLE001
        return None


_PREFIX_KEY = {"mrms": "state_mrms", "nwm": "state_nwm", "grid": "state_grid"}


def _prefix(kind: str) -> str:
    return config.keys()[_PREFIX_KEY[kind]]


def write_slice(kind: str, ts: pd.Timestamp, df: pd.DataFrame) -> None:
    b = config.keys()["bucket"]
    write_parquet(df, b, f"{_prefix(kind)}{stamp(ts)}.parquet")


# ── Gridded MRMS slices ──────────────────────────────────────────────────────
# The digest map needs a 24-h accumulation FIELD, not point samples. Re-reading
# 24 CONUS gribs in the alerter would take minutes and blow its timeout, so the
# poller — which already has each grid in memory — stores the Indiana window
# here (~0.5 MB/h) and the alerter just sums them.

def write_grid(ts: pd.Timestamp, arr, lats, lons) -> None:
    import io
    import numpy as np
    from .s3io import write_bytes
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr, lats=lats, lons=lons)
    write_bytes(buf.getvalue(), config.keys()["bucket"],
                f"{_prefix('grid')}{stamp(ts)}.npz",
                content_type="application/octet-stream")


def read_grid(ts: pd.Timestamp):
    """(arr, lats, lons) for one stored hour, or None if absent."""
    import io
    import numpy as np
    from .s3io import read_bytes
    try:
        z = np.load(io.BytesIO(read_bytes(config.keys()["bucket"],
                                          f"{_prefix('grid')}{stamp(ts)}.npz")))
    except Exception:  # noqa: BLE001
        return None
    return z["arr"], z["lats"], z["lons"]


def accumulate_grid(hours: list[pd.Timestamp]):
    """Summed depth over `hours`, plus how many were actually found.

    Returns (arr, lats, lons, n_found). The count is reported rather than
    silently absorbed so a partial window can be labelled as such.
    """
    import numpy as np
    acc = lats = lons = None
    n = 0
    for ts in hours:
        got = read_grid(ts)
        if got is None:
            continue
        a, lats, lons = got
        acc = a.astype(np.float64) if acc is None else acc + a
        n += 1
    return acc, lats, lons, n


def existing_hours(kind: str) -> dict[pd.Timestamp, str]:
    """Map available valid-hour -> S3 key for this state stream."""
    b = config.keys()["bucket"]
    out: dict[pd.Timestamp, str] = {}
    for k in list_keys(b, _prefix(kind)):
        ts = _parse_stamp(k)
        if ts is not None:
            out[ts] = k
    return out


def read_recent(kind: str, hours: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    b = config.keys()["bucket"]
    avail = existing_hours(kind)
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts in hours:
        key = avail.get(ts)
        if key is None:
            continue
        try:
            out[ts] = read_parquet(b, key)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read state slice %s: %s", key, e)
    return out


def prune(kind: str, keep_after: pd.Timestamp) -> int:
    """Delete slice files older than `keep_after`. Returns count removed."""
    from .s3io import delete_keys
    b = config.keys()["bucket"]
    stale = [k for ts, k in existing_hours(kind).items() if ts < keep_after]
    if stale:
        delete_keys(b, stale)
    return len(stale)


# ── Alert-dedup state ────────────────────────────────────────────────────────

_ALERT_COLS = ["bridge_id", "trigger_type", "last_wet_hour",
               "last_alert_hour", "last_severity_rp", "event_start_hour"]


def read_alert_state() -> pd.DataFrame:
    k = config.keys()
    try:
        df = read_parquet(k["bucket"], k["alert_state"])
    except Exception:  # noqa: BLE001  (missing on first run)
        return pd.DataFrame(columns=_ALERT_COLS)
    df["bridge_id"] = df["bridge_id"].astype(str)
    if "event_start_hour" not in df.columns:      # state written before re-alerting
        df["event_start_hour"] = df.get("last_alert_hour", pd.NaT)
    for c in ("last_wet_hour", "last_alert_hour", "event_start_hour"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def write_alert_state(df: pd.DataFrame) -> None:
    k = config.keys()
    for c in _ALERT_COLS:
        if c not in df.columns:
            df[c] = pd.NaT if c.endswith("_hour") else None
    write_parquet(df[_ALERT_COLS], k["bucket"], k["alert_state"])


# ── Source-health state ──────────────────────────────────────────────────────
# One tiny JSON holding whether the sources were last seen stale. Only the
# TRANSITION is notified: a mirror backlog can last many hours, and an email per
# hour through it would train the reader to ignore the alarm.

def read_health() -> dict:
    import json
    from .s3io import read_bytes
    k = config.keys()
    try:
        return json.loads(read_bytes(k["bucket"], k["health"]).decode())
    except Exception:  # noqa: BLE001  (absent on first run)
        return {}


def write_health(d: dict) -> None:
    import json
    from .s3io import write_bytes
    k = config.keys()
    write_bytes(json.dumps(d, default=str).encode(), k["bucket"], k["health"],
                content_type="application/json")
