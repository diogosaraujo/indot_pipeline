"""Alerter Lambda — builds the PDF for one firing bridge and emails it via SES.

Invoked asynchronously (InvocationType='Event') by the poller, once per firing
bridge. Payload:
    {"bridge_id": "...", "valid_hour": "2026-07-29T14:00:00+00:00",
     "triggers": [{"type": "precip"|"flow", "observed": .., "threshold": .., "severity_rp": ..}]}
"""
from __future__ import annotations

import logging
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests

from monitor_common import catalog, config, figure, mrms, state

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("monitor.alerter")

_REACH: dict[int, list] = {}     # warm-container COMID -> [(lon,lat), ...] cache


def _val(slice_df, key_col, key, value_col):
    if slice_df is None or slice_df.empty:
        return np.nan
    s = slice_df.set_index(key_col)[value_col] if key_col in slice_df.columns else None
    if s is None or key not in s.index:
        return np.nan
    v = s.loc[key]
    return float(v) if np.isscalar(v) or (hasattr(v, "size") and v.size == 1) else float(np.asarray(v).ravel()[0])


def _series(bridge, valid_hour):
    tc = int(bridge["tc_dur_hr"])
    lookback = 24 + tc
    hrs = list(pd.date_range(valid_hour - pd.Timedelta(hours=lookback - 1), valid_hour, freq="1h"))
    plot = hrs[-24:]

    mslices = state.read_recent("mrms", hrs)
    precip = np.array([_val(mslices.get(h), "bridge_id", bridge["bridge_id"], "precip_in") for h in hrs])
    precip = np.nan_to_num(precip, nan=0.0)
    accum = pd.Series(precip, index=hrs).rolling(tc, min_periods=1).sum().to_numpy()

    nslices = state.read_recent("nwm", plot)
    cid = bridge["comid"]
    C = config.CFS_PER_CMS

    def col(h, c):
        return _val(nslices.get(h), "comid", cid, c) if cid is not None else np.nan
    q_ol = np.array([col(h, "q_ol_cms") for h in plot]) * C
    q_aa = np.array([col(h, "q_aa_cms") for h in plot]) * C
    v_ol = np.array([col(h, "v_ol_ms") for h in plot])
    v_aa = np.array([col(h, "v_aa_ms") for h in plot])

    idx = {h: i for i, h in enumerate(hrs)}
    plot_precip = precip[[idx[h] for h in plot]]
    plot_accum = accum[[idx[h] for h in plot]]
    return plot, plot_precip, plot_accum, q_aa, q_ol, v_aa, v_ol


def _reach_geometry(comid: int | None):
    if comid is None:
        return None
    if comid in _REACH:
        return _REACH[comid]
    try:
        r = requests.get(f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}", timeout=20)
        r.raise_for_status()
        feats = r.json().get("features", [])
        coords: list = []
        for f in feats:
            g = f.get("geometry", {})
            if g.get("type") == "LineString":
                coords.extend(g["coordinates"])
            elif g.get("type") == "MultiLineString":
                for part in g["coordinates"]:
                    coords.extend(part)
        reach = [(float(c[0]), float(c[1])) for c in coords] or None
    except Exception as e:  # noqa: BLE001
        log.warning("NLDI reach geometry failed for COMID %s: %s", comid, e)
        reach = None
    _REACH[comid] = reach
    return reach


def _send_email(subject: str, body: str, pdf: bytes, filename: str) -> bool:
    if not config.ALERT_SENDER or not config.ALERT_RECIPIENTS:
        log.warning("SES not configured (MONITOR_ALERT_SENDER / MONITOR_ALERT_RECIPIENTS) "
                    "— PDF archived only, no email sent.")
        return False
    import boto3
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = config.ALERT_SENDER
    msg["To"] = ", ".join(config.ALERT_RECIPIENTS)
    msg.attach(MIMEText(body, "plain"))
    att = MIMEApplication(pdf, _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(att)
    boto3.client("ses", region_name=config.REGION).send_raw_email(
        Source=config.ALERT_SENDER, Destinations=config.ALERT_RECIPIENTS,
        RawMessage={"Data": msg.as_string()})
    log.info("Alert emailed to %s", config.ALERT_RECIPIENTS)
    return True


def handler(event, context=None):
    cfg = catalog.load()
    bridge = catalog.bridge_row(cfg, event["bridge_id"])
    valid_hour = pd.Timestamp(event["valid_hour"]).tz_convert("UTC")
    fired = event.get("triggers", [])
    log.info("Building alert for bridge %s (%s) valid %s", bridge["asset"],
             "scour" if bridge["scour"] else "over-water", valid_hour)

    hrs, precip_1h, tc_accum, q_aa, q_ol, v_aa, v_ol = _series(bridge, valid_hour)

    # bridge-cell MRMS grid for the map (peak of the last 24 h if available)
    grid = mrms.read_grid(valid_hour)
    reach = _reach_geometry(bridge["comid"])

    pdf = figure.build_pdf(bridge, valid_hour, fired, hrs, precip_1h, tc_accum,
                           q_aa, q_ol, v_aa, v_ol, grid, reach)

    k = config.keys()
    safe = "".join(ch if ch.isalnum() else "_" for ch in bridge["asset"]).strip("_")
    fname = f"alert_{safe}_{valid_hour:%Y%m%d%H}.pdf"
    from monitor_common.s3io import write_bytes
    write_bytes(pdf, k["bucket"], f"{k['alerts']}{fname}", content_type="application/pdf")

    sev = max((f["severity_rp"] for f in fired), default=0)
    types = "/".join(sorted({f["type"] for f in fired}))
    subject = (f"[BRIDGE FLOOD ALERT] {bridge['asset']} — {types} ≥ {sev}-yr "
               f"({valid_hour:%Y-%m-%d %H:%MZ})")
    body = (f"Bridge {bridge['asset']} ({'scour-critical' if bridge['scour'] else 'over-water'})\n"
            f"Location: {bridge['lat']:.5f}, {bridge['lon']:.5f}   COMID: {bridge['comid']}\n"
            f"Valid: {valid_hour:%Y-%m-%d %H:%MZ}\n\n"
            + "\n".join(f"  {f['type'].upper()}: observed {f['observed']:,.2f} "
                        f">= {'P' if f['type']=='precip' else 'Q'}{f['severity_rp']} "
                        f"threshold {f['threshold']:,.2f}" for f in fired)
            + "\n\nSee attached PDF for maps and 24-h time series.\n")
    _send_email(subject, body, pdf, fname)
    return {"bridge_id": bridge["bridge_id"], "pdf": fname, "severity_rp": sev}
