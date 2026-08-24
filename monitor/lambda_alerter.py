"""Alerter Lambda — builds ONE digest PDF for the whole run and emails it.

Invoked asynchronously (InvocationType='Event') by the poller, once per run:
    {"events_key": "…/monitor/alerts/pending/2026081215.parquet",
     "mrms_hour": "…", "nwm_hour": "…"}

The digest is a single PDF (statewide map + paginated bridge table) and a
single email whose body lists every affected bridge and what triggered it.
One message per run — not one per bridge, which swamps the SES send rate and
buries the reader.

The legacy per-bridge payload ({"bridge_id": …, "triggers": […]}) is still
accepted so a single bridge can be re-rendered in detail on demand.
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
# Lambda pre-configures the root logger, so basicConfig above is a no-op and
# INFO never reaches CloudWatch. Set the level explicitly.
logging.getLogger().setLevel(logging.INFO)
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


# SES caps a raw message at 10 MB, and MIME base64 inflates the payload by ~33%.
# Budget against the ENCODED size so a large PDF degrades to a link rather than
# failing the send: an oversized attachment must never cost the whole alert.
SES_RAW_LIMIT = 10 * 1024 * 1024
ATTACH_BUDGET = int(SES_RAW_LIMIT * 0.72)      # ~7.2 MB of PDF once encoded


def _presign(key: str, days: int = 7) -> str | None:
    import boto3
    try:
        return boto3.client("s3", region_name=config.REGION).generate_presigned_url(
            "get_object", Params={"Bucket": config.keys()["bucket"], "Key": key},
            ExpiresIn=days * 86400)
    except Exception as e:  # noqa: BLE001
        log.warning("presign failed for %s: %s", key, e)
        return None


def _send_email(subject: str, body: str, pdf: bytes, filename: str,
                pdf_key: str | None = None) -> bool:
    if not config.ALERT_SENDER or not config.ALERT_RECIPIENTS:
        log.warning("SES not configured (MONITOR_ALERT_SENDER / MONITOR_ALERT_RECIPIENTS) "
                    "— PDF archived only, no email sent.")
        return False
    import boto3
    attach = pdf is not None and len(pdf) <= ATTACH_BUDGET
    if pdf is not None and not attach:
        link = _presign(pdf_key) if pdf_key else None
        log.warning("PDF is %.1f MB, over the %.1f MB attachment budget — sending a "
                    "link instead of the attachment.",
                    len(pdf) / 1e6, ATTACH_BUDGET / 1e6)
        body += ("\n\n" + "-" * 78 +
                 f"\nThe report ({len(pdf) / 1e6:.1f} MB) was too large to attach.\n"
                 + (f"Download (link valid 7 days):\n{link}\n" if link else
                    f"It is archived at s3://{config.keys()['bucket']}/{pdf_key}\n"))

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = config.ALERT_SENDER
    msg["To"] = ", ".join(config.ALERT_RECIPIENTS)
    msg.attach(MIMEText(body, "plain"))
    if attach:
        att = MIMEApplication(pdf, _subtype="pdf")
        att.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(att)
    boto3.client("ses", region_name=config.REGION).send_raw_email(
        Source=config.ALERT_SENDER, Destinations=config.ALERT_RECIPIENTS,
        RawMessage={"Data": msg.as_string()})
    log.info("Alert emailed to %s (%s)", config.ALERT_RECIPIENTS,
             "with attachment" if attach else "link only")
    return True


def _digest_body(ev: pd.DataFrame, mrms_hour, nwm_hour) -> str:
    """Plain-text roster: every affected bridge and what set it off."""
    n_b = ev["bridge_id"].nunique()
    n_flow = int((ev["trigger_type"] == "flow").sum())
    n_prec = int((ev["trigger_type"] == "precip").sum())
    n_open = int((ev["map_class"] == "flow_open").sum())
    n_scour = int(ev["scour"].astype(bool).sum())
    hour = nwm_hour or mrms_hour

    L = [f"INDOT BRIDGE FLOOD ALERT — {n_b} bridge(s) triggered",
         f"Valid {hour:%Y-%m-%d %H:%M} UTC"
         + (f"   (MRMS {mrms_hour:%H%M}Z, NWM {nwm_hour:%H%M}Z)"
            if mrms_hour is not None and nwm_hour is not None else ""),
         "",
         f"  {n_flow} streamflow · {n_prec} precipitation · {n_scour} scour-critical",
         "  Severity: " + ", ".join(
             f"{int((ev['severity_rp'] == rp).sum())} x {rp}-yr"
             for rp in config.SEVERITY_RPS),
         ]
    if n_open:
        L += ["", f"  NOTE: {n_open} streamflow alert(s) are open-loop only — NWM's",
              "        data-assimilation run does not reproduce them. Marked 'A&A: NO'",
              "        below and orange on the map. Treat as unconfirmed."]
    L += ["", "The attached PDF has a statewide map (colored by which product",
          "confirms each alert) and the full table.", "",
          "-" * 78,
          f"{'BRIDGE':<28}{'TRIGGER':<10}{'SEV':>7}{'OBSERVED':>14}{'THRESHOLD':>14}{'  A&A':<6}",
          "-" * 78]

    ev = ev.sort_values(["severity_rp", "observed"], ascending=[False, False])
    for _, r in ev.iterrows():
        unit = "in" if r["trigger_type"] == "precip" else "cfs"
        fmt = "{:,.2f}" if r["trigger_type"] == "precip" else "{:,.0f}"
        aa = "  -" if r["trigger_type"] == "precip" else ("  yes" if r["aa_confirms"] else "  NO")
        name = str(r["asset"])[:26] + ("*" if bool(r["scour"]) else "")
        L.append(f"{name:<28}{r['trigger_type'].upper():<10}"
                 f"{int(r['severity_rp']):>4}-yr"
                 f"{fmt.format(r['observed']) + ' ' + unit:>14}"
                 f"{fmt.format(r['threshold']) + ' ' + unit:>14}{aa:<6}")
    L += ["-" * 78, "* = scour-critical (fires at the 10-yr; all others at the 50-yr)", "",
          "Precip trigger: trailing round(Kirpich Tc)-h MRMS >= Atlas-14 depth.",
          "Flow trigger:   NWM open-loop streamflow >= retrospective LP3 Q (04c).",
          "Alerts are de-duplicated on a 24-h event separation.", ""]
    return "\n".join(L)


def _digest_handler(event):
    from monitor_common.s3io import read_parquet, write_bytes
    k = config.keys()
    ev = read_parquet(k["bucket"], event["events_key"])
    if ev.empty:
        log.warning("Digest requested but the event table is empty — nothing sent.")
        return {"bridges": 0, "emailed": False}

    mrms_hour = pd.Timestamp(event["mrms_hour"]).tz_convert("UTC") if event.get("mrms_hour") else None
    nwm_hour = pd.Timestamp(event["nwm_hour"]).tz_convert("UTC") if event.get("nwm_hour") else None
    hour = nwm_hour or mrms_hour

    cfg = catalog.load()
    try:
        counties = read_parquet(k["bucket"], k["counties"])
    except Exception as e:  # noqa: BLE001  (map still renders without outlines)
        log.warning("County outlines unavailable (%s) — map drawn without them. "
                    "Run precompute/p07_map_assets.py to create them.", e)
        counties = None

    pdf = figure.build_digest_pdf(ev, cfg, counties, mrms_hour, nwm_hour)
    fname = f"digest_{hour:%Y%m%d%H}.pdf"
    write_bytes(pdf, k["bucket"], f"{k['alerts']}{fname}", content_type="application/pdf")

    n_b = ev["bridge_id"].nunique()
    top = int(ev["severity_rp"].max())
    n_scour = int(ev["scour"].astype(bool).sum())
    subject = (f"[BRIDGE FLOOD ALERT] {n_b} bridge(s) — up to {top}-yr"
               + (f", {n_scour} scour-critical" if n_scour else "")
               + f" ({hour:%Y-%m-%d %H:%MZ})")
    emailed = _send_email(subject, _digest_body(ev, mrms_hour, nwm_hour), pdf, fname,
                          pdf_key=f"{k['alerts']}{fname}")
    log.info("Digest built for %d bridges -> %s (emailed=%s)", n_b, fname, emailed)
    return {"bridges": n_b, "pdf": fname, "emailed": emailed}


def handler(event, context=None):
    if "events_key" in event:
        return _digest_handler(event)
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
