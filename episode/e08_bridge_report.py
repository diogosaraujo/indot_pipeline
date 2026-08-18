"""e08 — single-bridge event report: maps, hyetograph, hydrographs, gauge check.

Built for the question "was this bridge flagged, and should it have been?".
Follows the Lanesville figures' layout: a 3-panel map row (MRMS | NWM open-loop
| NWM A&A) over the bridge, then the time series that drive each trigger, then
the observed USGS record where a co-located gauge exists.

Page 1  3-panel map for the peak day + verdict box
Page 2  hyetograph with Atlas-14 depths, NWM hydrograph with retro-LP3 quantiles
Page 3  USGS observed discharge and stage (separate panels — never a dual axis)

Writes  episode/bridge/<bridge>_report.pdf   (and to S3)

Usage:
    python episode/e08_bridge_report.py --bridge 29-00151 --usgs 03350700
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pathlib
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from common import (HAIRLINE, INK, INK2, MUTED, PRECIP_ALPHA, SURFACE, TIER_C,  # noqa: E402
                    TZ, bucket, draw_counties, draw_flowlines, ep_key, hour_range,
                    load_config, load_counties, load_events, load_flowlines,
                    load_mrms_hour, load_nwm_hour, precip_cmap, set_geo)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e08")

CFS = 35.3146667
OL_C, AA_C, OBS_C = "#14608c", "#7b4ea8", "#0f1519"


# ── series ───────────────────────────────────────────────────────────────────

def mrms_at(lat, lon) -> pd.Series:
    """Hourly MRMS at the grid cell containing the bridge, from the e01 store."""
    out = {}
    for ts in _all_hours():
        got = load_mrms_hour(ts)
        if got is None:
            continue
        arr, lats, lons = got
        i = int(np.abs(lats - lat).argmin()); j = int(np.abs(lons - lon).argmin())
        out[ts] = float(arr[i, j])
    return pd.Series(out).sort_index()


def nwm_at(comid) -> pd.DataFrame:
    rows = []
    for ts in _all_hours():
        d = load_nwm_hour(ts)
        if d is None or comid not in d.index:
            continue
        rows.append(dict(utc=ts, ol=float(d.at[comid, "q_ol_cms"]) * CFS,
                         aa=float(d.at[comid, "q_aa_cms"]) * CFS))
    return pd.DataFrame(rows).set_index("utc").sort_index()


def _all_hours():
    from common import DAYS
    return [h for d in DAYS for h in hour_range(d)]


# NWS flood-category colours, in the AHPS convention the USGS viewer follows.
# Darkened from the pure AHPS hues so they stay legible on a light ground.
NWS_CATS = [("action", "#c99000"), ("minor", "#dd7230"),
            ("moderate", "#c02b28"), ("major", "#8b2fb5")]


def nwps_categories(lid: str) -> dict:
    """NWS action/minor/moderate/major stages for a forecast point.

    Categories are keyed by NWS LID, not USGS site number, and the gauge index
    does not carry the USGS id — SCNI3 lists it as null — so the LID has to be
    passed in rather than looked up by site.
    """
    if not lid:
        return {}
    try:
        req = urllib.request.Request(f"https://api.water.noaa.gov/nwps/v1/gauges/{lid}",
                                     headers={"User-Agent": "indot-bridge-monitor/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:  # noqa: BLE001
        log.warning("NWPS lookup failed for %s (%s) — stage panel will have no "
                    "flood categories", lid, type(e).__name__)
        return {}
    cats = (d.get("flood") or {}).get("categories", {}) or {}
    out = {k: v.get("stage") for k, v in cats.items()
           if isinstance(v, dict) and isinstance(v.get("stage"), (int, float))
           and v["stage"] > -900}
    log.info("NWPS %s (%s): %s", lid, d.get("name"),
             ", ".join(f"{k} {v} ft" for k, v in out.items()) or "none defined")
    return out


def usgs_iv(site, t0, t1) -> dict:
    """Instantaneous discharge (00060) and gage height (00065)."""
    u = (f"https://waterservices.usgs.gov/nwis/iv/?sites={site}&startDT={t0}"
         f"&endDT={t1}&parameterCd=00060,00065&format=json&siteStatus=all")
    out = {}
    try:
        j = json.load(urllib.request.urlopen(u, timeout=90))
    except Exception as e:  # noqa: BLE001
        log.warning("USGS fetch failed (%s) — gauge page will be omitted", e)
        return out
    for s in j["value"]["timeSeries"]:
        code = s["variable"]["variableCode"][0]["value"]
        vals = [(pd.Timestamp(v["dateTime"]), float(v["value"]))
                for v in s["values"][0]["value"] if v["value"] not in ("", "-999999")]
        if vals:
            t, x = zip(*vals)
            out[code] = pd.Series(x, index=pd.DatetimeIndex(t)).sort_index()
    return out


# ── pages ────────────────────────────────────────────────────────────────────

def _thresholds(ax, levels: dict, unit: str, fmt: str):
    for rp, v in levels.items():
        if not np.isfinite(v):
            continue
        ax.axhline(v, color=TIER_C[rp], ls="--", lw=1.5, zorder=2)
        ax.annotate(f"{unit}{rp} = {fmt.format(v)}", xy=(0.995, v),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color=TIER_C[rp], fontweight="bold")


def page_maps(pdf, br, day, cfg, counties, flow, q100, alerted):
    la = (br["lat"] - 0.42, br["lat"] + 0.42)
    lo = (br["lon"] - 0.52, br["lon"] + 0.52)
    fig = plt.figure(figsize=(16.0, 7.4), facecolor=SURFACE)
    axes = [fig.add_axes([0.030 + i * 0.322, 0.075, 0.295, 0.66]) for i in range(3)]

    acc = lats = lons = None
    for ts in hour_range(day):
        got = load_mrms_hour(ts)
        if got is None:
            continue
        a, lats, lons = got
        acc = a.astype(float) if acc is None else acc + a
    peak = {}
    for ts in hour_range(day):
        d = load_nwm_hour(ts)
        if d is None:
            continue
        for c in ("q_ol_cms", "q_aa_cms"):
            if c in d.columns:
                peak[c] = d[c] if c not in peak else np.fmax(peak[c], d[c])

    ax = axes[0]
    draw_counties(ax, counties, lw=0.6, fc="#f7f6f2")
    if acc is not None:
        rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
        cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
        sub = np.ma.masked_less(acc[np.ix_(rs, cs)], 0.05)
        pm = ax.pcolormesh(lons[cs], lats[rs], sub, cmap=precip_cmap(), vmin=0.05,
                           vmax=max(0.5, float(np.nanpercentile(acc[np.ix_(rs, cs)], 99.5))),
                           shading="nearest", zorder=2, alpha=PRECIP_ALPHA)
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.04)
        cb.set_label("24-h accumulation (in)", fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=7.5, colors=INK2)
    draw_counties(ax, counties, lw=0.6, overlay=True)
    ax.set_title("MRMS 24-h accumulation", fontsize=12, color=INK, loc="left", pad=7)

    for ax, key, lbl in ((axes[1], "q_ol_cms", "Peak NWM open-loop"),
                         (axes[2], "q_aa_cms", "Peak NWM A&A")):
        draw_counties(ax, counties, lw=0.6, fc="#f7f6f2")
        if key in peak:
            draw_flowlines(ax, flow, (peak[key] * CFS) / q100.reindex(peak[key].index),
                           vmax=1.5, lw_base=1.0, lat=la, lon=lo)
        draw_counties(ax, counties, lw=0.6, overlay=True)
        ax.set_title(lbl, fontsize=12, color=INK, loc="left", pad=7)

    for ax in axes:
        ax.plot([br["lon"]], [br["lat"]], marker="^", ms=15, color="#a8402a",
                mec="white", mew=1.4, zorder=9)
        set_geo(ax, la, lo)

    fig.text(0.030, 0.965, f"Bridge {br['bridge_id']} — {pd.Timestamp(day):%A %d %B %Y}",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.030, 0.925,
             f"Stony Creek · COMID {int(br['comid'])} · {br['lat']:.5f}, {br['lon']:.5f} · "
             f"Tᴄ = {int(br['tc_dur_hr'])} h · "
             + ("ALERTED" if alerted else "no alert issued during 12–16 Aug"),
             fontsize=11.5, color=INK2, va="top")
    fig.text(0.030, 0.885, "▲ bridge · rivers shaded by peak flow ÷ reach 100-yr Q · "
                           "gray reaches carry no LP3 fit",
             fontsize=9, color=MUTED, va="top")
    pdf.savefig(fig); plt.close(fig)


def page_series(pdf, br, mr, nwm):
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=SURFACE)
    axp = fig.add_axes([0.055, 0.565, 0.90, 0.31])
    axq = fig.add_axes([0.055, 0.105, 0.90, 0.34])
    tc = int(br["tc_dur_hr"])

    t = mr.index.tz_convert(TZ)
    roll = mr.rolling(tc, min_periods=tc).sum()
    axp.bar(t, mr.values, width=1 / 26, color="#6ab4ff", edgecolor="none",
            label="MRMS 1-h QPE", zorder=3)
    axp.plot(t, roll.values, color="#14608c", lw=2.2,
             label=f"trailing {tc}-h accumulation (Tᴄ)", zorder=4)
    _thresholds(axp, {rp: br[f"P{rp}"] for rp in (10, 50, 100)}, "P", "{:.2f} in")
    axp.set_ylabel("Precipitation (in)", fontsize=10)
    axp.set_title("Precipitation trigger — MRMS at the bridge cell vs Atlas-14 depths",
                  fontsize=12.5, color=INK, loc="left", pad=8)
    axp.legend(loc="upper left", fontsize=8.5, framealpha=.92)

    tq = nwm.index.tz_convert(TZ)
    axq.plot(tq, nwm["ol"], color=OL_C, lw=2.3, label="NWM open-loop (drives the trigger)", zorder=4)
    axq.plot(tq, nwm["aa"], color=AA_C, lw=1.8, ls="-.", label="NWM A&A (with data assimilation)", zorder=4)
    _thresholds(axq, {rp: br[f"Q{rp}_cfs"] for rp in (10, 50, 100)}, "Q", "{:,.0f} cfs")
    axq.set_ylabel("Streamflow (cfs)", fontsize=10)
    axq.set_title("Streamflow trigger — NWM at COMID vs retrospective LP3 quantiles",
                  fontsize=12.5, color=INK, loc="left", pad=8)
    axq.legend(loc="upper left", fontsize=8.5, framealpha=.92)

    for ax in (axp, axq):
        ax.grid(axis="y", ls=":", alpha=.45)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(HAIRLINE)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.tick_params(labelsize=8.5)

    fig.text(0.055, 0.965, f"Bridge {br['bridge_id']} — trigger time series, 12–16 August 2026",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.928,
             f"Fires at Q{50 if not br['scour_critical'] else 10} "
             f"({br['Q50_cfs'] if not br['scour_critical'] else br['Q10_cfs']:,.0f} cfs) — "
             "not scour-critical, so the 10-yr tier does not trigger an alert",
             fontsize=10.5, color=INK2, va="top")
    pdf.savefig(fig); plt.close(fig)


def page_gauge(pdf, br, obs, nwm, site, site_name, cats, lid):
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=SURFACE)
    axq = fig.add_axes([0.055, 0.565, 0.90, 0.31])
    axs = fig.add_axes([0.055, 0.105, 0.90, 0.34])

    q = obs.get("00060")
    if q is not None and len(q):
        axq.plot(q.index, q.values, color=OBS_C, lw=2.2, label=f"USGS {site} observed", zorder=5)
    axq.plot(nwm.index.tz_convert(TZ).tz_localize(None), nwm["ol"], color=OL_C, lw=1.6,
             alpha=.9, label="NWM open-loop", zorder=4)
    axq.plot(nwm.index.tz_convert(TZ).tz_localize(None), nwm["aa"], color=AA_C, lw=1.6,
             ls="-.", alpha=.9, label="NWM A&A", zorder=4)
    _thresholds(axq, {rp: br[f"Q{rp}_cfs"] for rp in (10, 50, 100)}, "Q", "{:,.0f} cfs")
    axq.set_ylabel("Discharge (cfs)", fontsize=10)
    axq.set_title("Observed vs modelled discharge", fontsize=12.5, color=INK, loc="left", pad=8)
    axq.legend(loc="upper left", fontsize=8.5, framealpha=.92)

    h = obs.get("00065")
    for name, col in NWS_CATS:
        lvl = cats.get(name)
        if lvl is None:
            continue
        axs.axhline(lvl, color=col, ls="--", lw=1.6, zorder=3)
        hrs = ""
        if h is not None and len(h):
            n = int((h >= lvl).sum())
            hrs = f"  ·  {n * 15 / 60:.0f} h above" if n else "  ·  not reached"
        axs.annotate(f"{name} {lvl:g} ft{hrs}", xy=(0.995, lvl),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=8.5, color=col, fontweight="bold")
    if h is not None and len(h):
        axs.plot(h.index, h.values, color="#14608c", lw=2.2, zorder=4)
        axs.annotate(f"peak {h.max():.2f} ft\n{h.idxmax():%d %b %H:%M}",
                     xy=(h.idxmax(), h.max()), xytext=(12, -18),
                     textcoords="offset points", fontsize=9, color=INK,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=HAIRLINE))
    axs.set_ylabel("Gage height (ft)", fontsize=10)
    top = max([v for v in cats.values()] + ([h.max()] if h is not None and len(h) else [0]))
    axs.set_ylim(top=top * 1.10)
    axs.set_title(f"Observed stage vs NWS flood categories ({lid})",
                  fontsize=12.5, color=INK, loc="left", pad=8)

    for ax in (axq, axs):
        ax.grid(axis="y", ls=":", alpha=.45)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(HAIRLINE)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.tick_params(labelsize=8.5)

    fig.text(0.055, 0.965, f"USGS {site} — {site_name}", fontsize=18,
             fontweight="bold", color=INK, va="top")
    bits = [f"≈70 m downstream of bridge {br['bridge_id']}"]
    if q is not None and len(q):
        bits.append(f"observed peak {q.max():,.0f} cfs "
                    f"({q.max() / br['Q10_cfs'] * 100:.0f}% of Q10)")
    if h is not None and len(h):
        bits.append(f"peak stage {h.max():.2f} ft")
    fig.text(0.055, 0.928, "  ·  ".join(bits), fontsize=10.5, color=INK2, va="top")
    if "major" not in cats:
        fig.text(0.055, 0.038,
                 "No major-flood stage is defined for this gauge, so moderate (red) is the "
                 "highest NWS category it carries. Flood categories are stage-based; the "
                 "alerting thresholds above are discharge return periods — the two are not "
                 "the same scale, which is the point this page makes.",
                 fontsize=8.5, color=MUTED, va="top")
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="29-00151")
    ap.add_argument("--usgs", default="03350700")
    ap.add_argument("--usgs-name", default="Stony Creek near Noblesville, IN")
    ap.add_argument("--lid", default="SCNI3",
                    help="NWS forecast-point id for flood categories")
    ap.add_argument("--day", default=None, help="map day; default = peak open-loop day")
    ap.add_argument("--outdir", default="episode_out/bridge")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    row = cfg[cfg["bridge_id"].astype(str) == args.bridge]
    if row.empty:
        raise SystemExit(f"{args.bridge} not in the monitored configuration")
    br = row.iloc[0]
    comid = int(br["comid"])

    ev = load_events()
    hits = ev[ev["bridge_id"].astype(str) == args.bridge]
    log.info("%s: %d alert event(s) in the episode", args.bridge, len(hits))

    mr = mrms_at(br["lat"], br["lon"])
    nwm = nwm_at(comid)
    tc = int(br["tc_dur_hr"])
    roll = mr.rolling(tc, min_periods=tc).sum()
    log.info("  MRMS peak %d-h accumulation %.2f in (P10 %.2f) — %s",
             tc, roll.max(), br["P10"], "exceeds" if roll.max() >= br["P10"] else "below")
    log.info("  NWM open-loop peak %.0f cfs / A&A %.0f cfs (fires at Q50 = %.0f)",
             nwm["ol"].max(), nwm["aa"].max(), br["Q50_cfs"])

    day = args.day or nwm["ol"].idxmax().tz_convert(TZ).strftime("%Y-%m-%d")
    log.info("  map day: %s (peak open-loop)", day)

    obs = usgs_iv(args.usgs, "2026-08-12", "2026-08-17")
    cats = nwps_categories(args.lid)
    counties, flow = load_counties(), load_flowlines()
    q100 = cfg.dropna(subset=["comid"]).drop_duplicates("comid").set_index("comid")["Q100_cfs"]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        page_maps(pdf, br, day, cfg, counties, flow, q100, len(hits) > 0)
        page_series(pdf, br, mr, nwm)
        page_gauge(pdf, br, obs, nwm, args.usgs, args.usgs_name, cats, args.lid)
    blob = buf.getvalue()
    fp = out / f"{args.bridge}_report.pdf"
    fp.write_bytes(blob)
    log.info("Wrote %s (%.0f KB)", fp, len(blob) / 1024)
    if not args.no_upload:
        write_bytes(blob, bucket(), ep_key(f"bridge/{fp.name}"),
                    content_type="application/pdf")


if __name__ == "__main__":
    main()
