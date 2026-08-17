"""e03 — per-day summary PDF: state overview + labeled zoom pages.

One PDF per day:
  page 1     statewide map, that day's triggered bridges, region boxes outlined
  pages 2..n one page per zoom tile, bridges labeled by asset name, with a key
             giving trigger / severity / observed vs threshold

Zoom extents come from the FROZEN catalog (e00), so a region occupies identical
ground every day it appears. A region is TILED when one day puts more than
MAX_LABELS bridges in it — Aug 12 drops 140 into the Whitewater frame, which no
font size rescues.

Writes  episode/report/daily_{YYYY-MM-DD}.pdf   (and to S3)

Usage:
    python episode/e03_daily_report.py
    python episode/e03_daily_report.py --days 2026-08-14 --outdir /tmp/ep
"""
from __future__ import annotations

import argparse
import io
import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from common import (C_CONF, C_OPEN, C_PRECIP, DAYS, INK, INK2, MAX_LABELS,  # noqa: E402
                    MUTED, SEV_SIZE, SURFACE, TIER_C, active_regions, bucket,
                    day_peak_flow, draw_counties, draw_flowlines, ep_key,
                    load_config, load_counties, load_events, load_flowlines,
                    load_regions, place_labels, river_ramp_legend, set_geo,
                    sev_sizes, tile_region)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e03")

# 'unknown' means corroboration could not be assessed. It gets its own neutral
# colour rather than folding into flow_conf: a gap must not read as a
# confirmation, which is how the pre-digest Aug 12 run silently rendered all
# 213 of its flow alerts as A&A-corroborated.
CLASS_C = {"flow_conf": C_CONF, "flow_open": C_OPEN, "precip": C_PRECIP,
           "unknown": "#7d7a72"}


def _day_events(ev: pd.DataFrame, day: str) -> pd.DataFrame:
    """One row per bridge for the day — worst severity, all triggers merged."""
    d = ev[ev["day"] == day].copy()
    if d.empty:
        return d
    d = d.sort_values("severity_rp", ascending=False)
    agg = d.groupby("bridge_id").agg(
        asset=("asset", "first"), lat=("lat", "first"), lon=("lon", "first"),
        comid=("comid", "first"), scour=("scour", "first"),
        severity_rp=("severity_rp", "max"), map_class=("map_class", "first"),
        triggers=("trigger_type", lambda s: "+".join(sorted(set(s)))),
        hours=("valid_hour", "nunique"),
        first_hour=("valid_hour", "min")).reset_index()
    return agg


def _scatter(ax, d: pd.DataFrame, scale=1.0, lw=0.6) -> None:
    """Colour = which product confirms it; SIZE = severity tier."""
    for cls, col in CLASS_C.items():
        s = d[d["map_class"] == cls]
        if not len(s):
            continue
        mark = "^" if cls == "precip" else "o"
        ax.scatter(s["lon"], s["lat"],
                   s=sev_sizes(s["severity_rp"], scale * (1.25 if cls == "precip" else 1.0)),
                   c=col, marker=mark, edgecolors="white", linewidths=lw, zorder=6)


def _state_page(pdf, day, d, cfg, counties, flow, regions, act, peak) -> None:
    fig = plt.figure(figsize=(11.0, 8.5), facecolor=SURFACE)
    ax = fig.add_axes([0.03, 0.05, 0.60, 0.80])
    axk = fig.add_axes([0.66, 0.05, 0.32, 0.80]); axk.axis("off")
    axk.set_xlim(0, 1); axk.set_ylim(0, 1); axk.set_autoscale_on(False)

    draw_counties(ax, counties)
    ratio = None
    if flow is not None and not peak.empty and "q_ol_cms" in peak.columns:
        q100 = cfg.dropna(subset=["comid"]).drop_duplicates("comid").set_index("comid")["Q100_cfs"]
        ratio = (peak["q_ol_cms"] * 35.3146667) / q100.reindex(peak.index)
    draw_flowlines(ax, flow, ratio)
    ax.scatter(cfg["lon"], cfg["lat"], s=0.8, c=MUTED, alpha=0.20, linewidths=0, zorder=4)
    _scatter(ax, d, scale=0.95, lw=0.6)     # state page is a half-page panel

    for rid in act:
        r = regions[rid]
        la, lo = r["lat"], r["lon"]
        ax.add_patch(plt.Rectangle((lo[0], la[0]), lo[1] - lo[0], la[1] - la[0],
                                   fill=False, ec=INK2, lw=1.1, ls=(0, (4, 3)), zorder=7))
        ax.text(lo[1], la[1], f" {rid}", fontsize=9, fontweight="bold", color=INK2,
                va="bottom", ha="left", zorder=8)
    set_geo(ax, (37.72, 41.83), (-88.12, -84.72))

    n = len(d)
    fig.text(0.03, 0.965, f"{pd.Timestamp(day):%A %d %B %Y} — bridge flood alerts",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.925,
             f"{n} bridge(s) triggered   ·   {int(d['scour'].sum())} scour-critical   ·   "
             f"{d['comid'].nunique()} reaches   ·   {len(act)} affected region(s)",
             fontsize=11.5, color=INK2, va="top")

    y = 0.98
    axk.text(0, y, "Regions this day", fontsize=12, fontweight="bold", color=INK, va="top")
    y -= 0.045
    for rid in act:
        r = regions[rid]
        sub = d[(d["lat"].between(*r["lat"])) & (d["lon"].between(*r["lon"]))]
        axk.text(0.01, y, f"{rid}  —  {len(sub)} bridges", fontsize=10,
                 fontweight="bold", color=INK, va="top")
        y -= 0.030
        axk.text(0.05, y, f"{r['span_mi'][0]}×{r['span_mi'][1]} mi   "
                          f"centroid {r['centroid'][0]:.2f}, {r['centroid'][1]:.2f}",
                 fontsize=8.5, color=MUTED, va="top")
        y -= 0.042
    y -= 0.02
    axk.text(0, y, "Trigger / confirmation", fontsize=12, fontweight="bold",
             color=INK, va="top")
    y -= 0.05
    for cls, col, mark, lbl in (("flow_conf", C_CONF, "o", "Flow — A&A corroborates"),
                                ("flow_open", C_OPEN, "o", "Flow — open-loop only"),
                                ("precip", C_PRECIP, "^", "Precipitation"),
                                ("unknown", CLASS_C["unknown"], "o",
                                 "Flow — corroboration unassessed")):
        cnt = int((d["map_class"] == cls).sum())
        if cls == "unknown" and not cnt:
            continue                       # only shown when the gap actually exists
        axk.plot([0.025], [y - 0.012], marker=mark, ms=8, color=col, mec="white", mew=.9)
        axk.text(0.075, y, f"{lbl}  ({cnt})", fontsize=9.5, color=INK2, va="top")
        y -= 0.045
    y -= 0.015
    axk.text(0, y, "Severity = marker size", fontsize=12, fontweight="bold",
             color=INK, va="top")
    y -= 0.055
    for rp in (100, 50, 10):
        c = int((d["severity_rp"] == rp).sum())
        axk.scatter([0.030], [y - 0.008], s=SEV_SIZE[rp] * 0.55, c=INK2,
                    edgecolors="white", linewidths=0.6)
        axk.text(0.095, y, f"{rp}-yr  ({c})", fontsize=9.5, color=INK2, va="top")
        y -= 0.048
    y -= 0.015
    axk.text(0, y, "Colour is which product confirms the alert;\nsize is the tier it reached — "
                   "one channel\ncannot carry both.", fontsize=8.5, color=MUTED, va="top")
    y -= 0.10
    axk.text(0, y, "Gray reaches carry no LP3 fit, so they\nhave no ratio to shade.",
             fontsize=8.5, color=MUTED, va="top")
    river_ramp_legend(fig, [0.695, 0.10, 0.20, 0.016])
    pdf.savefig(fig); plt.close(fig)


def _zoom_page(pdf, day, rid, tile, d, cfg, counties, flow, peak, ratio) -> None:
    la, lo = tile["lat"], tile["lon"]
    sub = d[(d["lat"].between(*la)) & (d["lon"].between(*lo))].copy()
    if sub.empty:
        return
    fig = plt.figure(figsize=(11.0, 8.5), facecolor=SURFACE)
    ax = fig.add_axes([0.03, 0.05, 0.63, 0.82])
    axk = fig.add_axes([0.69, 0.05, 0.29, 0.82]); axk.axis("off")
    axk.set_xlim(0, 1); axk.set_ylim(0, 1); axk.set_autoscale_on(False)

    draw_counties(ax, counties, lw=0.8)
    draw_flowlines(ax, flow, ratio, lw_base=0.9, lat=la, lon=lo)
    ctx = cfg[(cfg["lat"].between(*la)) & (cfg["lon"].between(*lo))]
    ax.scatter(ctx["lon"], ctx["lat"], s=5, c=MUTED, alpha=0.30, linewidths=0, zorder=4)
    _scatter(ax, sub, scale=1.5, lw=0.8)
    set_geo(ax, la, lo)
    place_labels(ax, sub, "asset", fontsize=7.0)
    river_ramp_legend(fig, [0.695, 0.055, 0.19, 0.014])

    part = f"  (part {tile['part']} of {tile['parts']})" if tile["parts"] > 1 else ""
    fig.text(0.03, 0.965, f"{pd.Timestamp(day):%a %d %b %Y}  —  region {rid}{part}",
             fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.925, f"{len(sub)} bridge(s)   ·   "
                          f"{round((la[1]-la[0])*69)}×{round((lo[1]-lo[0])*53)} mi frame",
             fontsize=11, color=INK2, va="top")

    y = 0.99
    axk.text(0, y, "Bridges on this page", fontsize=11, fontweight="bold", color=INK, va="top")
    y -= 0.035
    for _, r in sub.sort_values(["severity_rp", "asset"], ascending=[False, True]).iterrows():
        tag = "*" if r["scour"] else ""
        axk.text(0.0, y, f"{str(r['asset'])[:22]}{tag}", fontsize=7.4, color=INK,
                 va="top", family="monospace")
        axk.text(0.60, y, f"{r['triggers'][:6].upper():<6} {int(r['severity_rp']):>3}yr",
                 fontsize=7.4, color=TIER_C.get(int(r["severity_rp"]), INK2),
                 va="top", family="monospace")
        y -= 0.0225
        if y < 0.02:
            break
    axk.text(0, max(y - 0.02, 0.005), "* scour-critical", fontsize=7.5, color=MUTED, va="top")
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    ap.add_argument("--outdir", default="episode_out")
    ap.add_argument("--max-labels", type=int, default=MAX_LABELS)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    ev, cfg = load_events(), load_config()
    counties, flow, regions = load_counties(), load_flowlines(), load_regions()
    q100 = cfg.dropna(subset=["comid"]).drop_duplicates("comid").set_index("comid")["Q100_cfs"]

    for day in args.days:
        d = _day_events(ev, day)
        if d.empty:
            log.info("%s: no alerts, skipped", day); continue
        peak = day_peak_flow(day)
        ratio = None
        if not peak.empty and "q_ol_cms" in peak.columns:
            ratio = (peak["q_ol_cms"] * 35.3146667) / q100.reindex(peak.index)
        act = [r for r in active_regions(regions, day)
               if len(d[(d["lat"].between(*regions[r]["lat"]))
                        & (d["lon"].between(*regions[r]["lon"]))])]

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            _state_page(pdf, day, d, cfg, counties, flow, regions, act, peak)
            pages = 1
            for rid in act:
                pts = d[(d["lat"].between(*regions[rid]["lat"]))
                        & (d["lon"].between(*regions[rid]["lon"]))]
                for tile in tile_region(regions[rid], pts, args.max_labels):
                    _zoom_page(pdf, day, rid, tile, d, cfg, counties, flow, peak, ratio)
                    pages += 1
        blob = buf.getvalue()
        fp = out / f"daily_{day}.pdf"
        fp.write_bytes(blob)
        log.info("%s: %d bridges, %d region(s), %d pages -> %s (%.0f KB)",
                 day, len(d), len(act), pages, fp, len(blob) / 1024)
        if not args.no_upload:
            write_bytes(blob, bucket(), ep_key(f"report/daily_{day}.pdf"),
                        content_type="application/pdf")


if __name__ == "__main__":
    main()
