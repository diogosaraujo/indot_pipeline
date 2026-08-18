"""e03 — per-day summary PDF: statewide map + a full bridge table.

One PDF per day:
  page 1     statewide map of that day's triggered bridges, region boxes outlined
  pages 2..n the bridge list — asset, lat/lon, county, nearest city, river,
             trigger, severity, observed vs threshold, and A&A corroboration

The labelled zoom maps this replaced could not be made legible: Aug 12 puts 140
bridges on one river corridor, and no combination of tiling, font size, or
leader lines fit that many names on a page. A table has no such limit and
carries strictly more per bridge than a label ever could. The statewide map is
kept because spatial pattern is the one thing a table cannot show.

County / city / river come from e07_bridge_places.py; without it those columns
render blank rather than blocking the report.

Writes  episode/report/daily_{YYYY-MM-DD}.pdf   (and to S3)

Usage:
    python episode/e03_daily_report.py
    python episode/e03_daily_report.py --days 2026-08-12 --outdir /tmp/ep
"""
from __future__ import annotations

import argparse
import io
import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from common import (CLASS_STYLE, DAYS, HAIRLINE, INK, INK2, MUTED,  # noqa: E402
                    SEV_SIZE, SURFACE, TIER_C, active_regions, bucket,
                    day_peak_flow, draw_bridges, draw_counties, draw_flowlines,
                    ep_key, load_config, load_counties, load_events,
                    load_flowlines, load_regions, river_ramp_legend, set_geo)
from monitor_common.s3io import read_parquet, write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e03")

# 'unknown' means corroboration could not be assessed. It gets its own neutral
# colour rather than folding into flow_conf: a gap must not read as a
# confirmation, which is how the pre-digest Aug 12 run silently rendered all
# 213 of its flow alerts as A&A-corroborated.
CLASS_C = {k: v[0] for k, v in CLASS_STYLE.items()}


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
        observed=("observed", "max"), threshold=("threshold", "max"),
        aa_confirms=("aa_confirms", "max"),
        hours=("valid_hour", "nunique"),
        first_hour=("valid_hour", "min")).reset_index()
    agg["observed_ratio"] = agg["observed"] / agg["threshold"]
    return agg


def _scatter(ax, d: pd.DataFrame, scale=1.0, lw=0.6) -> None:
    """Colour = which product confirms it; SIZE = severity tier."""
    draw_bridges(ax, d, scale, lw=lw, zorder=6)


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
    for cls, (col, mark, lbl) in CLASS_STYLE.items():
        cnt = int((d["map_class"] == cls).sum())
        if not cnt:
            continue
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


ROWS_PER_PAGE = 36

# x position and alignment per column. Kept as data so widths can be retuned
# without touching the drawing loop.
COLS = [("Bridge", 0.025, "l"), ("Lat", 0.208, "r"), ("Lon", 0.268, "r"),
        ("County", 0.278, "l"), ("City", 0.372, "l"), ("River", 0.487, "l"),
        ("Trigger", 0.630, "l"), ("Sev", 0.712, "r"), ("Observed", 0.822, "r"),
        ("Threshold", 0.930, "r"), ("A&A", 0.940, "l")]
HA = {"l": "left", "r": "right"}


def _table_pages(pdf, day, d, places) -> int:
    """Paginated bridge list. Replaces the zoom maps: at Aug-12 densities the
    labelled maps could not be made legible, and a table carries strictly more
    per bridge (county, city, river, observed vs threshold) than a label can."""
    d = d.merge(places, on="bridge_id", how="left")
    d = d.sort_values(["severity_rp", "observed_ratio"], ascending=[False, False])
    pages = int(np.ceil(len(d) / ROWS_PER_PAGE)) or 1
    for pg in range(pages):
        chunk = d.iloc[pg * ROWS_PER_PAGE:(pg + 1) * ROWS_PER_PAGE]
        fig = plt.figure(figsize=(11.0, 8.5), facecolor=SURFACE)
        fig.text(0.025, 0.968, f"{pd.Timestamp(day):%A %d %B %Y} — triggered bridges",
                 fontsize=16, fontweight="bold", color=INK, va="top")
        fig.text(0.025, 0.933, f"page {pg + 1} of {pages}   ·   {len(d)} bridge(s)   ·   "
                               "* = scour-critical   ·   city is the NEAREST place, "
                               "distance in miles",
                 fontsize=9.5, color=MUTED, va="top")
        y = 0.900
        for name, x, al in COLS:
            fig.text(x, y, name, fontsize=8.5, fontweight="bold", color=INK2, ha=HA[al])
        fig.lines.append(plt.Line2D([0.025, 0.975], [y - 0.011, y - 0.011],
                                    transform=fig.transFigure, color=HAIRLINE, lw=0.8))
        y -= 0.030
        for _, r in chunk.iterrows():
            unit = "in" if r["triggers"] == "precip" else "cfs"
            fmt = "{:,.2f}" if unit == "in" else "{:,.0f}"
            tier = TIER_C.get(int(r["severity_rp"]), INK)
            aa = "—" if r["triggers"] == "precip" else ("yes" if r["aa_confirms"] else "NO")
            city = str(r["city"])[:13] if pd.notna(r["city"]) else "—"
            if pd.notna(r.get("city_mi")):
                city = f"{city} {r['city_mi']:.0f}mi"
            vals = [
                (str(r["asset"])[:24] + ("*" if r["scour"] else ""), 0.025, "l", INK),
                (f"{r['lat']:.4f}", 0.208, "r", INK2),
                (f"{r['lon']:.4f}", 0.268, "r", INK2),
                (str(r["county"])[:12] if pd.notna(r["county"]) else "—", 0.278, "l", INK2),
                (city, 0.372, "l", INK2),
                (str(r["river"])[:20] if pd.notna(r["river"]) else "unnamed", 0.487, "l",
                 INK2 if pd.notna(r["river"]) else MUTED),
                (r["triggers"][:7].upper(), 0.630, "l", INK2),
                (f"{int(r['severity_rp'])}-yr", 0.712, "r", tier),
                (fmt.format(r["observed"]) + " " + unit, 0.822, "r", INK),
                (fmt.format(r["threshold"]) + " " + unit, 0.930, "r", INK2),
                (aa, 0.940, "l", INK2 if aa != "NO" else CLASS_C["flow_open"]),
            ]
            for txt, x, al, col in vals:
                fig.text(x, y, txt, fontsize=6.8, color=col, va="top",
                         ha=HA[al], family="monospace")
            y -= 0.0235
        pdf.savefig(fig); plt.close(fig)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    ap.add_argument("--outdir", default="episode_out")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    ev, cfg = load_events(), load_config()
    counties, flow, regions = load_counties(), load_flowlines(), load_regions()
    try:
        places = read_parquet(bucket(), ep_key("bridge_places.parquet"))
    except Exception as e:  # noqa: BLE001
        log.warning("bridge_places.parquet missing (%s) — county/city/river will be "
                    "blank. Run e07_bridge_places.py.", e)
        places = pd.DataFrame(columns=["bridge_id", "county", "city", "city_mi", "river"])

    for day in args.days:
        d = _day_events(ev, day)
        if d.empty:
            log.info("%s: no alerts, skipped", day); continue
        peak = day_peak_flow(day)
        act = [r for r in active_regions(regions, day)
               if len(d[(d["lat"].between(*regions[r]["lat"]))
                        & (d["lon"].between(*regions[r]["lon"]))])]

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            _state_page(pdf, day, d, cfg, counties, flow, regions, act, peak)
            pages = 1 + _table_pages(pdf, day, d, places)
        blob = buf.getvalue()
        fp = out / f"daily_{day}.pdf"
        fp.write_bytes(blob)
        log.info("%s: %d bridges, %d pages -> %s (%.0f KB)",
                 day, len(d), pages, fp, len(blob) / 1024)
        if not args.no_upload:
            write_bytes(blob, bucket(), ep_key(f"report/daily_{day}.pdf"),
                        content_type="application/pdf")


if __name__ == "__main__":
    main()
