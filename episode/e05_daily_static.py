"""e05 — static per-day map: 24-h accumulated rainfall + peak NWM flow.

The companion to the animations: where a GIF shows how the day unfolded, this
is the single frame you put in a report. Two layers on one map:

  * 24-h MRMS accumulation (local day), as a filled field
  * every reach coloured by its PEAK open-loop flow that day, expressed as a
    fraction of that reach's 100-yr Q so the colour means "how close to the
    design flood" rather than "how big is this river"

Rendered statewide plus one panel per zoom region active that day, so it uses
the same extents as everything else in this report.

Writes  episode/static/accum_{day}_{extent}.png   (and to S3)

Usage:
    python episode/e05_daily_static.py
    python episode/e05_daily_static.py --days 2026-08-12 --dpi 200
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (C_CONF, C_OPEN, C_PRECIP, DAYS, INK, INK2, MUTED, SEV_SIZE,  # noqa: E402
                    SURFACE, active_regions, bucket, day_accum, day_peak_flow,
                    draw_counties, draw_flowlines, ep_key, load_config,
                    load_counties, load_events, load_flowlines, load_regions,
                    river_ramp_legend, set_geo, sev_sizes)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e05")

CFS = 35.3146667
STATE = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72), name="statewide")


def _bridges(ax, de) -> None:
    for cls, c, m in (("flow_conf", C_CONF, "o"), ("flow_open", C_OPEN, "o"),
                      ("precip", C_PRECIP, "^")):
        s = de[de["map_class"] == cls]
        if len(s):
            ax.scatter(s["lon"], s["lat"],
                       s=sev_sizes(s["severity_rp"], 1.0 if m == "o" else 1.25),
                       c=c, marker=m, edgecolors="white", linewidths=0.7, zorder=8)


def render(day, extent, acc, lats, lons, nhr, ratio_ol, ratio_aa, de, cfg,
           counties, flow, dpi, outdir, upload):
    """Three equal panels: rainfall, then the two NWM products side by side.

    Overlaying rainfall and streamflow on one map made each harder to read, and
    it also hid the open-loop/A&A disagreement, which is the whole reason both
    products are carried. Separate panels let the eye compare them directly.
    """
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=PANEL_FIG, facecolor=SURFACE, dpi=dpi)
    axes = [fig.add_axes(r) for r in panel_rects()]

    # panel 1 — 24-h MRMS accumulation
    ax = axes[0]
    draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    pm = None
    if acc is not None:
        rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
        cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
        if rs.size and cs.size:
            sub = np.ma.masked_less(acc[np.ix_(rs, cs)], 0.05)
            vmax = max(0.5, float(np.nanpercentile(acc[np.ix_(rs, cs)], 99.8)))
            pm = ax.pcolormesh(lons[cs], lats[rs], sub, cmap="YlGnBu", vmin=0,
                               vmax=vmax, shading="nearest", zorder=2)
    draw_flowlines(ax, flow, None, lw_base=0.35, lat=la, lon=lo)   # channels for context only
    _bridges(ax, de); set_geo(ax, la, lo)
    ax.set_title("24-h MRMS accumulation", fontsize=13, color=INK, loc="left", pad=8)

    # panels 2 & 3 — peak NWM, identical scale so they can be compared
    for ax, ratio, lbl in ((axes[1], ratio_ol, "Peak NWM open-loop (trigger)"),
                           (axes[2], ratio_aa, "Peak NWM A&A (with DA)")):
        draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
        if ratio is None:
            ax.text(0.5, 0.5, "product unavailable", transform=ax.transAxes,
                    ha="center", color=MUTED, fontsize=12)
        else:
            draw_flowlines(ax, flow, ratio, vmax=1.5, lw_base=0.55, lat=la, lon=lo)
        _bridges(ax, de); set_geo(ax, la, lo)
        ax.set_title(lbl, fontsize=13, color=INK, loc="left", pad=8)

    # legends: rainfall under panel 1, one shared streamflow ramp under 2 & 3
    cb_rect, ramp_rect = panel_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("24-h accumulation (in)", fontsize=9, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
    river_ramp_legend(fig, ramp_rect,
                      label="peak flow ÷ reach 100-yr Q  (both NWM panels share this scale)")

    fig.text(0.028, 0.975, f"{pd.Timestamp(day):%A %d %B %Y}  —  24-h rainfall and "
                           f"peak streamflow", fontsize=20, fontweight="bold",
             color=INK, va="top")
    fig.text(0.028, 0.938, f"{extent['name']}   ·   {len(de)} bridge(s) triggered   ·   "
                           f"{nhr}/24 MRMS hours available   ·   "
                           f"gray reaches carry no LP3 fit",
             fontsize=11.5, color=INK2, va="top")

    # marker key, laid out along the header so it steals no map area
    x = 0.545
    for cls, c, m, lbl in (("flow_conf", C_CONF, "o", "A&A corroborates"),
                           ("flow_open", C_OPEN, "o", "open-loop only"),
                           ("precip", C_PRECIP, "^", "precipitation")):
        n = int((de["map_class"] == cls).sum())
        fig.text(x, 0.975, "●" if m == "o" else "▲", fontsize=13, color=c, va="top")
        fig.text(x + 0.013, 0.973, f"{lbl} ({n})", fontsize=10.5, color=INK2, va="top")
        x += 0.115
    x = 0.545
    fig.text(x, 0.938, "severity = size:", fontsize=10.5, color=INK, va="top")
    x += 0.078
    for rp in (10, 50, 100):
        n = int((de["severity_rp"] == rp).sum())
        fig.text(x, 0.938, "•", fontsize=8 + rp / 14.0, color=INK2, va="top")
        fig.text(x + 0.016, 0.938, f"{rp}-yr ({n})", fontsize=10, color=INK2, va="top")
        x += 0.098

    tag = extent["name"].replace(" ", "_")
    fp = pathlib.Path(outdir) / f"accum_{day}_{tag}.png"
    fig.savefig(fp, facecolor=SURFACE)
    plt.close(fig)
    log.info("%s %-16s -> %s (%.1f MB)", day, tag, fp, fp.stat().st_size / 1e6)
    if upload:
        write_bytes(fp.read_bytes(), bucket(), ep_key(f"static/{fp.name}"),
                    content_type="image/png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    ap.add_argument("--outdir", default="episode_out/static")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--only-state", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    ev, cfg = load_events(), load_config()
    counties, flow, regions = load_counties(), load_flowlines(), load_regions()
    q100 = cfg.dropna(subset=["comid"]).drop_duplicates("comid").set_index("comid")["Q100_cfs"]

    for day in args.days:
        d = ev[ev["day"] == day]
        if d.empty:
            log.info("%s: no alerts, skipped", day); continue
        de = (d.sort_values("severity_rp", ascending=False).groupby("bridge_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   map_class=("map_class", "first"),
                   severity_rp=("severity_rp", "max")).reset_index())
        acc, lats, lons, nhr = day_accum(day)
        if nhr < 24:
            log.warning("%s: only %d/24 MRMS hours — accumulation is a partial day",
                        day, nhr)
        peak = day_peak_flow(day)
        ratio_ol = ratio_aa = None
        if not peak.empty:
            if "q_ol_cms" in peak.columns:
                ratio_ol = (peak["q_ol_cms"] * CFS) / q100.reindex(peak.index)
            if "q_aa_cms" in peak.columns:
                ratio_aa = (peak["q_aa_cms"] * CFS) / q100.reindex(peak.index)

        extents = [STATE]
        if not args.only_state:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            render(day, extent, acc, lats, lons, nhr, ratio_ol, ratio_aa, de,
                   cfg, counties, flow, args.dpi, out, not args.no_upload)


if __name__ == "__main__":
    main()
