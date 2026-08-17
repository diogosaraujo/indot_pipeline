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


def render(day, extent, acc, lats, lons, nhr, ratio, de, cfg, counties, flow,
           dpi, outdir, upload):
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=(12.5, 9.0), facecolor=SURFACE, dpi=dpi)
    ax = fig.add_axes([0.035, 0.055, 0.70, 0.80])
    cax = fig.add_axes([0.775, 0.42, 0.020, 0.40])
    axk = fig.add_axes([0.755, 0.055, 0.235, 0.32]); axk.axis("off")
    axk.set_xlim(0, 1); axk.set_ylim(0, 1); axk.set_autoscale_on(False)

    draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    pm = None
    if acc is not None:
        rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
        cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
        if rs.size and cs.size:
            sub = np.ma.masked_less(acc[np.ix_(rs, cs)], 0.05)
            vmax = max(0.5, float(np.nanpercentile(acc[np.ix_(rs, cs)], 99.8)))
            pm = ax.pcolormesh(lons[cs], lats[rs], sub, cmap="YlGnBu", vmin=0,
                               vmax=vmax, shading="nearest", zorder=2, alpha=0.92)
    draw_flowlines(ax, flow, ratio, vmax=1.5, lw_base=0.55, lat=la, lon=lo)

    for cls, c, m, lbl in (("flow_conf", C_CONF, "o", "Flow — A&A corroborates"),
                           ("flow_open", C_OPEN, "o", "Flow — open-loop only"),
                           ("precip", C_PRECIP, "^", "Precipitation")):
        s = de[de["map_class"] == cls]
        if len(s):
            ax.scatter(s["lon"], s["lat"],
                       s=sev_sizes(s["severity_rp"], 1.0 if m == "o" else 1.25),
                       c=c, marker=m, edgecolors="white", linewidths=0.7,
                       zorder=8, label=lbl)
    set_geo(ax, la, lo)

    if pm is not None:
        cb = fig.colorbar(pm, cax=cax)
        cb.set_label("24-h MRMS accumulation (in)", fontsize=9.5, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
    else:
        cax.axis("off")

    fig.text(0.035, 0.968, f"{pd.Timestamp(day):%A %d %B %Y}  —  24-h rainfall "
                           f"and peak streamflow", fontsize=18, fontweight="bold",
             color=INK, va="top")
    fig.text(0.035, 0.928, f"{extent['name']}   ·   {len(de)} bridge(s) triggered   ·   "
                           f"{nhr}/24 MRMS hours available",
             fontsize=11, color=INK2, va="top")

    y = 0.97
    for cls, c, m, lbl in (("flow_conf", C_CONF, "o", "Flow — A&A corroborates"),
                           ("flow_open", C_OPEN, "o", "Flow — open-loop only"),
                           ("precip", C_PRECIP, "^", "Precipitation")):
        n = int((de["map_class"] == cls).sum())
        axk.plot([0.03], [y - 0.012], marker=m, ms=8, color=c, mec="white", mew=.9)
        axk.text(0.11, y, f"{lbl} ({n})", fontsize=8.5, color=INK2, va="top")
        y -= 0.085
    y -= 0.02
    axk.text(0, y, "Severity = marker size", fontsize=9.5, fontweight="bold",
             color=INK, va="top")
    y -= 0.085
    for rp in (100, 50, 10):
        n = int((de["severity_rp"] == rp).sum()) if "severity_rp" in de.columns else 0
        axk.scatter([0.035], [y - 0.010], s=SEV_SIZE[rp] * 0.5, c=INK2,
                    edgecolors="white", linewidths=0.6)
        axk.text(0.11, y, f"{rp}-yr ({n})", fontsize=8.5, color=INK2, va="top")
        y -= 0.078
    river_ramp_legend(fig, [0.775, 0.30, 0.185, 0.014])

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
        ratio = None
        if not peak.empty and "q_ol_cms" in peak.columns:
            ratio = (peak["q_ol_cms"] * CFS) / q100.reindex(peak.index)

        extents = [STATE]
        if not args.only_state:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            render(day, extent, acc, lats, lons, nhr, ratio, de, cfg, counties,
                   flow, args.dpi, out, not args.no_upload)


if __name__ == "__main__":
    main()
