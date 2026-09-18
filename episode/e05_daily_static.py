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
from common import (CLASS_STYLE, DAYS, INK, INK2, MUTED, PANEL_FIG,  # noqa: E402
                    PRECIP_ALPHA, SEV_SIZE, SURFACE, active_regions, bucket,
                    day_accum, day_peak_flow, draw_bridges, draw_counties,
                    draw_county_outline, draw_flowlines, draw_roads, ep_key,
                    load_config, load_counties, load_counties_named,
                    load_events, load_flowlines, load_regions,
                    load_roads, panel_legend_rects, panel_rects, precip_cmap,
                    river_ramp_legend, set_geo)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e05")

CFS = 35.3146667
STATE = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72), name="statewide")

# Same extents e04 animates, so a GIF and this plate land on identical ground.
# See e04_animations.py for how each box was sized.
FOCUSES = {
    "carmel": dict(lat=(39.94, 40.12), lon=(-86.26, -85.99),
                   name="Carmel - Cool Creek"),
    # mscale per-extent; see e04_animations.py for why statewide keeps 1.9.
    "wayne": dict(lat=(39.673, 40.051), lon=(-85.56, -84.78),
                  name="Wayne County", roads=True, highlight="I-70",
                  county="Wayne", mscale=0.8),
    # Zoom on the reported I-70 washout near Centerville; see e04_animations.py
    # for how the box was derived from the article's mile marker and distance.
    # East edge stops at the Indiana line; see e04_animations.py for why.
    "i70": dict(lat=(39.797, 39.923), lon=(-85.07, -84.81),
                name="I-70 at Centerville", roads=True, highlight="I-70",
                county="Wayne", mscale=1.1),
}


def render(day, extent, acc, lats, lons, nhr, ratio_ol, ratio_aa, de, cfg,
           counties, flow, roads, cnamed, dpi, outdir, upload, mscale=1.9):
    """Three equal panels: rainfall, then the two NWM products side by side.

    Overlaying rainfall and streamflow on one map made each harder to read, and
    it also hid the open-loop/A&A disagreement, which is the whole reason both
    products are carried. Separate panels let the eye compare them directly.
    """
    la, lo = extent["lat"], extent["lon"]
    # Clip the events to the frame BEFORE anything counts them. Every number on
    # this figure — the subtitle and both keys — is derived from `de`, and
    # set_geo only crops what is drawn, not what is counted. Unclipped, a
    # focused extent printed "1097 bridge(s) triggered" beside a map showing 57.
    de = de[de["lat"].between(la[0], la[1]) & de["lon"].between(lo[0], lo[1])]
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
            pm = ax.pcolormesh(lons[cs], lats[rs], sub, cmap=precip_cmap(),
                               vmin=0.05, vmax=vmax, shading="nearest",
                               zorder=2, alpha=PRECIP_ALPHA)
    # no river network here — this panel is the rainfall field, and the NWM
    # channels belong to the two panels that actually encode flow
    draw_counties(ax, counties, lw=0.6, overlay=True)   # restate geography on top
    # zorder 2.5: above the rainfall mesh so the roads stay readable over it.
    if extent.get("roads"):
        draw_roads(ax, roads, lat=la, lon=lo, zorder=2.5,
                   highlight=extent.get("highlight"))
    if extent.get("county"):
        draw_county_outline(ax, cnamed, extent["county"], zorder=7)
    draw_bridges(ax, de, mscale); set_geo(ax, la, lo)
    ax.set_title("24-h MRMS accumulation", fontsize=16, color=INK, loc="left", pad=8)

    # panel 2 — peak NWM, from A&A (the assimilated product). ratio_ol is still
    # computed upstream and still drives nothing here on purpose: the open-loop
    # disagreement now lives in the marker classes below, not in a second map.
    ax = axes[1]
    draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    if ratio_aa is None:
        ax.text(0.5, 0.5, "product unavailable", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    else:
        draw_flowlines(ax, flow, ratio_aa, vmax=1.5, lw_base=0.55, lat=la, lon=lo)
    draw_counties(ax, counties, lw=0.5, overlay=True)
    # Below the hot reaches at zorder 3, so roads never bury the flood signal.
    if extent.get("roads"):
        draw_roads(ax, roads, lat=la, lon=lo, zorder=2.5,
                   highlight=extent.get("highlight"))
    if extent.get("county"):
        draw_county_outline(ax, cnamed, extent["county"], zorder=7)
    draw_bridges(ax, de, mscale); set_geo(ax, la, lo)
    ax.set_title("NWM", fontsize=16, color=INK, loc="left", pad=8)

    # legends: rainfall under panel 1, one shared streamflow ramp under 2 & 3
    cb_rect, ramp_rect = panel_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("24-h accumulation (in)", fontsize=13, color=INK2)
        cb.ax.tick_params(labelsize=12, colors=INK2)
    # Only one NWM panel now, so the ramp no longer claims to be shared.
    river_ramp_legend(fig, ramp_rect, label="peak flow ÷ reach 100-yr Q")

    # Title, subtitle and the two keys each get their OWN row. At 13.3 in wide
    # the title alone runs to ~0.74 of the canvas, so a key anchored on the same
    # line lands on top of it — which is exactly what happened when these were
    # sized for the 24 in canvas.
    fig.text(0.030, 0.978, f"{pd.Timestamp(day):%A %d %B %Y}",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.030, 0.905, f"{extent['name']}   ·   {len(de)} bridge(s) triggered",
             fontsize=13, color=INK2, va="top")

    # Marker key. This used hand-placed x offsets (0.545, then += 0.115) tuned
    # to the old 24-inch canvas; on a 13.3-inch slide with 12 pt text those
    # labels run off the right edge and collide with each other. fig.legend
    # measures the rendered text and lays it out itself, so the key survives
    # both the narrower canvas and any future font change.
    handles = []
    for cls, (c, m, lbl) in CLASS_STYLE.items():
        n = int((de["map_class"] == cls).sum())
        if not n:
            continue
        handles.append(Line2D([], [], marker=m, linestyle="", markerfacecolor=c,
                              markeredgecolor="white", markeredgewidth=1.0,
                              markersize=11, label=f"{lbl} ({n})"))
    if handles:
        # Row 2, beside the SUBTITLE. This key is the wide one (~9.3 in for
        # three classes); the title is ~4.4 in, so pairing them overflowed the
        # 13.3 in canvas. It fits beside the much shorter subtitle instead.
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(0.985, 0.908), ncol=len(handles),
                   frameon=False, fontsize=12.5, handletextpad=0.5,
                   columnspacing=1.5)

    # Severity key, same treatment. markersize is the marker DIAMETER, so take
    # sqrt of the point^2 areas in SEV_SIZE to keep the key honest about the
    # ratios it is showing.
    sev = []
    for rp in (10, 50, 100):
        n = int((de["severity_rp"] == rp).sum())
        sev.append(Line2D([], [], marker="o", linestyle="", markerfacecolor=INK2,
                          markeredgecolor="white", markeredgewidth=0.8,
                          markersize=np.sqrt(SEV_SIZE[rp] * mscale) * 0.95,
                          label=f"{rp}-yr ({n})"))
    # Row 1, beside the TITLE — this key is the narrow one (~4.7 in), so it is
    # the one that fits up there.
    fig.legend(handles=sev, loc="upper right", bbox_to_anchor=(0.985, 0.998),
               ncol=3, frameon=False, fontsize=12.5, handletextpad=0.5,
               columnspacing=1.5, title="severity = size",
               title_fontsize=12.5, alignment="left")

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
    ap.add_argument("--marker-scale", type=float, default=1.9,
                    help="bridge symbol size multiplier (severity sets the tier)")
    ap.add_argument("--only-state", action="store_true")
    ap.add_argument("--focus", choices=sorted(FOCUSES),
                    help="render ONLY the named focus extent")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    ev, cfg = load_events(), load_config()
    counties, flow, regions = load_counties(), load_flowlines(), load_regions()
    roads = load_roads() if args.focus else None
    cnamed = load_counties_named() if args.focus else None
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

        extents = [FOCUSES[args.focus]] if args.focus else [STATE]
        if not args.focus and not args.only_state:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            render(day, extent, acc, lats, lons, nhr, ratio_ol, ratio_aa, de,
                   cfg, counties, flow, roads, cnamed, args.dpi, out,
                   not args.no_upload,
                   extent.get("mscale", args.marker_scale))


if __name__ == "__main__":
    main()
