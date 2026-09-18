"""e04 — hourly 3-panel animations: MRMS | NWM open-loop | NWM A&A.

One statewide GIF per day; --regions adds one per active zoom region. Same
panel layout as the Lanesville event figure:
precipitation on the left, then the two NWM products side by side so the
open-loop/A&A divergence is visible frame by frame rather than only in summary.

Animations use the WHOLE region extent, never the label tiles — they show
fields, and tiling would just chop the storm in half.

Colour scales are fixed across all 24 frames of a GIF (and shared by the two
NWM panels) so motion reads as the storm moving, not the legend rescaling.

Writes  episode/anim/{day}_{extent}.gif   (and to S3)

Usage:
    python episode/e04_animations.py
    python episode/e04_animations.py --days 2026-08-14 --regions --fps 3
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
from PIL import Image  # noqa: E402

from common import (DAYS, INK, INK2, MUTED, PANEL_FIG, PRECIP_ALPHA,  # noqa: E402
                    SURFACE, TZ, active_regions, bucket, draw_bridges,
                    draw_counties, draw_flowlines, ep_key, hour_range, load_config,
                    draw_county_outline, draw_roads, load_counties,
                    load_counties_named, load_events, load_flowlines,
                    load_mrms_hour, load_nwm_hour, load_regions, load_roads,
                    panel_legend_rects, panel_rects,
                    precip_cmap, river_ramp_legend, set_geo)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e04")

CFS = 35.3146667
STATE = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72), name="statewide")

# Cool Creek through Westfield and Carmel, plus the White River corridor just
# east of it. The Cool Creek bridges are the 10 the INDOT inventory marks
# "LEGAL DRAIN: YES - COOL CREEK DRAIN" (29-00137/145/146/153N/153S/195/317/
# 318/319/329), which sit at lon -86.116..-86.133 — NOT where the road name
# would suggest, and no road name exists in any table we hold. The box is
# widened east to the White River because that is where this episode's alerts
# actually are; a box tight on Cool Creek renders with no bridge markers at all.
# ASCII only: the name becomes the output filename via
# tag = name.replace(" ", "_"), and an em dash there produced
# accum_..._Carmel_—_Cool_Creek.png, which mangles in logs and shells.
#
# Wayne County: 303 monitored bridges, 231 of which alerted in this episode
# (207 on Aug 12), across all three marker classes — far more to show than
# Carmel had. The lon span is set first, at the Indiana line in the east, and
# the lat span derived from it so the box matches the panel's aspect; sized the
# other way round the county sat in a letterboxed strip.
FOCUSES = {
    "carmel": dict(lat=(39.94, 40.12), lon=(-86.26, -85.99),
                   name="Carmel - Cool Creek"),
    # mscale is per-extent, not global: 409 bridges land in this frame and at
    # the statewide 1.9 they merge into blobs. Statewide keeps 1.9 — it was
    # signed off, and one number cannot serve both densities.
    "wayne": dict(lat=(39.673, 40.051), lon=(-85.56, -84.78),
                  name="Wayne County", roads=True, highlight="I-70",
                  county="Wayne", mscale=0.8),
    # The I-70 washout reported on 2026-08-13: westbound lanes lost "just east
    # of the Centerville Road exit at mile marker 146", "about 9 miles west of
    # the Ohio state line". Nine miles west of the Wayne County line (-84.8109)
    # at this latitude puts it near (39.860, -84.981); the box spans Centerville
    # through Richmond to the state line so the reported site, the city and the
    # border are all in frame. Markers run a little larger than the county view
    # because this frame holds half the bridges per square mile.
    # East edge stops AT the state line (-84.81), not past it. Reaching into
    # Ohio bought nothing: counties, flowlines, roads and bridges are all
    # Indiana-only, so the overhang rendered as an empty strip with a hard
    # vertical seam, and it reduced the Wayne outline to a lone vertical line
    # that read as a frame artifact. lat is re-derived to keep the panel aspect.
    "i70": dict(lat=(39.797, 39.923), lon=(-85.07, -84.81),
                name="I-70 at Centerville", roads=True, highlight="I-70",
                county="Wayne", mscale=1.1),
}


def _frame(ts, extent, day_events, cfg, counties, flow, roads, cnamed, q100,
           vmax_p, vmax_q, dpi, colors=192, mscale=1.9):
    """One 2-panel frame -> PIL Image."""
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=PANEL_FIG, facecolor=SURFACE, dpi=dpi)
    # Taller than the shared default. That default reserves header room for the
    # static map's FOUR rows (title, subtitle, class key, severity key); a frame
    # here has only two, so inheriting it left ~1.2 in of dead space under the
    # subtitle and shrank the maps for nothing. Capped so the panel titles still
    # clear the subtitle at 0.925.
    axes = [fig.add_axes(r) for r in panel_rects(y=0.165, h=0.665)]

    mr = load_mrms_hour(ts)
    nw = load_nwm_hour(ts)
    local = ts.tz_convert(TZ)

    # panel 1 — MRMS 1-h QPE
    ax = axes[0]
    draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    pm = None
    if mr is not None:
        arr, lats, lons = mr
        rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
        cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
        if rs.size and cs.size:
            sub = np.ma.masked_less(arr[np.ix_(rs, cs)], 0.01)
            pm = ax.pcolormesh(lons[cs], lats[rs], sub, cmap=precip_cmap(),
                               vmin=0.01, vmax=vmax_p, shading="nearest",
                               zorder=2, alpha=PRECIP_ALPHA)
    else:
        ax.text(0.5, 0.5, "no MRMS this hour", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    # no river network here — this panel is the rainfall field
    draw_counties(ax, counties, lw=0.6, overlay=True)   # restate geography on top
    ax.set_title("MRMS 1-h QPE", fontsize=16, color=INK, loc="left", pad=8)

    # panel 2 — NWM streamflow, coloured from A&A (the assimilated product).
    # The open-loop PANEL is gone from the deck, but the open-loop TRIGGER is
    # not: the markers still separate flow_conf from flow_open, so where the two
    # products disagree is still on the figure — in the layer that names
    # bridges, which is the layer a viewer acts on.
    ax = axes[1]
    draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    if nw is not None and "q_aa_cms" in nw.columns:
        ratio = (nw["q_aa_cms"] * CFS) / q100.reindex(nw.index)
        draw_flowlines(ax, flow, ratio, vmax=vmax_q, lw_base=0.55, lat=la, lon=lo)
    else:
        ax.text(0.5, 0.5, "no NWM this hour", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    draw_counties(ax, counties, lw=0.5, overlay=True)
    ax.set_title("NWM", fontsize=16, color=INK, loc="left", pad=8)

    # bridges triggered at or before this hour, on every panel. Same scale as
    # the static map — the panels are identical in size, so the markers must be
    # too, or the two products disagree about the same event.
    # Clip to the frame before counting: the subtitle reports len(shown), and on
    # a focused extent the unclipped set names the statewide total next to a
    # handful of visible markers.
    inframe = day_events[day_events["lat"].between(la[0], la[1])
                         & day_events["lon"].between(lo[0], lo[1])]
    shown = inframe[inframe["first_hour"] <= ts]
    for ax in axes:
        # zorder 2.5: above the quiet grey rivers and above the rainfall mesh,
        # but BELOW the hot reaches at 3 — roads are the reference the reader
        # orients from, not something that should bury the flood signal.
        if extent.get("roads"):
            draw_roads(ax, roads, lat=la, lon=lo, zorder=2.5,
                       highlight=extent.get("highlight"))
        # Above roads and rivers, below the bridge markers at 8: the county is
        # the frame's subject, but it must not sit on top of the alerts.
        if extent.get("county"):
            draw_county_outline(ax, cnamed, extent["county"], zorder=7)
        draw_bridges(ax, shown, mscale)
        set_geo(ax, la, lo)

    cb_rect, ramp_rect = panel_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("1-h QPE (in)", fontsize=13, color=INK2)
        cb.ax.tick_params(labelsize=12, colors=INK2)
    # Only one NWM panel now, so the ramp no longer claims to be shared.
    river_ramp_legend(fig, ramp_rect, vmax=vmax_q,
                      label="flow ÷ reach 100-yr Q")

    fig.text(0.030, 0.975, f"{local:%A %d %B %Y  ·  %H:%M %Z}   —   {extent['name']}",
             fontsize=21, fontweight="bold", color=INK, va="top")
    fig.text(0.030, 0.925, f"{len(shown)} bridge(s) triggered so far today",
             fontsize=13, color=INK2, va="top")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=colors)


def _scales(day, extent, q100):
    """Fixed colour limits for the whole day, so frames are comparable."""
    pmax, qmax = 0.25, 1.0
    la, lo = extent["lat"], extent["lon"]
    for ts in hour_range(day):
        mr = load_mrms_hour(ts)
        if mr is not None:
            arr, lats, lons = mr
            rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
            cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
            if rs.size and cs.size:
                pmax = max(pmax, float(np.nanpercentile(arr[np.ix_(rs, cs)], 99.9)))
        nw = load_nwm_hour(ts)
        # Scale from the product the panel actually draws. This read q_ol_cms
        # while the panels showed both; now that only A&A is drawn, scaling on
        # open-loop would fix the ramp to a series that appears nowhere on the
        # figure, and the colours would not reach the ends of their own legend.
        if nw is not None and "q_aa_cms" in nw.columns:
            r = (nw["q_aa_cms"] * CFS) / q100.reindex(nw.index)
            v = np.nanpercentile(r.replace([np.inf, -np.inf], np.nan).dropna(), 99.9) \
                if r.notna().any() else 1.0
            qmax = max(qmax, float(v))
    return max(pmax, 0.15), float(np.clip(qmax, 1.0, 3.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    ap.add_argument("--outdir", default="episode_out/anim")
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--dpi", type=int, default=90,
                    help="90 -> ~2160x864 px (panels match the static map's layout). "
                         "Raise for print; GIF size scales roughly with dpi^2.")
    ap.add_argument("--colors", type=int, default=192,
                    help="GIF palette size; lower shrinks files on flat maps")
    ap.add_argument("--marker-scale", type=float, default=1.9,
                    help="bridge symbol size multiplier (severity sets the tier)")
    ap.add_argument("--regions", action="store_true",
                    help="also render one GIF per active zoom region; statewide "
                         "only by default")
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
        de = (d.sort_values("severity_rp", ascending=False)
              .groupby("bridge_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   map_class=("map_class", "first"),
                   severity_rp=("severity_rp", "max"),
                   first_hour=("valid_hour", "min")).reset_index())

        extents = [FOCUSES[args.focus]] if args.focus else [STATE]
        if args.regions and not args.focus:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            vp, vq = _scales(day, extent, q100)
            frames = []
            for ts in hour_range(day):
                frames.append(_frame(ts, extent, de, cfg, counties, flow, roads,
                                     cnamed, q100, vp, vq, args.dpi, args.colors,
                                     extent.get("mscale", args.marker_scale)))
            tag = extent["name"].replace(" ", "_")
            fp = out / f"{day}_{tag}.gif"
            frames[0].save(fp, save_all=True, append_images=frames[1:], loop=0,
                           duration=int(1000 / args.fps), optimize=True)
            mb = fp.stat().st_size / 1e6
            log.info("%s %-16s %d frames -> %s (%.1f MB)", day, tag, len(frames), fp, mb)
            if not args.no_upload:
                write_bytes(fp.read_bytes(), bucket(), ep_key(f"anim/{fp.name}"),
                            content_type="image/gif")


if __name__ == "__main__":
    main()
