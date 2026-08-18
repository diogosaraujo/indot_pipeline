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
                    load_counties, load_events, load_flowlines, load_mrms_hour,
                    load_nwm_hour, load_regions, panel_legend_rects, panel_rects,
                    precip_cmap, river_ramp_legend, set_geo)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e04")

CFS = 35.3146667
STATE = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72), name="statewide")


def _frame(ts, extent, day_events, cfg, counties, flow, q100, vmax_p, vmax_q,
           dpi, colors=192, mscale=1.9):
    """One 3-panel frame -> PIL Image."""
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=PANEL_FIG, facecolor=SURFACE, dpi=dpi)
    axes = [fig.add_axes(r) for r in panel_rects()]

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
                ha="center", color=MUTED, fontsize=13)
    # no river network here — this panel is the rainfall field
    draw_counties(ax, counties, lw=0.6, overlay=True)   # restate geography on top
    ax.set_title("MRMS 1-h QPE", fontsize=13, color=INK, loc="left", pad=8)

    # panels 2 & 3 — NWM open-loop and A&A, identical scale
    for ax, col, lbl in ((axes[1], "q_ol_cms", "NWM open-loop (trigger)"),
                         (axes[2], "q_aa_cms", "NWM A&A (with DA)")):
        draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
        if nw is not None and col in nw.columns:
            ratio = (nw[col] * CFS) / q100.reindex(nw.index)
            draw_flowlines(ax, flow, ratio, vmax=vmax_q, lw_base=0.55, lat=la, lon=lo)
        else:
            ax.text(0.5, 0.5, "no NWM this hour", transform=ax.transAxes,
                    ha="center", color=MUTED, fontsize=13)
        draw_counties(ax, counties, lw=0.5, overlay=True)
        ax.set_title(lbl, fontsize=13, color=INK, loc="left", pad=8)

    # bridges triggered at or before this hour, on every panel. Same scale as
    # the static map — the panels are identical in size, so the markers must be
    # too, or the two products disagree about the same event.
    shown = day_events[day_events["first_hour"] <= ts]
    for ax in axes:
        draw_bridges(ax, shown, mscale)
        set_geo(ax, la, lo)

    cb_rect, ramp_rect = panel_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("1-h QPE (in)", fontsize=9, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
    river_ramp_legend(fig, ramp_rect, vmax=vmax_q,
                      label="flow ÷ reach 100-yr Q  (both NWM panels share this scale)")

    fig.text(0.028, 0.975, f"{local:%A %d %B %Y  ·  %H:%M %Z}   —   {extent['name']}",
             fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.028, 0.938, f"{len(shown)} bridge(s) triggered so far today   ·   "
                           f"colour = which product confirms, size = severity tier   ·   "
                           f"scales fixed for all 24 frames",
             fontsize=11.5, color=INK2, va="top")

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
        if nw is not None and "q_ol_cms" in nw.columns:
            r = (nw["q_ol_cms"] * CFS) / q100.reindex(nw.index)
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
        de = (d.sort_values("severity_rp", ascending=False)
              .groupby("bridge_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   map_class=("map_class", "first"),
                   severity_rp=("severity_rp", "max"),
                   first_hour=("valid_hour", "min")).reset_index())

        extents = [STATE]
        if args.regions:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            vp, vq = _scales(day, extent, q100)
            frames = []
            for ts in hour_range(day):
                frames.append(_frame(ts, extent, de, cfg, counties, flow, q100,
                                     vp, vq, args.dpi, args.colors,
                                     args.marker_scale))
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
