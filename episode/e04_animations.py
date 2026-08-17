"""e04 — hourly 3-panel animations: MRMS | NWM open-loop | NWM A&A.

One GIF per (day, map extent), where extents are the statewide view plus every
zoom region active that day. Same panel layout as the Lanesville event figure:
precipitation on the left, then the two NWM products side by side so the
open-loop/A&A divergence is visible frame by frame rather than only in summary.

Animations use the WHOLE region extent, never the label tiles — they show
fields, and tiling would just chop the storm in half.

Colour scales are fixed across all 24 frames of a GIF (and shared by the two
NWM panels) so motion reads as the storm moving, not the legend rescaling.

Writes  episode/anim/{day}_{extent}.gif   (and to S3)

Usage:
    python episode/e04_animations.py
    python episode/e04_animations.py --days 2026-08-14 --only-state --fps 3
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
from PIL import Image  # noqa: E402

from common import (C_CONF, C_OPEN, C_PRECIP, DAYS, INK, INK2, MUTED, SURFACE,  # noqa: E402
                    TZ, active_regions, bucket, draw_counties, draw_flowlines,
                    ep_key, hour_range, load_config, load_counties, load_events,
                    load_flowlines, load_mrms_hour, load_nwm_hour, load_regions,
                    set_geo, sev_sizes)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e04")

CFS = 35.3146667
STATE = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72), name="statewide")


def _frame(ts, extent, day_events, cfg, counties, flow, q100, vmax_p, vmax_q, dpi):
    """One 3-panel frame -> PIL Image."""
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=(15.0, 6.0), facecolor=SURFACE, dpi=dpi)
    axes = [fig.add_axes([0.015 + i * 0.330, 0.045, 0.305, 0.80]) for i in range(3)]

    mr = load_mrms_hour(ts)
    nw = load_nwm_hour(ts)
    local = ts.tz_convert(TZ)

    # panel 1 — MRMS 1-h QPE
    ax = axes[0]
    draw_counties(ax, counties, lw=0.4)
    if mr is not None:
        arr, lats, lons = mr
        rs = np.where((lats >= la[0]) & (lats <= la[1]))[0]
        cs = np.where((lons >= lo[0]) & (lons <= lo[1]))[0]
        if rs.size and cs.size:
            sub = np.ma.masked_less(arr[np.ix_(rs, cs)], 0.01)
            ax.pcolormesh(lons[cs], lats[rs], sub, cmap="YlGnBu", vmin=0,
                          vmax=vmax_p, shading="nearest", zorder=2)
    else:
        ax.text(0.5, 0.5, "no MRMS this hour", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=11)
    ax.set_title("MRMS 1-h QPE (in)", fontsize=11, color=INK2, loc="left")

    # panels 2 & 3 — NWM open-loop and A&A, identical scale
    for ax, col, lbl in ((axes[1], "q_ol_cms", "NWM open-loop (trigger)"),
                         (axes[2], "q_aa_cms", "NWM A&A (with DA)")):
        draw_counties(ax, counties, lw=0.4)
        if nw is not None and col in nw.columns:
            ratio = (nw[col] * CFS) / q100.reindex(nw.index)
            draw_flowlines(ax, flow, ratio, vmax=vmax_q, lw_base=0.5, lat=la, lon=lo)
        else:
            ax.text(0.5, 0.5, "no NWM this hour", transform=ax.transAxes,
                    ha="center", color=MUTED, fontsize=11)
        ax.set_title(lbl, fontsize=11, color=INK2, loc="left")

    # bridges triggered at or before this hour, on every panel
    shown = day_events[day_events["first_hour"] <= ts]
    for ax in axes:
        for cls, c, m in (("flow_conf", C_CONF, "o"), ("flow_open", C_OPEN, "o"),
                          ("precip", C_PRECIP, "^")):
            s = shown[shown["map_class"] == cls]
            if len(s):
                ax.scatter(s["lon"], s["lat"], s=sev_sizes(s["severity_rp"], 0.55),
                           c=c, marker=m, edgecolors="white", linewidths=0.5, zorder=8)
        set_geo(ax, la, lo)

    fig.text(0.015, 0.965, f"{local:%A %d %B %Y  ·  %H:%M %Z}   —   {extent['name']}",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.905, f"{len(shown)} bridge(s) triggered so far today   ·   "
                           f"rivers shaded by flow ÷ reach 100-yr Q (cap {vmax_q:g}×)   ·   "
                           f"QPE scale 0–{vmax_p:.2f} in/h",
             fontsize=9.5, color=MUTED, va="top")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=192)


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
    ap.add_argument("--dpi", type=int, default=72)
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
        de = (d.sort_values("severity_rp", ascending=False)
              .groupby("bridge_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   map_class=("map_class", "first"),
                   severity_rp=("severity_rp", "max"),
                   first_hour=("valid_hour", "min")).reset_index())

        extents = [STATE]
        if not args.only_state:
            for rid in active_regions(regions, day):
                r = regions[rid]
                extents.append(dict(lat=tuple(r["lat"]), lon=tuple(r["lon"]),
                                    name=f"region {rid}"))
        for extent in extents:
            vp, vq = _scales(day, extent, q100)
            frames = []
            for ts in hour_range(day):
                frames.append(_frame(ts, extent, de, cfg, counties, flow, q100,
                                     vp, vq, args.dpi))
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
