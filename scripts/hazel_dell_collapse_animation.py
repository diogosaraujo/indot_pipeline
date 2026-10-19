"""Hazel Dell Pkwy bridge (29-00250) collapse, 15 Aug 2026 — animated hydrology.

What the animation shows
────────────────────────
The White River rises past the mouth of Blue Woods Creek, backs up the creek,
and pushes flow toward the quarry pit west of Hazel Dell Pkwy. The bridge sits
on the creek, not the river.

Real vs illustrative — please read before using this anywhere
─────────────────────────────────────────────────────────────
REAL, from data:
  * all geometry — White River, Blue Woods Creek, the quarry ponds and the
    other tributaries — NHDPlus HR (nhdflowline_in / nhdwaterbody_in / nhdarea_in)
  * the hydrograph — USGS 03351000 WHITE RIVER NEAR NORA, 4.2 km downstream,
    15-minute discharge and gage height, complete through the event
  * the bridge location and its Q50 gate, from the monitor config

ILLUSTRATIVE, not modelled:
  * the inundation extent. It is a stage-scaled wash around the channel, drawn
    in the style of a flood map. It is NOT a hydraulic model and must not be
    read as one. A DEM bathtub fill was considered and rejected: the quarry is
    an active pit whose floor changes every year, so any public DEM would give
    false precision in exactly the place the story depends on.
  * the moment backflow starts. No gauge measured the creek. Onset is set at a
    stated stage (BACKFLOW_STAGE_FT) and labelled on the figure as assumed.

The other gauge, 03350800 Conner Prairie 5.9 km upstream, is deliberately not
used: its record stops at 08-14 15:00Z when it went out mid-flood, so its last
reading is not a peak.

Two findings the figure is built to carry
─────────────────────────────────────────
  * The observed peak at Nora was 34,500 cfs. The bridge's flow trigger is the
    WHITE RIVER's Q50 = 62,864 cfs, so the real flood reached 55% of the gate
    and would never have fired it. Against the NWM-retro LP3 for that reach
    (Q2 20,880 / Q5 32,665) this was roughly a 5-year event.
  * The bridge is 2.7 m from Blue Woods Creek (8.67 km2) and 253 m from the
    White River reach it is gated on — one of the 2,146 NOT_CROSSING bridges.

Usage
    python scripts/hazel_dell_collapse_animation.py
    python scripts/hazel_dell_collapse_animation.py --fps 12 --step 2
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hazeldell")

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(__file__).resolve().parents[1] / "results"
BRIDGE_LON, BRIDGE_LAT = -86.07489, 39.93995
CRS = 26916
GATE_Q50 = 62_864.0          # the bridge's White River flow trigger, cfs
HALF_W, HALF_H = 1150, 820   # metres, view half-extent

# Backflow onset — ASSUMED, not measured. Chosen on the rising limb where the
# river surface would plausibly top the creek mouth; labelled as such.
BACKFLOW_STAGE_FT = 14.0

# dataviz reference palette (light surface)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
LAND = "#f3f1ea"
WATER = "#2a78d6"
FLOOD_1, FLOOD_2 = "#9ec5f4", "#5598e7"
CRITICAL = "#d03b3b"
QUARRY = "#52514e"


def load_site():
    fl = gpd.read_file(SCRATCH / "site_flowlines.gpkg")
    wb = gpd.read_file(SCRATCH / "site_waterbodies.gpkg")
    ar = gpd.read_file(SCRATCH / "site_area.gpkg")
    return fl.to_crs(CRS), wb.to_crs(CRS), ar.to_crs(CRS)


def load_hydrograph():
    d = pd.read_parquet(SCRATCH / "usgs_event_aug2026.parquet")
    d = d[d.site == "03351000"]
    q = d[d.param == "00060"].set_index("t").val.rename("q_cfs")
    h = d[d.param == "00065"].set_index("t").val.rename("stage_ft")
    out = pd.concat([q, h], axis=1).sort_index().interpolate()
    return out.loc["2026-08-13 06:00":"2026-08-17 00:00"].resample("1h").mean().dropna()


def arrows(ax, line, n, reverse, color, scale):
    """Lay n arrowheads along a LineString, pointing down- or up-stream."""
    L = line.length
    for f in np.linspace(0.06, 0.94, n):
        p0 = line.interpolate((f - 0.012) * L)
        p1 = line.interpolate((f + 0.012) * L)
        a, b = (p1, p0) if reverse else (p0, p1)
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0 * scale,
                                    mutation_scale=11 * scale, alpha=0.95))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--step", type=int, default=1, help="use every Nth hour")
    ap.add_argument("--out", default=str(SCRATCH / "hazel_dell_collapse.gif"))
    args = ap.parse_args()

    fl, wb, ar = load_site()
    hyd = load_hydrograph().iloc[::args.step]
    log.info("%d frames, %s -> %s", len(hyd), hyd.index[0], hyd.index[-1])

    bridge = gpd.GeoSeries([Point(BRIDGE_LON, BRIDGE_LAT)], crs=4326).to_crs(CRS).iloc[0]
    cx, cy = bridge.x, bridge.y
    xlim = (cx - HALF_W, cx + HALF_W); ylim = (cy - HALF_H, cy + HALF_H)

    wr = unary_union(fl[fl.gnis_name == "White River"].geometry.tolist())
    bwc = unary_union(fl[fl.gnis_name == "Blue Woods Creek"].geometry.tolist())
    others = fl[~fl.gnis_name.isin(["White River", "Blue Woods Creek"])]
    river_poly = unary_union(ar.geometry.tolist())
    # Quarry water only: ponds west of the parkway. NHD has no polygon for the
    # dry excavated pit, so only the ponded part of the quarry can be drawn.
    wbb = wb[wb.geometry.area > 8000].copy()
    wbb = wbb[wbb.geometry.centroid.x < bridge.x - 60]
    # Split compact pits from elongated water. NHD types both as LakePond, but a
    # polygon with elongation 28 is the ponded creek corridor, not a pit, and
    # filling it dark made it read as quarry right across the backflow path.
    wbb["elong"] = wbb.geometry.length ** 2 / (4 * np.pi * wbb.geometry.area)
    quarry = wbb[wbb.elong <= 6.0]
    corridor = wbb[wbb.elong > 6.0]

    # Merge the creek's segments into continuous lines and take the one that
    # actually passes the bridge. Picking the LONGEST segment gave a 477 m
    # ArtificialPath 281 m away instead of the 75 m StreamRiver 2.7 m away.
    def nearest_merged(name):
        merged = linemerge(unary_union(fl[fl.gnis_name == name].geometry.tolist()))
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        return min(parts, key=lambda g: g.distance(bridge))

    bwc_line = nearest_merged("Blue Woods Creek")
    wr_line = nearest_merged("White River")
    log.info("creek line %.0f m long, %.1f m from the bridge; river line %.0f m",
             bwc_line.length, bwc_line.distance(bridge), wr_line.length)

    qmin, qmax = float(hyd.q_cfs.min()), float(hyd.q_cfs.max())
    smin, smax = float(hyd.stage_ft.min()), float(hyd.stage_ft.max())

    fig = plt.figure(figsize=(12.6, 7.4), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.45, 1.0], height_ratios=[1.0, 0.52],
                          wspace=0.14, hspace=0.26,
                          left=0.015, right=0.975, top=0.90, bottom=0.07)
    axm = fig.add_subplot(gs[:, 0]); axh = fig.add_subplot(gs[0, 1]); axt = fig.add_subplot(gs[1, 1])

    def draw(i):
        row = hyd.iloc[i]; t = hyd.index[i]
        q, s = float(row.q_cfs), float(row.stage_ft)
        f = (s - smin) / (smax - smin)                 # 0..1 stage fraction
        reversed_now = s >= BACKFLOW_STAGE_FT

        # ── map ──────────────────────────────────────────────────────────────
        axm.clear(); axm.set_facecolor(LAND)
        axm.set_xlim(*xlim); axm.set_ylim(*ylim); axm.set_aspect("equal"); axm.axis("off")

        # illustrative inundation: two stage-scaled washes around the channel
        for mult, col, al in ((2.4, FLOOD_1, 0.40), (1.25, FLOOD_2, 0.45)):
            band = river_poly.buffer(18 + 190 * f * mult)
            gpd.GeoSeries([band], crs=CRS).plot(ax=axm, color=col, alpha=al,
                                                ec="none", zorder=1)
        if reversed_now:                                # backwater up the creek
            up = bwc.buffer(12 + 95 * f)
            gpd.GeoSeries([up], crs=CRS).plot(ax=axm, color=FLOOD_1, alpha=0.5,
                                              ec="none", zorder=1)

        gpd.GeoSeries([river_poly], crs=CRS).plot(ax=axm, color=WATER, alpha=0.95,
                                                  ec="none", zorder=3)
        others.plot(ax=axm, color=WATER, lw=0.8, alpha=0.45, zorder=3)
        gpd.GeoSeries([bwc], crs=CRS).plot(ax=axm, color=WATER,
                                           lw=1.6 + 2.2 * f, zorder=4)
        if len(corridor):
            corridor.plot(ax=axm, facecolor=WATER, ec="none", alpha=0.9, zorder=4)
        quarry.plot(ax=axm, facecolor=QUARRY, ec="none", alpha=0.80, zorder=5)

        arrows(axm, wr_line, 7, False, "#ffffff", 0.75 + 0.9 * f)
        arrows(axm, bwc_line, 5, reversed_now,
               CRITICAL if reversed_now else "#ffffff", 0.8 + 0.7 * f)

        axm.plot([cx], [cy], marker="o", ms=11, mfc=CRITICAL, mec="white",
                 mew=1.8, zorder=12)
        axm.annotate("bridge 29-00250\nHazel Dell Pkwy", (cx, cy),
                     xytext=(14, -34), textcoords="offset points", fontsize=9,
                     color=INK, zorder=12, ha="left",
                     bbox=dict(fc="white", ec=GRID, lw=0.6, boxstyle="round,pad=0.3"))
        axm.annotate("White River", (wr_line.interpolate(0.62, normalized=True).x,
                                     wr_line.interpolate(0.62, normalized=True).y),
                     fontsize=9, color="white", style="italic", zorder=11)
        axm.annotate("Blue Woods Creek", (bwc_line.interpolate(0.45, normalized=True).x,
                                          bwc_line.interpolate(0.45, normalized=True).y),
                     xytext=(0, 9), textcoords="offset points",
                     fontsize=8.5, color=INK_2, style="italic", zorder=11)
        if len(quarry):
            qc = quarry.geometry.union_all().centroid
            axm.annotate("quarry ponds", (qc.x, qc.y), fontsize=8.5, color="white",
                         ha="center", zorder=11)
        if reversed_now:
            axm.annotate("BACKFLOW — creek reverses toward the quarry",
                         (0.5, 0.045), xycoords="axes fraction", ha="center",
                         fontsize=11, color=CRITICAL, weight="bold", zorder=13,
                         bbox=dict(fc="white", ec=CRITICAL, lw=1.0,
                                   boxstyle="round,pad=0.4"))
        axm.annotate(f"{t:%d %b %Y  %H:%M} UTC", (0.015, 0.965),
                     xycoords="axes fraction", fontsize=12, color=INK,
                     weight="bold", va="top", zorder=13)
        axm.annotate("inundation is illustrative, not a hydraulic model",
                     (0.015, 0.018), xycoords="axes fraction", fontsize=7.5,
                     color=MUTED, zorder=13)
        sb = 500
        axm.plot([xlim[0]+120, xlim[0]+120+sb], [ylim[0]+90]*2, color=INK_2, lw=2.2, zorder=13)
        axm.annotate("500 m", (xlim[0]+120+sb/2, ylim[0]+110), ha="center",
                     fontsize=8, color=INK_2, zorder=13)

        # ── hydrograph ───────────────────────────────────────────────────────
        axh.clear(); axh.set_facecolor("#fcfcfb")
        axh.grid(True, color=GRID, lw=0.6, ls="-"); axh.set_axisbelow(True)
        for sp in ("top", "right"): axh.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axh.spines[sp].set_color(AXIS); axh.spines[sp].set_linewidth(0.8)
        axh.plot(hyd.index, hyd.q_cfs, color=WATER, lw=1.9)
        axh.fill_between(hyd.index[:i+1], hyd.q_cfs[:i+1], color=WATER, alpha=0.22)
        axh.plot([t], [q], "o", ms=7, mfc=WATER, mec="white", mew=1.4, zorder=5)
        axh.axhline(GATE_Q50, color=CRITICAL, lw=1.4, ls="-")
        axh.annotate(f"bridge gate  Q50 = {GATE_Q50:,.0f} cfs", (0.03, GATE_Q50),
                     xycoords=("axes fraction", "data"), xytext=(0, 5),
                     textcoords="offset points", fontsize=8, color=CRITICAL)
        axh.set_ylim(0, GATE_Q50 * 1.18)
        axh.set_ylabel("White River discharge (cfs)", fontsize=9, color=INK_2)
        axh.xaxis.set_major_locator(mdates.DayLocator())
        axh.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axh.xaxis.set_minor_locator(mdates.HourLocator(byhour=(12,)))
        axh.tick_params(colors=MUTED, labelsize=8)
        axh.set_title("USGS 03351000 near Nora, 4.2 km downstream",
                      fontsize=9, color=INK, loc="left")
        axh.annotate(f"{q:,.0f} cfs   stage {s:.1f} ft", (0.97, 0.93),
                     xycoords="axes fraction", ha="right", fontsize=10,
                     color=INK, weight="bold")

        # ── text panel ───────────────────────────────────────────────────────
        axt.clear(); axt.axis("off")
        pk = hyd.q_cfs.max()
        lines = [
            ("Observed peak", f"{pk:,.0f} cfs  =  {100*pk/GATE_Q50:.0f}% of the gate"),
            ("Return period", "~5-year on the NWM-retro LP3 for this reach"),
            ("Bridge sits on", "Blue Woods Creek, 8.7 km2 — 2.7 m away"),
            ("Monitor gates it on", "White River COMID 18476771 — 253 m away"),
            ("Would it have fired?", "No. The river never reached Q50."),
        ]
        axt.annotate("Why no flow alert was possible", (0.0, 1.0),
                     xycoords="axes fraction", fontsize=10, color=INK,
                     weight="bold", va="top")
        for k, (lab, val) in enumerate(lines):
            y = 0.80 - k * 0.175
            axt.annotate(lab, (0.0, y), xycoords="axes fraction", fontsize=8.5,
                         color=MUTED, va="top")
            axt.annotate(val, (0.0, y - 0.072), xycoords="axes fraction",
                         fontsize=9, color=INK, va="top")

        fig.suptitle("Hazel Dell Pkwy bridge failure — White River backflow into Blue Woods Creek, "
                     "13–17 Aug 2026", fontsize=13.5, color=INK, y=0.965)

    out = pathlib.Path(args.out)
    # Still first: rendering it after anim.save left the previous frame's text
    # ghosted under the new one.
    draw(int(np.argmax(hyd.q_cfs.values)))
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor="#fcfcfb")
    log.info("wrote %s", out.with_suffix(".png"))
    anim = FuncAnimation(fig, draw, frames=len(hyd), interval=1000 / args.fps)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    log.info("wrote %s  (%d frames)", out, len(hyd))


if __name__ == "__main__":
    main()
