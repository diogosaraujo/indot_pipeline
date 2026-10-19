"""Hazel Dell Pkwy bridge (29-00250) collapse, 15 Aug 2026 — animated hydrology.

Sized for slides: 13 x 6 inches. Satellite basemap, map left, hydrograph right.

What it shows
─────────────
The White River rises past the mouth of Blue Woods Creek, backs up the creek,
and pushes flow toward the quarry west of Hazel Dell Pkwy. The bridge sits on
the creek, not the river.

Real vs illustrative — read before using this anywhere
──────────────────────────────────────────────────────
REAL, from data:
  * basemap — Esri World Imagery tiles, cached under .tilecache
  * geometry — White River, Blue Woods Creek, quarry ponds, tributaries, from
    the local NHDPlus HR geodatabase
  * hydrograph — USGS 03351000 WHITE RIVER NEAR NORA, 4.2 km downstream,
    15-minute discharge and gage height, complete through the event
  * bridge location and its Q50 gate, from the monitor config

ILLUSTRATIVE, not modelled, and labelled as such on the figure:
  * the inundation extent — a stage-scaled wash drawn in the style of a flood
    map. NOT a hydraulic model. A DEM bathtub fill was considered and rejected:
    the quarry is an active pit whose floor changes every year, so a public DEM
    would give false precision in exactly the place the story depends on.
  * the moment backflow starts. No gauge measured the creek; onset is set at
    BACKFLOW_STAGE_FT.

Gauge 03350800 at Conner Prairie, 5.9 km upstream, is deliberately unused: its
record stops at 08-14 15:00Z when it went out mid-flood, so its last reading is
not a peak.

Projection note: everything is drawn in Web Mercator (EPSG:3857) so the vectors
register against the tiles, but Mercator inflates distance by 1/cos(lat) ~ 1.30
here. Every true-metre quantity (buffers, scale bar) is multiplied by MERC
before use, and the nearest-line selection is done in UTM 16N where metres are
metres.

Usage
    python scripts/hazel_dell_collapse_animation.py
    python scripts/hazel_dell_collapse_animation.py --zoom 17 --step 1 --fps 12
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import pathlib
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hazeldell")

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"

# reuse the Esri tile fetcher already in the repo
_spec = importlib.util.spec_from_file_location(
    "map13437963", pathlib.Path(__file__).with_name("map_comid_13437963.py"))
_m13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m13)

BRIDGE_LON, BRIDGE_LAT = -86.07489, 39.93995
METRIC, WEB = 26916, 3857
MERC = 1.0 / math.cos(math.radians(BRIDGE_LAT))   # true metres -> 3857 units
GATE_Q50 = 62_864.0
HALF_W, HALF_H = 1250, 620        # true metres; ~2:1 to match the map axes
BACKFLOW_STAGE_FT = 14.0

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
WATER = "#2a78d6"
FLOOD_1, FLOOD_2 = "#9ec5f4", "#5598e7"
CRITICAL = "#d03b3b"
HOT = "#ff5a4d"                   # reversed-flow arrows, readable on imagery
HALO = [pe.withStroke(linewidth=2.6, foreground="#101010", alpha=0.85)]


def load_site():
    fl = gpd.read_file(RES / "site_flowlines.gpkg")
    wb = gpd.read_file(RES / "site_waterbodies.gpkg")
    ar = gpd.read_file(RES / "site_area.gpkg")
    return fl, wb, ar


def load_hydrograph():
    d = pd.read_parquet(RES / "usgs_event_aug2026.parquet")
    d = d[d.site == "03351000"]
    q = d[d.param == "00060"].set_index("t").val.rename("q_cfs")
    h = d[d.param == "00065"].set_index("t").val.rename("stage_ft")
    out = pd.concat([q, h], axis=1).sort_index().interpolate()
    return out.loc["2026-08-13 06:00":"2026-08-17 00:00"].resample("1h").mean().dropna()


def arrows(ax, line, n, reverse, color, scale):
    L = line.length
    for f in np.linspace(0.06, 0.94, n):
        p0 = line.interpolate((f - 0.014) * L)
        p1 = line.interpolate((f + 0.014) * L)
        a, b = (p1, p0) if reverse else (p0, p1)
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3 * scale,
                                    mutation_scale=13 * scale, alpha=0.97,
                                    path_effects=[pe.withStroke(linewidth=2.4,
                                                                foreground="#101010",
                                                                alpha=0.55)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--out", default=str(RES / "hazel_dell_collapse.gif"))
    args = ap.parse_args()

    fl, wb, ar = load_site()
    hyd = load_hydrograph().iloc[::args.step]
    log.info("%d frames, %s -> %s", len(hyd), hyd.index[0], hyd.index[-1])

    # ── selection in a true-metre CRS ────────────────────────────────────────
    flm, wbm = fl.to_crs(METRIC), wb.to_crs(METRIC)
    br_m = gpd.GeoSeries([Point(BRIDGE_LON, BRIDGE_LAT)], crs=4326).to_crs(METRIC).iloc[0]

    def nearest_merged(gdf, name):
        merged = linemerge(unary_union(gdf[gdf.gnis_name == name].geometry.tolist()))
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        return min(parts, key=lambda g: g.distance(br_m))

    bwc_m, wr_m = nearest_merged(flm, "Blue Woods Creek"), nearest_merged(flm, "White River")
    log.info("creek line %.0f m, %.1f m from the bridge", bwc_m.length, bwc_m.distance(br_m))

    wbb = wbm[wbm.geometry.area > 8000].copy()
    wbb = wbb[wbb.geometry.centroid.x < br_m.x - 60]
    # NHD types pits and ponded channel alike as LakePond; elongation separates
    # them, so the ponded creek corridor is not filled as if it were a pit.
    wbb["elong"] = wbb.geometry.length ** 2 / (4 * np.pi * wbb.geometry.area)
    quarry_m, corridor_m = wbb[wbb.elong <= 6.0], wbb[wbb.elong > 6.0]
    river_poly_m = unary_union(ar.to_crs(METRIC).geometry.tolist())
    view_m = Point(br_m.x, br_m.y).buffer(1).envelope  # replaced below once extent known
    bwc_all_m = unary_union(flm[flm.gnis_name == "Blue Woods Creek"].geometry.tolist())

    def to_web(geom):
        return gpd.GeoSeries([geom], crs=METRIC).to_crs(WEB).iloc[0]

    # ── everything below is Web Mercator, to register against the tiles ──────
    br = gpd.GeoSeries([Point(BRIDGE_LON, BRIDGE_LAT)], crs=4326).to_crs(WEB).iloc[0]
    cx, cy = br.x, br.y
    hw, hh = HALF_W * MERC, HALF_H * MERC
    xlim, ylim = (cx - hw, cx + hw), (cy - hh, cy + hh)

    from shapely.geometry import box as _box
    view_m = _box(br_m.x - HALF_W, br_m.y - HALF_H, br_m.x + HALF_W, br_m.y + HALF_H)
    bwc_line, wr_line = to_web(bwc_m), to_web(wr_m)
    quarry, corridor = quarry_m.to_crs(WEB), corridor_m.to_crs(WEB)
    others = flm[~flm.gnis_name.isin(["White River", "Blue Woods Creek"])].to_crs(WEB)

    ll = gpd.GeoSeries([Point(xlim[0], ylim[0]), Point(xlim[1], ylim[1])],
                       crs=WEB).to_crs(4326)
    img, ext = _m13.fetch_basemap(ll.iloc[0].x, ll.iloc[0].y,
                                  ll.iloc[1].x, ll.iloc[1].y, args.zoom)

    smin, smax = float(hyd.stage_ft.min()), float(hyd.stage_ft.max())

    fig = plt.figure(figsize=(13, 6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.17,
                          left=0.008, right=0.978, top=0.885, bottom=0.085)
    axm = fig.add_subplot(gs[0, 0]); axh = fig.add_subplot(gs[0, 1])

    def draw(i):
        row = hyd.iloc[i]; t = hyd.index[i]
        q, s = float(row.q_cfs), float(row.stage_ft)
        f = (s - smin) / (smax - smin)
        rev = s >= BACKFLOW_STAGE_FT

        axm.clear()
        axm.imshow(img, extent=ext, origin="upper", zorder=0, interpolation="bilinear")
        axm.set_xlim(*xlim); axm.set_ylim(*ylim); axm.set_aspect("equal"); axm.axis("off")

        for mult, col, al in ((1.45, FLOOD_1, 0.20), (0.70, FLOOD_2, 0.27)):
            band = river_poly_m.buffer(18 + 150 * f * mult)
            gpd.GeoSeries([band.intersection(view_m)], crs=METRIC).to_crs(WEB).plot(
                ax=axm, color=col, alpha=al, ec="none", zorder=2)
        if rev:
            up = bwc_all_m.buffer(12 + 70 * f).intersection(view_m)
            gpd.GeoSeries([up], crs=METRIC).to_crs(WEB).plot(
                ax=axm, color=FLOOD_1, alpha=0.30, ec="none", zorder=2)

        if len(corridor):
            corridor.plot(ax=axm, facecolor=WATER, ec="none", alpha=0.75, zorder=3)
        if len(quarry):
            quarry.plot(ax=axm, facecolor=FLOOD_2, ec="white", lw=0.7, alpha=0.55, zorder=3)
        others.plot(ax=axm, color="#cfe4fb", lw=0.7, alpha=0.5, zorder=3)
        gpd.GeoSeries([bwc_line], crs=WEB).plot(ax=axm, color="#bfe0ff",
                                                lw=1.4 + 2.0 * f, alpha=0.9, zorder=4)

        arrows(axm, wr_line, 7, False, "#ffffff", 0.75 + 0.9 * f)
        arrows(axm, bwc_line, 5, rev, HOT if rev else "#ffffff", 0.85 + 0.7 * f)

        axm.plot([cx], [cy], marker="o", ms=12, mfc=CRITICAL, mec="white",
                 mew=2.0, zorder=12)
        axm.annotate("bridge 29-00250\nHazel Dell Pkwy", (cx, cy), xytext=(15, -36),
                     textcoords="offset points", fontsize=9.5, color="white",
                     ha="left", zorder=12, path_effects=HALO)
        for g, lab, dy in ((wr_line, "White River", 0), (bwc_line, "Blue Woods Creek", 10)):
            p = g.interpolate(0.55 if lab.startswith("White") else 0.42, normalized=True)
            axm.annotate(lab, (p.x, p.y), xytext=(0, dy), textcoords="offset points",
                         fontsize=9, color="white", style="italic", zorder=11,
                         path_effects=HALO)
        if len(quarry):
            qc = quarry.geometry.union_all().centroid
            axm.annotate("quarry", (qc.x, qc.y), fontsize=9.5, color="white",
                         ha="center", zorder=11, path_effects=HALO)
        if rev:
            axm.annotate("BACKFLOW — creek reverses toward the quarry",
                         (0.5, 0.045), xycoords="axes fraction", ha="center",
                         fontsize=11.5, color="white", weight="bold", zorder=13,
                         bbox=dict(fc=CRITICAL, ec="white", lw=1.1,
                                   boxstyle="round,pad=0.42", alpha=0.95))
        axm.annotate(f"{t:%d %b %Y  %H:%M} UTC", (0.013, 0.962),
                     xycoords="axes fraction", fontsize=13, color="white",
                     weight="bold", va="top", zorder=13, path_effects=HALO)
        axm.annotate("inundation illustrative, not a hydraulic model  ·  "
                     "imagery © Esri, Maxar", (0.013, 0.022),
                     xycoords="axes fraction", fontsize=7, color="#e8e8e8",
                     zorder=13, path_effects=HALO)
        sb = 500 * MERC
        x0, y0 = xlim[1] - 0.19 * (xlim[1] - xlim[0]), ylim[0] + 0.075 * (ylim[1] - ylim[0])
        axm.plot([x0, x0 + sb], [y0, y0], color="white", lw=2.6, zorder=13,
                 path_effects=HALO)
        axm.annotate("500 m", (x0 + sb / 2, y0), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     color="white", zorder=13, path_effects=HALO)

        axh.clear(); axh.set_facecolor("#fcfcfb")
        axh.grid(True, color=GRID, lw=0.6, ls="-"); axh.set_axisbelow(True)
        for sp in ("top", "right"): axh.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axh.spines[sp].set_color(AXIS); axh.spines[sp].set_linewidth(0.8)
        axh.plot(hyd.index, hyd.q_cfs, color=WATER, lw=2.0)
        axh.fill_between(hyd.index[:i + 1], hyd.q_cfs[:i + 1], color=WATER, alpha=0.22)
        axh.plot([t], [q], "o", ms=7.5, mfc=WATER, mec="white", mew=1.5, zorder=5)
        axh.axhline(GATE_Q50, color=CRITICAL, lw=1.5)
        axh.annotate(f"bridge gate  Q50 = {GATE_Q50:,.0f} cfs", (0.03, GATE_Q50),
                     xycoords=("axes fraction", "data"), xytext=(0, 5),
                     textcoords="offset points", fontsize=8.5, color=CRITICAL)
        axh.set_ylim(0, GATE_Q50 * 1.16)
        axh.set_ylabel("White River discharge (cfs)", fontsize=9, color=INK_2)
        axh.xaxis.set_major_locator(mdates.DayLocator())
        axh.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axh.tick_params(colors=MUTED, labelsize=8.5)
        axh.set_title("USGS 03351000 near Nora, 4.2 km downstream",
                      fontsize=9, color=INK, loc="left")
        axh.annotate(f"{q:,.0f} cfs   stage {s:.1f} ft", (0.97, 0.93),
                     xycoords="axes fraction", ha="right", fontsize=11,
                     color=INK, weight="bold")
        axh.annotate(f"peak {hyd.q_cfs.max():,.0f} cfs = "
                     f"{100*hyd.q_cfs.max()/GATE_Q50:.0f}% of the gate",
                     (0.97, 0.055), xycoords="axes fraction", ha="right",
                     fontsize=9, color=MUTED)

        fig.suptitle("Hazel Dell Pkwy bridge failure — White River backflow into "
                     "Blue Woods Creek, 13–17 Aug 2026",
                     fontsize=13, color=INK, y=0.958)

    out = pathlib.Path(args.out)
    draw(int(np.argmax(hyd.q_cfs.values)))          # still first; see git history
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="#fcfcfb")
    log.info("wrote %s", out.with_suffix(".png"))
    FuncAnimation(fig, draw, frames=len(hyd), interval=1000 / args.fps).save(
        out, writer=PillowWriter(fps=args.fps))
    log.info("wrote %s  (%d frames, 13 x 6 in)", out, len(hyd))


if __name__ == "__main__":
    main()
