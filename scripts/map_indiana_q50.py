#!/usr/bin/env python3
"""Statewide map of the 50-year streamflow (Q50) on every Indiana COMID we fit.

The companion to the Atlas 14 P50-24h precipitation plate. That one is a
gridded product and can be drawn as a raster; this one cannot — Q50 exists only
on the reaches we ran the LP3 fit for, so the map is the river network itself,
coloured by its own 50-year discharge.

Two decisions worth stating, because they are what make the plate readable:

  LOG COLOUR SCALE. Q50 runs 0.5 -> 910,000 cfs across the network, six orders
  of magnitude, with a median near 1,900. On a linear plasma ramp every reach
  in Indiana except the Ohio and the lower Wabash collapses to a single colour.
  The ramp is therefore log, spanning 1e2 -> 1e5 cfs, which brackets roughly the
  2nd to the 99th percentile; the tails saturate and the colorbar says so with
  extend arrows.

  WIDTH CARRIES THE VALUE TOO. Colour alone is fragile at 4.95 inches wide and
  dies in greyscale, so line width tracks the same number. It also restores the
  thing a reader expects from a river map: big rivers look big.

Reaches we could not fit are drawn thin grey rather than dropped — an absent
reach reads as "no river here", which is a different and false claim.

Output is sized for the report plate: 4.95 in wide x 5.9 in tall.

Usage:
    python scripts/map_indiana_q50.py
    python scripts/map_indiana_q50.py --no-s3        # skip the S3 upload
    python scripts/map_indiana_q50.py --vmin 100 --vmax 100000
"""
from __future__ import annotations

import argparse
import io
import pathlib

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon, Rectangle

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
CRS = 26916                       # NAD83 / UTM 16N — the right projection for Indiana
FIG_W, FIG_H = 4.95, 5.90         # report plate size, inches

# palette + chrome, shared with monitor_common/maps.py so the plates match
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HAIRLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
LAND, OUTSIDE = "#ffffff", "#e8e7e2"
QUIET = "#b8b6b0"                 # reaches with no usable fit

# Reversed: high flow goes dark, the same direction monitor_common/maps.py uses
# for RIVER_CMAP, so this plate and the operational digest read the same way.
#
# The FULL ramp, yellow end included, to stay identical to the other plates in
# the set. The cost is known and deliberate: reversing drops plasma's brightest
# yellow onto the lowest flows, which are also the thinnest lines, and against
# white land that yellow carries almost no luminance contrast. Nothing in the
# background can fix it — a darker land tint scores WORSE, because the yellow
# outshines any fill mild enough for a light report plate. So the width floor
# below is what keeps the low end visible; do not lower it while this ramp runs
# to yellow.
CMAP = "plasma_r"
M_PER_MI = 1609.344

# Orientation only. Kept few and spread out; a dense gazetteer would compete
# with the network, which is the subject.
CITIES = [
    ("Gary",         -87.346, 41.593, "left",   "bottom"),
    ("South Bend",   -86.250, 41.676, "center", "bottom"),
    ("Fort Wayne",   -85.139, 41.079, "right",  "center"),
    ("Lafayette",    -86.875, 40.417, "right",  "center"),
    ("Muncie",       -85.386, 40.193, "left",   "center"),
    ("Indianapolis", -86.158, 39.768, "center", "top"),
    ("Terre Haute",  -87.414, 39.467, "right",  "center"),
    ("Evansville",   -87.571, 37.977, "left",   "bottom"),
]


def s3_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(
        Bucket=BUCKET, Key=f"{PREFIX}{key}")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def projector():
    from pyproj import Transformer
    t = Transformer.from_crs(4326, CRS, always_xy=True)
    return lambda lon, lat: t.transform(np.asarray(lon, float), np.asarray(lat, float))


def state_outline(counties: pd.DataFrame):
    """Dissolve the county rings into the state boundary.

    The rings ship as flat lon/lat tables, so rebuild polygons and union them.
    Stroking the rings as-is would draw 92 county borders, which the Atlas 14
    plate does not have and which would clutter a map this size.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for _, g in counties.groupby("part_id"):
        if len(g) > 3:
            p = Polygon(np.column_stack([g.lon.to_numpy(), g.lat.to_numpy()]))
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
    return unary_union(polys)


def rings_of(geom):
    """Exterior rings of a (Multi)Polygon as lon/lat arrays."""
    geoms = getattr(geom, "geoms", [geom])
    return [np.asarray(g.exterior.coords) for g in geoms]


def north_arrow(ax, x=0.085, y=0.90, h=0.070) -> None:
    w = h * 0.30
    ax.add_patch(MplPolygon([[x, y], [x - w, y - h], [x, y - h * 0.68]],
                            closed=True, facecolor=INK, edgecolor="none",
                            transform=ax.transAxes, zorder=12))
    ax.add_patch(MplPolygon([[x, y], [x + w, y - h], [x, y - h * 0.68]],
                            closed=True, facecolor=SURFACE, edgecolor=INK,
                            linewidth=0.6, transform=ax.transAxes, zorder=12))
    ax.text(x, y + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, weight="bold", color=INK, zorder=12)


def scale_bar(fig, x0: float, y: float, frac_per_m: float,
              miles=(0, 25, 50), h: float = 0.010) -> None:
    """Alternating-tone bar in miles, drawn in FIGURE coordinates.

    It lives in the margin beside the map, not inside it: Indiana's southern tip
    runs to the bottom-left corner of the frame, which is the one place a map of
    this state has no room for chrome.
    """
    edges = [x0 + m * M_PER_MI * frac_per_m for m in miles]
    for i in range(len(edges) - 1):
        fig.patches.append(Rectangle(
            (edges[i], y), edges[i + 1] - edges[i], h, transform=fig.transFigure,
            facecolor=(INK if i % 2 == 0 else SURFACE), edgecolor=INK,
            linewidth=0.5, zorder=12))
    for m, e in zip(miles, edges):
        fig.text(e, y + h + 0.005, f"{m}", ha="center", va="bottom",
                 fontsize=6.0, color=INK, zorder=12)
    # Unit below and centred, not trailing the bar: the bar already ends near
    # the right edge of the frame, and a trailing word runs off it.
    fig.text((edges[0] + edges[-1]) / 2, y - 0.007, "Miles", ha="center",
             va="top", fontsize=6.0, color=INK, zorder=12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/indiana_q50_map")
    ap.add_argument("--vmin", type=float, default=1e2)
    ap.add_argument("--vmax", type=float, default=1e5)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--no-s3", action="store_true")
    args = ap.parse_args()

    print("Loading tables from S3 ...")
    lp3 = s3_parquet("monitor/precompute/bridge_comid_lp3.parquet")
    flow = s3_parquet("monitor/assets/flowlines.parquet")
    counties = s3_parquet("monitor/assets/in_counties.parquet")

    q50 = pd.to_numeric(lp3.set_index("comid")["Q50_cfs"], errors="coerce")
    q50 = q50[q50 > 0]
    print(f"  {len(q50):,} COMIDs with a usable Q50   "
          f"(median {q50.median():,.0f} cfs, max {q50.max():,.0f} cfs)")

    proj = projector()
    flow["x"], flow["y"] = proj(flow["lon"].to_numpy(), flow["lat"].to_numpy())

    norm = LogNorm(vmin=args.vmin, vmax=args.vmax)
    cmap = plt.get_cmap(CMAP)

    quiet, hot, hot_c, hot_w, hot_v = [], [], [], [], []
    for (comid, _pid), g in flow.groupby(["comid", "part_id"], sort=False):
        seg = g[["x", "y"]].to_numpy()
        if len(seg) < 2:
            continue
        v = q50.get(comid, np.nan)
        if not np.isfinite(v):
            quiet.append(seg)
            continue
        frac = float(np.clip(norm(v), 0, 1))
        hot.append(seg)
        hot_c.append(cmap(frac))
        # Floor the width harder than the pure value ramp would: with the scale
        # reversed the faintest colour now sits on the THINNEST line, and a
        # 0.22 pt pale stroke on white land is not a stream, it is nothing.
        hot_w.append(0.34 + 1.50 * frac)
        hot_v.append(v)
    print(f"  drawing {len(hot):,} valued reaches, {len(quiet):,} unfitted")

    # Big rivers last, so a trunk is never buried under the tributary that
    # happens to be drawn after it.
    order = np.argsort(np.asarray(hot_v))
    hot = [hot[i] for i in order]
    hot_c = [hot_c[i] for i in order]
    hot_w = [hot_w[i] for i in order]

    state = state_outline(counties)
    minx, miny = proj([state.bounds[0]], [state.bounds[1]])
    maxx, maxy = proj([state.bounds[2]], [state.bounds[3]])
    pad = 6_000.0
    x_lo, x_hi = float(minx[0]) - pad, float(maxx[0]) + pad
    y_lo, y_hi = float(miny[0]) - pad, float(maxy[0]) + pad
    dx, dy = x_hi - x_lo, y_hi - y_lo

    # Size the map axes to the data aspect so it fills the height exactly; the
    # slack that leaves on the right is where the legend goes, the same way the
    # Atlas 14 plate uses it.
    # bottom leaves a strip under the map for the two credit lines; the legend
    # and scale bar derive from w_frac, so they follow this automatically.
    top, bottom = 0.982, 0.032
    h_frac = top - bottom
    w_frac = (h_frac * FIG_H) / (dy / dx) / FIG_W

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=OUTSIDE)
    ax = fig.add_axes([0.012, bottom, w_frac, h_frac])
    ax.set_facecolor(OUTSIDE)

    for ring in rings_of(state):
        rx, ry = proj(ring[:, 0], ring[:, 1])
        ax.fill(rx, ry, facecolor=LAND, edgecolor="none", zorder=1)
        # UNDER the network, not over it. Indiana's southern border IS the Ohio
        # River, so a boundary stroked above the data paints a grey line exactly
        # along the largest reaches on the map and erases them — the Ohio was
        # being drawn at full width in the darkest indigo and still not showing.
        ax.plot(rx, ry, color=BASELINE, linewidth=0.9, zorder=1.6,
                solid_joinstyle="round")

    if quiet:
        ax.add_collection(LineCollection(quiet, colors=QUIET, linewidths=0.22,
                                         zorder=2, rasterized=True))
    ax.add_collection(LineCollection(hot, colors=hot_c, linewidths=hot_w,
                                     zorder=3, rasterized=True,
                                     capstyle="round", joinstyle="round"))

    for name, lon, lat, ha, va in CITIES:
        cx, cy = proj([lon], [lat])
        ax.plot(cx, cy, marker="o", ms=2.3, mfc=SURFACE, mec=INK, mew=0.6,
                zorder=9, linestyle="none")
        off = {"left": 3.5, "right": -3.5, "center": 0}[ha]
        voff = {"bottom": 3.0, "top": -3.0, "center": 0}[va]
        ax.annotate(name, xy=(cx[0], cy[0]), xytext=(off, voff),
                    textcoords="offset points", ha=ha, va=va, fontsize=6.3,
                    color=INK, zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    north_arrow(ax)

    # ── chrome, in the slack the state's aspect leaves on the right ──────────
    lx = 0.012 + w_frac + 0.012
    lw_ = max(0.06, 1.0 - lx - 0.014)
    box_y, box_h = 0.105, 0.315
    fig.patches.append(Rectangle((lx, box_y), lw_, box_h, transform=fig.transFigure,
                                 facecolor=SURFACE, edgecolor=BASELINE,
                                 linewidth=0.7, zorder=11))
    fig.text(lx + 0.035, box_y + box_h - 0.022, "NWM Q50", fontsize=7.8,
             weight="bold", color=INK, va="top", zorder=12)
    fig.text(lx + 0.035, box_y + box_h - 0.049, "(cfs)", fontsize=7.8,
             color=INK, va="top", zorder=12)

    cax = fig.add_axes([lx + 0.048, box_y + 0.040, 0.040, box_h - 0.125])
    cax.set_zorder(12)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      extend="both", extendfrac=0.045)
    cb.set_ticks([1e2, 1e3, 1e4, 1e5])
    cb.set_ticklabels(["100", "1,000", "10,000", "100,000"])
    # A log colorbar volunteers a minor tick per decade step; at this size that
    # is a rash of hairlines against the ramp, saying nothing the labels do not.
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.3, length=2, pad=1.8, colors=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(BASELINE)

    scale_bar(fig, lx + 0.014, box_y - 0.048, w_frac / dx)

    fig.text(0.988, 0.018,
             f"NWM Retrospective v3.0 (1979–2023), Bulletin 17C LP3  ·  "
             f"{len(hot):,} reaches  ·  NAD83 / UTM 16N",
             fontsize=4.9, color=MUTED, ha="right", va="bottom")
    # Say why the big rivers break up. Coverage follows the bridge inventory, so
    # the Ohio — the largest flow in the state — is fitted only where a monitored
    # bridge crosses it. Without this line a reader reads those gaps as low flow,
    # which is the opposite of the truth.
    fig.text(0.988, 0.007,
             "Coverage is bridge-driven: gaps on the Ohio and other large rivers "
             "are uncomputed, not low flow.",
             fontsize=4.9, color=MUTED, ha="right", va="bottom")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=args.dpi, facecolor=OUTSIDE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=OUTSIDE)
    print(f"Wrote {png} ({png.stat().st_size/1e6:.1f} MB) and {out.with_suffix('.pdf')}")

    if not args.no_s3:
        s3 = boto3.client("s3")
        for p, ct in ((png, "image/png"), (out.with_suffix(".pdf"), "application/pdf")):
            key = f"{PREFIX}figures/{p.name}"
            s3.upload_file(str(p), BUCKET, key, ExtraArgs={"ContentType": ct})
            print(f"  s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
