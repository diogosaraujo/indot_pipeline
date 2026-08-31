"""Map primitives for the alert digest — no geopandas, no shapely.

Lives in monitor_common because it is baked into the Lambda image; episode/
imports the same functions so the operational digest and the analysis figures
cannot drift apart in colour, scale, or symbology.

Geometry arrives pre-flattened (county rings from precompute/p07, flowlines
from episode/e02) as plain lon/lat tables, so nothing here needs a geometry
library. Distances use an equirectangular aspect correction rather than a
projection, which is accurate enough at one-state extent.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ── palette (dataviz reference instance, slots 1-3 + chrome) ─────────────────
C_CONF, C_OPEN, C_PRECIP, C_UNKNOWN = "#2a78d6", "#eb6834", "#1baf7a", "#7d7a72"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HAIRLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TIER_C = {10: "#f0ad4e", 50: "#d9534f", 100: "#7b1fa2"}

IN_BBOX = dict(lat=(37.70, 41.85), lon=(-88.15, -84.70))
STATE_EXTENT = dict(lat=(37.72, 41.83), lon=(-88.12, -84.72))

# Colour by which product confirms the alert; SIZE by severity tier. One channel
# cannot carry both, and a class missing from a product's list is invisible
# there — so every consumer iterates this dict rather than its own tuple list.
CLASS_STYLE = {
    "flow_conf": (C_CONF, "o", "Flow — A&A corroborates"),
    "flow_open": (C_OPEN, "o", "Flow — open-loop only"),
    "precip": (C_PRECIP, "^", "Precipitation"),
    "unknown": (C_UNKNOWN, "o", "Flow — corroboration unassessed"),
}
SEV_SIZE = {10: 26, 50: 58, 100: 108}

# Precipitation ramp from scripts/visualize_lanesville_event.py ("nws_precip"):
# transparent at zero, then blues -> greens -> yellow -> orange -> red -> purple.
PRECIP_COLORS = [
    (1.0, 1.0, 1.0, 0.0),
    "#b3d9ff", "#6ab4ff", "#1f78b4",
    "#33a02c", "#b2df8a",
    "#ffff33", "#ff7f00",
    "#e31a1c", "#fb9a99",
    "#6a0dad",
]
PRECIP_ALPHA = 0.70
RIVER_CMAP = "plasma_r"      # yellow (low) -> dark purple (high): floods go dark
RIVER_QUIET = "0.72"

PANEL_FIG = (24.0, 9.6)
PANEL_W, PANEL_X0, PANEL_GAP = 0.293, 0.028, 0.014

# ── ARI (average recurrence interval) ────────────────────────────────────────
# Return period is not a continuous quantity to the reader — it is a ladder of
# named tiers, and the three that gate alerts are 10 / 50 / 100. So bin it, and
# put TIER_C's amber / red / purple exactly at those bin edges: a cell that
# turns amber is a cell that just reached the tier a scour-critical bridge
# fires at. A continuous ramp would look prettier and say less.
ARI_BOUNDS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
ARI_COLORS = [
    "#dbeaf7", "#9ecae1", "#4a97d2",     # < 10 yr — ordinary rain, cool
    "#f0ad4e", "#e8893a",                # 10, 25  — TIER 10 amber
    "#d9534f", "#a32c28",                # 50, 100 — TIER 50 red
    "#7b1fa2", "#4f0d6b",                # 200, 500 — TIER 100 purple and beyond
]
ARI_OVER = "#24052f"      # >= 1000 yr
ARI_UNDER = "none"        # < 1 yr — leave the basemap showing through
ARI_ALPHA = 0.78


def panel_rects(y: float = 0.125, h: float = 0.715) -> list[list[float]]:
    return [[PANEL_X0 + i * (PANEL_W + PANEL_GAP), y, PANEL_W, h] for i in range(3)]


# Two panels side by side, for the daily summary's "field | its ARI" pages.
PAIR_FIG = (17.0, 9.6)
PAIR_W, PAIR_X0, PAIR_GAP = 0.443, 0.036, 0.032


def pair_rects(y: float = 0.135, h: float = 0.700) -> list[list[float]]:
    return [[PAIR_X0 + i * (PAIR_W + PAIR_GAP), y, PAIR_W, h] for i in range(2)]


def pair_legend_rects(y: float = 0.070, h: float = 0.017):
    left = [PAIR_X0 + 0.045, y, PAIR_W - 0.09, h]
    right = [PAIR_X0 + PAIR_W + PAIR_GAP + 0.045, y, PAIR_W - 0.09, h]
    return left, right


def ari_cmap_norm():
    """Discrete colormap + BoundaryNorm over ARI_BOUNDS (years).

    Alpha is baked into the colours rather than passed to pcolormesh as a
    scalar. A scalar `alpha=` overwrites the alpha channel of EVERY colour the
    colormap produces — including the transparent under/bad colours — which
    turns "no data" and "below 1 year" into opaque near-black and paints a dry
    day as if it were a catastrophic one. Callers must mask sub-1-year and NaN
    cells instead of relying on under/bad to disappear.
    """
    import matplotlib.colors as mcolors
    cols = [mcolors.to_rgba(c, ARI_ALPHA) for c in ARI_COLORS]
    cm = mcolors.ListedColormap(cols)
    cm.set_over(mcolors.to_rgba(ARI_OVER, ARI_ALPHA))
    cm.set_under((0, 0, 0, 0))
    cm.set_bad((0, 0, 0, 0))                # NaN = outside the Atlas-14 domain
    return cm, mcolors.BoundaryNorm(ARI_BOUNDS, cm.N)


def mask_below_ari(a):
    """Mask NaN and anything under the first ARI bin, so neither is painted."""
    return np.ma.masked_invalid(np.ma.masked_less(a, ARI_BOUNDS[0]))


def ari_legend(fig, rect, label: str = "average recurrence interval (years)") -> None:
    """Discrete ARI swatches. Drawn by hand rather than as a colorbar so the
    tier edges carry readable labels (10 / 50 / 100) instead of tick soup."""
    ax = fig.add_axes(rect)
    n = len(ARI_COLORS) + 1
    for i, c in enumerate([*ARI_COLORS, ARI_OVER]):
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (i / n, 0), 1 / n, 1, facecolor=c, edgecolor="white", linewidth=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([(i + 1) / n for i in range(len(ARI_BOUNDS) - 1)])
    ax.set_xticklabels([str(b) for b in ARI_BOUNDS[1:]], fontsize=7.5, color=INK2)
    ax.tick_params(length=2, pad=1.5, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, fontsize=8, color=INK2, loc="left", pad=3)


def panel_legend_rects(y: float = 0.062, h: float = 0.016):
    cb = [PANEL_X0 + 0.035, y, PANEL_W - 0.07, h]
    ramp = [PANEL_X0 + (PANEL_W + PANEL_GAP) + 0.055, y,
            (PANEL_W * 2 + PANEL_GAP) - 0.11, h]
    return cb, ramp


def precip_cmap():
    import matplotlib.colors as mcolors
    cm = mcolors.LinearSegmentedColormap.from_list("nws_precip", PRECIP_COLORS, N=256)
    cm.set_under("none")
    return cm


def sev_sizes(sev, scale: float = 1.0) -> np.ndarray:
    return np.array([SEV_SIZE.get(int(s), 40) * scale for s in np.asarray(sev)])


def draw_counties(ax, counties, lw=0.5, fc="#f4f3ef", overlay=False,
                  color="#6f6c65") -> None:
    """overlay=True strokes boundaries ABOVE the data — a filled raster at any
    useful alpha buries a basemap drawn beneath it."""
    if counties is None or getattr(counties, "empty", True):
        return
    for _, ring in counties.groupby("part_id"):
        x, y = ring["lon"].to_numpy(), ring["lat"].to_numpy()
        if overlay:
            ax.plot(x, y, color=color, linewidth=lw, zorder=6, rasterized=True,
                    solid_joinstyle="round", solid_capstyle="round")
        else:
            ax.fill(x, y, facecolor=fc, edgecolor=HAIRLINE, linewidth=lw,
                    zorder=1, rasterized=True)


def draw_flowlines(ax, flow, values: pd.Series | None = None, vmax: float = 1.5,
                   lw_base: float = 0.45, cmap: str = RIVER_CMAP,
                   lat=None, lon=None, norm=None, width_vmax: float | None = None,
                   lw_min: float = 0.0) -> None:
    """River network coloured by a per-COMID value (flow ÷ reach 100-yr Q).

    Colour AND width both track the value, so the encoding survives greyscale
    and small panels. Reaches with no value stay thin and grey — the quiet
    rivers are the context that makes the loud ones mean something.

    Pass `norm` (e.g. the BoundaryNorm from ari_cmap_norm) to map values through
    a scale instead of the default v/vmax fraction — the ARI panels need the
    same discrete tier bins the raster uses. `width_vmax` then sets the width
    ramp separately, since ARI years and a 0-1 fraction are not the same scale.

    `lw_min` floors the width of any VALUED reach. NWM reaches run 1-5 km, so at
    one-state extent a single reach over its 100-yr flow is a four-pixel dash
    lost in the grey network — the one thing the reader most needs to find.
    Widespread events read fine without it (many contiguous reaches light up at
    once); an isolated one does not.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if flow is None or getattr(flow, "empty", True):
        return
    f = flow
    if lat is not None and lon is not None:
        f = f[(f["lat"] >= lat[0] - .05) & (f["lat"] <= lat[1] + .05)
              & (f["lon"] >= lon[0] - .05) & (f["lon"] <= lon[1] + .05)]
        if f.empty:
            return
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    wmax = width_vmax if width_vmax is not None else vmax
    quiet, hot, hot_c, hot_w = [], [], [], []
    for (comid, _pid), g in f.groupby(["comid", "part_id"], sort=False):
        seg = g[["lon", "lat"]].to_numpy()
        if len(seg) < 2:
            continue
        v = None if values is None else values.get(comid)
        if v is None or not np.isfinite(v):
            quiet.append(seg)
        else:
            col = cm(norm(v)) if norm is not None else cm(float(np.clip(v / vmax, 0, 1)))
            frac = float(np.clip(v / wmax, 0, 1))
            hot.append(seg); hot_c.append(col)
            hot_w.append(max(lw_base * (1.0 + 3.5 * frac), lw_min))
    # rasterized: the network is ~300k vertices, and three vector copies of it
    # pushed the digest PDF past the 10 MB SES limit. Text and markers stay
    # vector because they are drawn above these.
    if quiet:
        ax.add_collection(LineCollection(quiet, colors=RIVER_QUIET,
                                         linewidths=lw_base, zorder=2,
                                         rasterized=True))
    if hot:
        ax.add_collection(LineCollection(hot, colors=hot_c, linewidths=hot_w,
                                         zorder=3, rasterized=True))


def draw_bridges(ax, d: pd.DataFrame, mscale: float = 1.9, lw: float = 0.9,
                 zorder: int = 8) -> None:
    for cls, (col, mark, _lbl) in CLASS_STYLE.items():
        s = d[d["map_class"] == cls] if "map_class" in d.columns else d.iloc[0:0]
        if not len(s):
            continue
        ax.scatter(s["lon"], s["lat"],
                   s=sev_sizes(s["severity_rp"], mscale * (1.25 if mark == "^" else 1.0)),
                   c=col, marker=mark, edgecolors="white", linewidths=lw, zorder=zorder)


def river_ramp_legend(fig, rect, vmax: float = 1.5, cmap: str = RIVER_CMAP,
                      label: str = "flow ÷ reach 100-yr Q") -> None:
    import matplotlib.pyplot as plt
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto",
              cmap=plt.get_cmap(cmap))
    ax.set_yticks([])
    ax.set_xticks([0, 127, 255])
    ax.set_xticklabels(["0", f"{vmax/2:.2g}×", f"≥{vmax:.2g}×"], fontsize=7.5, color=INK2)
    ax.tick_params(length=2, pad=1.5, colors=INK2)
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.6)
    ax.set_title(label, fontsize=8, color=INK2, loc="left", pad=3)


def set_geo(ax, lat, lon) -> None:
    ax.set_xlim(lon[0], lon[1]); ax.set_ylim(lat[0], lat[1])
    ax.set_aspect(1.0 / math.cos(math.radians((lat[0] + lat[1]) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.8)
