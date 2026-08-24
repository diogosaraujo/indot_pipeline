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


def panel_rects(y: float = 0.125, h: float = 0.715) -> list[list[float]]:
    return [[PANEL_X0 + i * (PANEL_W + PANEL_GAP), y, PANEL_W, h] for i in range(3)]


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
                   lat=None, lon=None) -> None:
    """River network coloured by a per-COMID value (flow ÷ reach 100-yr Q).

    Colour AND width both track the value, so the encoding survives greyscale
    and small panels. Reaches with no value stay thin and grey — the quiet
    rivers are the context that makes the loud ones mean something.
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
    cm = plt.get_cmap(cmap)
    quiet, hot, hot_c, hot_w = [], [], [], []
    for (comid, _pid), g in f.groupby(["comid", "part_id"], sort=False):
        seg = g[["lon", "lat"]].to_numpy()
        if len(seg) < 2:
            continue
        v = None if values is None else values.get(comid)
        if v is None or not np.isfinite(v):
            quiet.append(seg)
        else:
            frac = float(np.clip(v / vmax, 0, 1))
            hot.append(seg); hot_c.append(cm(frac))
            hot_w.append(lw_base * (1.0 + 3.5 * frac))
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
