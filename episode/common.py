"""Shared pieces for the episode report (2026-08-12 .. 08-16 flood sequence).

This is an ANALYSIS product, not part of the operational monitor. It re-fetches
MRMS and NWM from the NOAA public buckets because the monitor prunes its own
state at STATE_HOURS (48 h), so anything older than two days is already gone.

Region model
------------
Zoom regions are a FIXED catalog derived once from the whole episode's alert
geometry (single-linkage at 6 mi, small clusters merged into a nearby larger
one). Fixed means a region keeps the same extent every day it appears, so two
days can be laid side by side over identical ground.

Label pages additionally TILE a region when a single day puts more than
MAX_LABELS bridges in it — Aug 12 drops 140 bridges into the Whitewater frame,
which no amount of font tuning makes legible. Animations do not tile: they show
fields rather than labels, so they use the whole region extent.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO / "monitor"), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from monitor_common import config  # noqa: E402
from monitor_common.s3io import read_parquet, write_parquet  # noqa: E402

log = logging.getLogger("episode")

# ── Episode definition ───────────────────────────────────────────────────────
DAYS = ["2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16"]
TZ = "US/Eastern"                 # days are local days; data is fetched in UTC
IN_BBOX = dict(lat=(37.70, 41.85), lon=(-88.15, -84.70))    # Indiana + margin
MAX_LABELS = 22                   # bridges per label page before tiling

# ── Palette (dataviz reference instance, slots 1-3 + chrome) ─────────────────
C_CONF, C_OPEN, C_PRECIP = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HAIRLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TIER_C = {10: "#f0ad4e", 50: "#d9534f", 100: "#7b1fa2"}

# ── S3 layout for this product ───────────────────────────────────────────────
def ep_key(name: str) -> str:
    _, prefix = config.bucket_prefix()
    return f"{prefix}episode/{name}"


def bucket() -> str:
    return config.bucket_prefix()[0]


REGIONS_JSON = REPO / "episode" / "regions.json"


# ── Region catalog ───────────────────────────────────────────────────────────

def derive_regions(ev: pd.DataFrame, link_mi: float = 6.0,
                   min_bridges: int = 5, merge_within_mi: float = 45.0) -> dict:
    """Cluster the episode's alerting bridges into a fixed zoom catalog."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    b = ev.groupby("bridge_id").agg(lat=("lat", "first"), lon=("lon", "first")).dropna()
    xy = np.c_[b["lat"] * 69.0, b["lon"] * 53.0]
    cl = pd.Series(fcluster(linkage(pdist(xy), "single"), link_mi, "distance"), index=b.index)

    # fold small clusters into the nearest big one so we don't emit 6-bridge pages
    sizes = cl.value_counts()
    big = list(sizes[sizes >= 12].index)
    cent = {c: (b.loc[cl == c, "lat"].mean(), b.loc[cl == c, "lon"].mean()) for c in sizes.index}
    for c in sizes.index:
        if c in big:
            continue
        best, bestd = None, np.inf
        for t in big:
            d = math.hypot((cent[c][0] - cent[t][0]) * 69, (cent[c][1] - cent[t][1]) * 53)
            if d < bestd:
                best, bestd = t, d
        if best is not None and bestd <= merge_within_mi:
            cl[cl == c] = best
    sizes = cl.value_counts()
    keep = list(sizes[sizes >= min_bridges].index)

    ev = ev.copy()
    ev["cl"] = ev["bridge_id"].map(cl)
    out = {}
    for rank, c in enumerate(sorted(keep, key=lambda k: -sizes[k]), 1):
        d = b[cl == c]
        lat, lon = _padded_extent(d["lat"], d["lon"])
        per_day = (ev[ev["cl"] == c].groupby("day")["bridge_id"].nunique()
                   .astype(int).to_dict())
        out[f"R{rank}"] = dict(
            bridges=int(sizes[c]), lat=lat, lon=lon,
            centroid=[round(d["lat"].mean(), 3), round(d["lon"].mean(), 3)],
            span_mi=[round((lat[1] - lat[0]) * 69), round((lon[1] - lon[0]) * 53)],
            per_day={str(k): int(v) for k, v in per_day.items()},
            name=f"R{rank}")
    return out


def _padded_extent(lats, lons, pad=0.07, min_lat_span=0.26, min_lon_span=0.34):
    la0, la1 = float(lats.min()) - pad, float(lats.max()) + pad
    lo0, lo1 = float(lons.min()) - pad, float(lons.max()) + pad
    if la1 - la0 < min_lat_span:
        m = (la0 + la1) / 2; la0, la1 = m - min_lat_span / 2, m + min_lat_span / 2
    if lo1 - lo0 < min_lon_span:
        m = (lo0 + lo1) / 2; lo0, lo1 = m - min_lon_span / 2, m + min_lon_span / 2
    return [round(la0, 3), round(la1, 3)], [round(lo0, 3), round(lo1, 3)]


def load_regions() -> dict:
    """Repo copy first (reviewable, version-controlled), else the S3 mirror.

    e00 writes both: the repo file is the artifact you read in a diff, the S3
    copy is what a machine that didn't generate it can still pick up.
    """
    if REGIONS_JSON.exists():
        return json.loads(REGIONS_JSON.read_text())
    from monitor_common.s3io import read_bytes
    try:
        return json.loads(read_bytes(bucket(), ep_key("regions.json")).decode())
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"No region catalog at {REGIONS_JSON} or s3 episode/regions.json — "
            "run e00_regions.py first") from e


def active_regions(regions: dict, day: str) -> list[str]:
    """Region ids that had at least one alerting bridge on `day`."""
    return [k for k, r in regions.items() if r["per_day"].get(day, 0) > 0]


def _split_balanced(pts: pd.DataFrame, cap: int) -> list[pd.DataFrame]:
    """Recursively halve a point set at the MEDIAN of its longer axis.

    A fixed grid splits a region into equal areas, but bridges cluster along
    rivers, so one cell inherits nearly all of them and stays unreadable while
    its neighbours sit empty. Splitting at the median balances COUNT instead,
    which is what determines whether labels fit.
    """
    if len(pts) <= cap:
        return [pts]
    dlat = (pts["lat"].max() - pts["lat"].min()) * 69.0
    dlon = (pts["lon"].max() - pts["lon"].min()) * 53.0
    col = "lat" if dlat >= dlon else "lon"
    med = pts[col].median()
    a, b = pts[pts[col] <= med], pts[pts[col] > med]
    if not len(a) or not len(b):          # ties collapse the median — split by rank
        s = pts.sort_values(col)
        h = len(s) // 2
        a, b = s.iloc[:h], s.iloc[h:]
    return _split_balanced(a, cap) + _split_balanced(b, cap)


def tile_region(region: dict, pts: pd.DataFrame, max_labels: int = MAX_LABELS) -> list[dict]:
    """Split one region into label pages of <= max_labels bridges.

    Each tile is shrink-wrapped to its own points, so a sparse tile is not
    mostly empty and a dense one gets the magnification it needs. Tiles share
    no bridges, so labels never collide across a page boundary.
    """
    groups = _split_balanced(pts, max_labels)
    groups.sort(key=lambda g: (-g["lat"].mean(), g["lon"].mean()))   # N->S, W->E
    tiles = []
    for g in groups:
        lat, lon = _padded_extent(g["lat"], g["lon"])
        tiles.append(dict(lat=lat, lon=lon, n=len(g)))
    for i, t in enumerate(tiles, 1):
        t["part"], t["parts"] = i, len(tiles)
    return tiles


# ── Data access ──────────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    """All alert events for the episode, with bridge metadata attached."""
    return read_parquet(bucket(), ep_key("episode_events.parquet"))


def load_config() -> pd.DataFrame:
    k = config.keys()
    return read_parquet(k["bucket"], k["config"])


def load_counties() -> pd.DataFrame | None:
    try:
        return read_parquet(bucket(), config.keys()["counties"])
    except Exception as e:  # noqa: BLE001
        log.warning("county outlines unavailable: %s", e)
        return None


def load_flowlines() -> pd.DataFrame | None:
    """Flattened NHDPlus flowlines: comid, part_id, lon, lat (from e02)."""
    try:
        return read_parquet(bucket(), ep_key("flowlines.parquet"))
    except Exception as e:  # noqa: BLE001
        log.warning("flowlines unavailable (%s) — run e02_flowlines.py", e)
        return None


def load_counties_named() -> pd.DataFrame | None:
    """County rings WITH names: part_id, lon, lat, name, geoid.

    The plain in_counties.parquet has no names, so it cannot answer "stroke
    Wayne County". Built by scripts/build_in_counties_named.py.
    """
    _, prefix = config.bucket_prefix()
    try:
        return read_parquet(bucket(),
                            f"{prefix}monitor/assets/in_counties_named.parquet")
    except Exception as e:  # noqa: BLE001
        log.warning("named counties unavailable (%s)", e)
        return None


def draw_county_outline(ax, cnamed, name: str, color: str = "#1c1a17",
                        lw: float = 2.0, zorder: int = 7) -> None:
    """Stroke ONE county's boundary above the data — the frame's subject.

    Carries a white casing so the line stays readable where it crosses the
    rainfall raster, which is the same trick the city labels use.
    """
    import matplotlib.patheffects as pe

    if cnamed is None or getattr(cnamed, "empty", True) or not name:
        return
    d = cnamed[cnamed["name"].astype(str).str.upper() == str(name).upper()]
    if d.empty:
        log.warning("county %r not found in the named-counties asset", name)
        return
    for _pid, g in d.groupby("part_id", sort=False):
        ax.plot(g["lon"].to_numpy(), g["lat"].to_numpy(), color=color,
                linewidth=lw, zorder=zorder, solid_joinstyle="round",
                solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=lw + 1.8, foreground="white")])


def load_roads() -> pd.DataFrame | None:
    """Flattened TIGER primary+secondary roads: part_id, lon, lat, cls, name.

    Same shape as flowlines — plain lon/lat rows, no geometry library — so it
    draws with the rest of the basemap. Built by scripts/build_in_roads.py.
    """
    _, prefix = config.bucket_prefix()
    try:
        return read_parquet(bucket(), f"{prefix}monitor/assets/in_roads.parquet")
    except Exception as e:  # noqa: BLE001
        log.warning("roads unavailable (%s) — run scripts/build_in_roads.py", e)
        return None


def hour_range(day: str) -> list[pd.Timestamp]:
    """The 24 UTC hours whose LOCAL timestamp falls on `day`."""
    start = pd.Timestamp(f"{day} 00:00", tz=TZ).tz_convert("UTC")
    return [start + pd.Timedelta(hours=h) for h in range(24)]


def mrms_key(ts: pd.Timestamp) -> str:
    return ep_key(f"mrms/{ts:%Y%m%d%H}.npz")


def nwm_key(ts: pd.Timestamp) -> str:
    return ep_key(f"nwm/{ts:%Y%m%d%H}.parquet")


def load_mrms_hour(ts: pd.Timestamp):
    """(arr, lats_desc, lons_asc) in inches for one hour, or None if absent."""
    import io as _io
    from monitor_common.s3io import read_bytes
    try:
        z = np.load(_io.BytesIO(read_bytes(bucket(), mrms_key(ts))))
    except Exception:  # noqa: BLE001
        return None
    return z["arr"], z["lats"], z["lons"]


def load_nwm_hour(ts: pd.Timestamp) -> pd.DataFrame | None:
    try:
        return read_parquet(bucket(), nwm_key(ts)).set_index("comid")
    except Exception:  # noqa: BLE001
        return None


def day_accum(day: str) -> tuple:
    """24-h MRMS accumulation (inches) for a local day, plus the hour count."""
    acc = lats = lons = None
    n = 0
    for ts in hour_range(day):
        got = load_mrms_hour(ts)
        if got is None:
            continue
        arr, lats, lons = got
        acc = arr.astype(np.float64) if acc is None else acc + arr
        n += 1
    return acc, lats, lons, n


def day_peak_flow(day: str) -> pd.DataFrame:
    """Per-COMID peak open-loop and A&A flow (cms) over a local day."""
    best = None
    for ts in hour_range(day):
        d = load_nwm_hour(ts)
        if d is None:
            continue
        cur = d[[c for c in ("q_ol_cms", "q_aa_cms") if c in d.columns]]
        best = cur if best is None else np.fmax(best, cur.reindex(best.index))
    return best if best is not None else pd.DataFrame()


# ── Plot helpers ─────────────────────────────────────────────────────────────

# Bridge marker styling, defined ONCE. Every product iterates this dict rather
# than its own tuple list: e04 previously listed only three classes, so Aug 12's
# events — all 'unknown' before the backfill — matched nothing and vanished from
# the animations entirely. A class missing from one product's list is invisible
# there while showing fine elsewhere, which is a hard bug to see.
C_UNKNOWN = "#7d7a72"
CLASS_STYLE = {
    "flow_conf": (C_CONF, "o", "Flow — A&A corroborates"),
    "flow_open": (C_OPEN, "o", "Flow — open-loop only"),
    "precip": (C_PRECIP, "^", "Precipitation"),
    "unknown": (C_UNKNOWN, "o", "Flow — corroboration unassessed"),
}


# Road weights. Interstates carry the widest, darkest stroke because on a
# county map they are the landmark a reader orients from; state routes stay
# hairline so they never compete with the river network, which is the subject.
ROAD_STYLE = {
    "interstate": ("#6b6760", 1.4),
    "us": ("#918d84", 0.8),
    "state": ("#b0aca2", 0.45),
}


def draw_roads(ax, roads, lat=None, lon=None, zorder: int = 4,
               highlight: str | None = None, highlight_color: str = "#33302a",
               scale: float = 1.0) -> None:
    """Roads above the county fill, below the bridge markers.

    `highlight` is a route name such as "I-70"; matching parts are drawn heavier
    and darker so the named road reads as the landmark. TIGER writes interstate
    names with a space after the dash ("I- 70"), so the match strips spaces from
    both sides rather than comparing literally — a plain == "I-70" finds nothing.
    """
    from matplotlib.collections import LineCollection

    if roads is None or getattr(roads, "empty", True):
        return
    r = roads
    if lat is not None and lon is not None:
        r = r[(r["lat"] >= lat[0] - .05) & (r["lat"] <= lat[1] + .05)
              & (r["lon"] >= lon[0] - .05) & (r["lon"] <= lon[1] + .05)]
        if r.empty:
            return
    key = highlight.replace(" ", "").upper() if highlight else None

    segs, cols, widths, hi = [], [], [], []
    for _pid, g in r.groupby("part_id", sort=False):
        xy = g[["lon", "lat"]].to_numpy()
        if len(xy) < 2:
            continue
        if key and key in str(g["name"].iloc[0]).replace(" ", "").upper():
            hi.append(xy)
            continue
        c, w = ROAD_STYLE.get(str(g["cls"].iloc[0]), ROAD_STYLE["state"])
        segs.append(xy); cols.append(c); widths.append(w * scale)

    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths,
                                         zorder=zorder, alpha=0.85,
                                         capstyle="round", joinstyle="round"))
    if hi:
        ax.add_collection(LineCollection(hi, colors=highlight_color,
                                         linewidths=2.4 * scale, zorder=zorder + 0.5,
                                         capstyle="round", joinstyle="round"))
        # Label the highlighted route on the map. Placed at the midpoint of its
        # LONGEST part, which keeps the tag on open road rather than on a stub
        # near the frame edge. Horizontal rather than rotated: set_geo applies an
        # aspect correction, so a rotation computed in degrees of lon/lat does
        # not match the drawn angle and the text ends up visibly off the road.
        longest = max(hi, key=len)
        mx, my = longest[len(longest) // 2]
        # Offset the tag OFF the carriageway, and make its plate opaque. Centred
        # on the line with a 0.92-alpha box, the 2.4 pt highlight showed through
        # and read as a strike-through. It sits just north of the road instead,
        # the way a route shield is placed on a printed map.
        if lat is not None:
            my = float(my) + (lat[1] - lat[0]) * 0.030
        # zorder 9 puts it above the bridge markers (8): a route label buried
        # under a cluster of alerts is a label that does not do its job.
        ax.text(mx, my, highlight, fontsize=12.5, fontweight="bold",
                color=highlight_color, ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=highlight_color, lw=0.9))


def draw_bridges(ax, d: pd.DataFrame, mscale: float = 1.9, lw: float = 0.9,
                 zorder: int = 8) -> None:
    """Every alert class, colour by confirmation and size by severity."""
    for cls, (col, mark, _lbl) in CLASS_STYLE.items():
        s = d[d["map_class"] == cls]
        if not len(s):
            continue
        ax.scatter(s["lon"], s["lat"],
                   s=sev_sizes(s["severity_rp"], mscale * (1.25 if mark == "^" else 1.0)),
                   c=col, marker=mark, edgecolors="white", linewidths=lw, zorder=zorder)


# Precipitation colour scale, lifted from scripts/visualize_lanesville_event.py
# (its "nws_precip" ramp): transparent at zero, then blues -> greens -> yellow
# -> orange -> red -> purple. set_under("none") keeps dry cells fully clear so
# the basemap shows through rather than being covered by a pale wash.
PRECIP_COLORS = [
    (1.0, 1.0, 1.0, 0.0),          # transparent for zero
    "#b3d9ff", "#6ab4ff", "#1f78b4",
    "#33a02c", "#b2df8a",
    "#ffff33", "#ff7f00",
    "#e31a1c", "#fb9a99",
    "#6a0dad",
]
PRECIP_ALPHA = 0.70                # let counties and rivers read through the field


def precip_cmap():
    import matplotlib.colors as mcolors
    cm = mcolors.LinearSegmentedColormap.from_list("nws_precip", PRECIP_COLORS, N=256)
    cm.set_under("none")
    return cm


# River colour scale, matching scripts/visualize_lanesville_event.py: plasma_r
# runs yellow (low) -> magenta -> dark purple (high), so the flooding reaches go
# dark and prominent on a light basemap instead of fading out the way the light
# end of a single-hue blue ramp does. Quiet reaches use Lanesville's 0.72 gray.
RIVER_CMAP = "plasma_r"
RIVER_QUIET = "0.72"


def draw_flowlines(ax, flow, values: pd.Series | None = None, vmax: float = 1.5,
                   lw_base: float = 0.45, cmap: str = RIVER_CMAP,
                   lat=None, lon=None) -> None:
    """River network, optionally colored by a per-COMID value (e.g. q/Q100).

    Reaches with no value are drawn thin and gray so the network still reads as
    a network — the quiet rivers are the context that makes the loud ones mean
    something. Colour AND width both track the value, so the encoding survives
    greyscale printing and small panel sizes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if flow is None or flow.empty:
        return
    f = flow
    if lat is not None and lon is not None:      # clip to the frame before building segments
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
    if quiet:
        ax.add_collection(LineCollection(quiet, colors=RIVER_QUIET,
                                         linewidths=lw_base, zorder=2))
    if hot:
        ax.add_collection(LineCollection(hot, colors=hot_c, linewidths=hot_w, zorder=3))


# Shared 2-panel geometry (MRMS | NWM). e04 and e05 both use it so an animation
# frame and the static map for the same day are the same size and land on the
# same ground — you can flip between them without the eye having to re-register
# the map.
#
# The canvas is the PowerPoint slide's TRUE width, and that is the whole reason
# the text reads. Font sizes are absolute points on the figure, so a 24-inch
# canvas dropped into a 13.33-inch slide is scaled to 56% and every label
# shrinks with it — 13 pt panel titles arrived at ~7 pt on screen. Raising the
# point sizes on an oversized canvas would not have fixed that. Sizing the
# figure to the slide makes points map 1:1, so 12 pt here is 12 pt in the deck.
# If you ever change this canvas, scale every font in e04/e05 by the same factor
# or the deck silently goes back to being unreadable.
#
# Height is capped at 6 in by request — shorter than a full 16:9 slide (7.5 in),
# leaving room on the slide under the figure. Because the fonts do NOT shrink
# with it, that lost 1.5 in comes straight out of the panels, which is why the
# header below packs two rows instead of four.
PANEL_FIG = (13.333, 6.0)
PANELS = 2
PANEL_W, PANEL_X0, PANEL_GAP = 0.445, 0.030, 0.020


# Two header rows, not four: with the explanatory clauses dropped, the title is
# short enough to share row 1 with the class key and the subtitle to share row 2
# with the severity key. On a 6 in canvas that recovered the panel height the
# shorter figure would otherwise have cost.
def panel_rects(y: float = 0.165, h: float = 0.625) -> list[list[float]]:
    return [[PANEL_X0 + i * (PANEL_W + PANEL_GAP), y, PANEL_W, h]
            for i in range(PANELS)]


def panel_legend_rects(y: float = 0.090, h: float = 0.026):
    """(rainfall colourbar rect, streamflow ramp rect) — one under each panel."""
    cb = [PANEL_X0 + 0.035, y, PANEL_W - 0.07, h]
    ramp = [PANEL_X0 + (PANEL_W + PANEL_GAP) + 0.035, y, PANEL_W - 0.07, h]
    return cb, ramp


# Severity is encoded as marker SIZE, not colour: colour already carries which
# product confirms the alert, and one channel cannot do two jobs. Sizes are in
# matplotlib points^2.
SEV_SIZE = {10: 26, 50: 58, 100: 108}


def sev_sizes(sev, scale: float = 1.0) -> np.ndarray:
    return np.array([SEV_SIZE.get(int(s), 40) * scale for s in np.asarray(sev)])


def river_ramp_legend(fig, rect, vmax: float = 1.5, cmap: str = RIVER_CMAP,
                      label: str = "peak flow ÷ reach 100-yr Q") -> None:
    """Horizontal strip decoding the river shading.

    Shows the full 0-1 range because draw_flowlines now samples the whole
    colormap; if that mapping is ever narrowed again, narrow this to match or
    the legend lies about the ends.
    """
    import matplotlib.pyplot as plt
    ax = fig.add_axes(rect)
    grad = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=plt.get_cmap(cmap))
    ax.set_yticks([])
    ax.set_xticks([0, 127, 255])
    # 12 pt is the floor for this deck: these ramp ticks are the smallest text
    # on the figure, so nothing else may go below them.
    ax.set_xticklabels(["0", f"{vmax/2:.2g}×", f"≥{vmax:.2g}×"], fontsize=12, color=INK2)
    ax.tick_params(length=3, pad=3, colors=INK2)
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.6)
    ax.set_title(label, fontsize=13, color=INK2, loc="left", pad=5)


def place_labels(ax, pts: pd.DataFrame, text_col: str, fontsize=7.0,
                 min_gap_frac=0.030) -> None:
    """Direct labels with greedy vertical de-confliction and leader lines.

    Labels start to the right of their marker, are pushed apart vertically until
    none overlap, and get a leader line once moved far enough that the pairing
    would otherwise be ambiguous.
    """
    if pts.empty:
        return
    y0, y1 = ax.get_ylim(); x0, x1 = ax.get_xlim()
    gap = (y1 - y0) * min_gap_frac
    dx = (x1 - x0) * 0.012
    mid = (x0 + x1) / 2

    # De-conflict each SIDE independently. Pushing left- and right-hand labels
    # down a single shared stack wastes half the column and drives the lower
    # ones off the frame, which is what made the dense pages unreadable.
    for side in ("left", "right"):
        d = pts[(pts["lon"] > mid) if side == "left" else (pts["lon"] <= mid)]
        if d.empty:
            continue
        d = d.sort_values("lat", ascending=False).copy()
        ys = d["lat"].to_numpy(float).copy()
        for i in range(1, len(ys)):          # push down to clear the one above
            if ys[i - 1] - ys[i] < gap:
                ys[i] = ys[i - 1] - gap
        overflow = ys[-1] - (y0 + gap * 0.5)  # re-centre if the stack ran off
        if overflow < 0:
            ys -= overflow
        for (_, r), ty in zip(d.iterrows(), ys):
            sx = r["lon"] - dx if side == "left" else r["lon"] + dx
            ha = "right" if side == "left" else "left"
            if abs(ty - r["lat"]) > gap * 0.55:
                ax.plot([r["lon"], sx], [r["lat"], ty], lw=0.4, color=MUTED,
                        zorder=5, solid_capstyle="round")
            ax.text(sx, ty, str(r[text_col]), fontsize=fontsize, ha=ha,
                    va="center", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.80))


def draw_counties(ax, counties, lw=0.5, fc="#f4f3ef", overlay=False,
                  color="#6f6c65") -> None:
    """County polygons.

    overlay=False fills them as the basemap, under everything. overlay=True
    strokes the boundaries ABOVE the data instead — a filled raster at any
    useful alpha buries a basemap drawn beneath it, so the geography has to be
    restated on top or the reader loses all sense of where they are.
    """
    if counties is None or counties.empty:
        return
    for _, ring in counties.groupby("part_id"):
        x, y = ring["lon"].to_numpy(), ring["lat"].to_numpy()
        if overlay:
            ax.plot(x, y, color=color, linewidth=lw, zorder=6,
                    solid_joinstyle="round", solid_capstyle="round")
        else:
            ax.fill(x, y, facecolor=fc, edgecolor=HAIRLINE, linewidth=lw, zorder=1)


def set_geo(ax, lat, lon) -> None:
    ax.set_xlim(lon[0], lon[1]); ax.set_ylim(lat[0], lat[1])
    ax.set_aspect(1.0 / math.cos(math.radians((lat[0] + lat[1]) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.8)
