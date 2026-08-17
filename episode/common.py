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
MAX_LABELS = 40                   # bridges per label page before tiling

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


def tile_region(region: dict, pts: pd.DataFrame, max_labels: int = MAX_LABELS) -> list[dict]:
    """Split one region into label pages of <= max_labels bridges.

    Halves along the longer axis (in miles) until every tile is under the cap,
    then shrink-wraps each tile to its own points so a sparse tile is not mostly
    empty. Tiles inherit nothing from each other, so labels never collide across
    a page boundary.
    """
    n = len(pts)
    if n <= max_labels:
        lat, lon = _padded_extent(pts["lat"], pts["lon"])
        return [dict(lat=lat, lon=lon, n=n, part=1, parts=1)]

    parts = int(math.ceil(n / max_labels))
    ncol = int(math.ceil(math.sqrt(parts * (region["span_mi"][1] / max(region["span_mi"][0], 1)))))
    ncol = max(1, ncol)
    nrow = int(math.ceil(parts / ncol))

    lo0, lo1 = region["lon"]; la0, la1 = region["lat"]
    tiles = []
    for r in range(nrow):
        for c in range(ncol):
            x0 = lo0 + (lo1 - lo0) * c / ncol
            x1 = lo0 + (lo1 - lo0) * (c + 1) / ncol
            y1 = la1 - (la1 - la0) * r / nrow
            y0 = la1 - (la1 - la0) * (r + 1) / nrow
            sel = pts[(pts["lon"] >= x0) & (pts["lon"] < x1)
                      & (pts["lat"] >= y0) & (pts["lat"] < y1)]
            if not len(sel):
                continue
            lat, lon = _padded_extent(sel["lat"], sel["lon"])
            tiles.append(dict(lat=lat, lon=lon, n=len(sel)))
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


# Shared 3-panel geometry (MRMS | NWM open-loop | NWM A&A). e04 and e05 both
# use it so an animation frame and the static map for the same day are the same
# size and land on the same ground — you can flip between them without the eye
# having to re-register the map.
PANEL_FIG = (24.0, 9.6)
PANEL_W, PANEL_X0, PANEL_GAP = 0.293, 0.028, 0.014


def panel_rects(y: float = 0.125, h: float = 0.715) -> list[list[float]]:
    return [[PANEL_X0 + i * (PANEL_W + PANEL_GAP), y, PANEL_W, h] for i in range(3)]


def panel_legend_rects(y: float = 0.062, h: float = 0.016):
    """(rainfall colourbar rect, shared streamflow ramp rect)."""
    cb = [PANEL_X0 + 0.035, y, PANEL_W - 0.07, h]
    ramp = [PANEL_X0 + (PANEL_W + PANEL_GAP) + 0.055, y,
            (PANEL_W * 2 + PANEL_GAP) - 0.11, h]
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
    ax.set_xticklabels(["0", f"{vmax/2:.2g}×", f"≥{vmax:.2g}×"], fontsize=7.5, color=INK2)
    ax.tick_params(length=2, pad=1.5, colors=INK2)
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.6)
    ax.set_title(label, fontsize=8, color=INK2, loc="left", pad=3)


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
    span = y1 - y0
    gap = span * min_gap_frac
    d = pts.sort_values("lat", ascending=False).copy()
    left = d["lon"] > (x0 + x1) / 2          # flip side near the right edge
    d["ty"] = d["lat"]
    ys = d["ty"].to_numpy()
    for i in range(1, len(ys)):              # push down to clear the one above
        if ys[i - 1] - ys[i] < gap:
            ys[i] = ys[i - 1] - gap
    d["ty"] = ys
    dx = (x1 - x0) * 0.012
    for (_, r), lft in zip(d.iterrows(), left):
        sx = r["lon"] - dx if lft else r["lon"] + dx
        ha = "right" if lft else "left"
        if abs(r["ty"] - r["lat"]) > gap * 0.55:
            ax.plot([r["lon"], sx], [r["lat"], r["ty"]], lw=0.4, color=MUTED,
                    zorder=5, solid_capstyle="round")
        ax.text(sx, r["ty"], str(r[text_col]), fontsize=fontsize, ha=ha,
                va="center", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.72))


def draw_counties(ax, counties, lw=0.5, fc="#f4f3ef") -> None:
    if counties is None or counties.empty:
        return
    for _, ring in counties.groupby("part_id"):
        ax.fill(ring["lon"].to_numpy(), ring["lat"].to_numpy(),
                facecolor=fc, edgecolor=HAIRLINE, linewidth=lw, zorder=1)


def set_geo(ax, lat, lon) -> None:
    ax.set_xlim(lon[0], lon[1]); ax.set_ylim(lat[0], lat[1])
    ax.set_aspect(1.0 / math.cos(math.radians((lat[0] + lat[1]) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.8)
