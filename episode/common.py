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
