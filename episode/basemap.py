"""Raster navigation basemap for the episode maps, in plain lon/lat axes.

scripts/map_comid_13437963.py already stitches Esri tiles, but it draws its whole
figure in Web Mercator so the overlay registers against the tiles. The episode
products cannot do that: every layer they draw — MRMS mesh, NHDPlus flowlines,
county rings, bridge markers — is lon/lat, and set_geo fakes the projection with
an aspect ratio. So this module goes the other way and RESAMPLES the stitched
Mercator image onto a regular lon/lat grid, which then drops straight into
ax.imshow(extent=[lon0, lon1, lat0, lat1]) with no reprojection anywhere else.

The resampling matters less than it sounds at this scale — over a 0.26 deg tall
frame, dy/dlat changes by 0.4%, about one pixel — but it is one line of numpy and
it means the basemap cannot creep away from the data on a taller frame.

Tiles are cached on disk between runs (a 120-frame animation re-uses ONE fetch,
and re-running the script re-uses the cache), so a rerun costs no requests.

Usage:
    img, extent, attrib = basemap.fetch((39.81, 40.07), (-86.36, -85.79))
    ax.imshow(img, extent=extent, zorder=0, aspect="auto", interpolation="bilinear")
"""
from __future__ import annotations

import logging
import math
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

log = logging.getLogger("episode.basemap")

R_M = 6378137.0
UA = {"User-Agent": "indot-bridge-pipeline/1.0"}

# Raster services that need no key. "street" is the navigation-style map: named
# roads, route shields, town labels — the layer a reader uses to work out WHERE
# something is, which a county outline cannot do.
PROVIDERS = {
    "street": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
               "Basemap: Esri, HERE, Garmin, USGS, NGA"),
    "topo": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
             "Basemap: Esri, HERE, Garmin, USGS, NGA"),
    "gray": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
             "Basemap: Esri, HERE, Garmin, NGA, USGS"),
    "imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Imagery: Esri, Maxar, Earthstar Geographics"),
}

CACHE = pathlib.Path(tempfile.gettempdir()) / "indot_tilecache"


def _merc_y(lat):
    return R_M * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2))


def _deg2tile(lon, lat, z):
    n = 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def _tile_bounds(x, y, z):
    """(west_m, south_m, east_m, north_m) of one tile, in Web Mercator metres."""
    world = 2 * math.pi * R_M
    s = world / (2 ** z)
    return (-world / 2 + x * s, world / 2 - (y + 1) * s,
            -world / 2 + (x + 1) * s, world / 2 - y * s)


def zoom_for(lon, px_wide: int, lo: int = 9, hi: int = 15) -> int:
    """Smallest zoom whose tiles hold at least `px_wide` pixels across the frame.

    Picking this from the rendered panel width rather than hard-coding it keeps a
    200-dpi plate sharp without making a 90-dpi animation fetch four times the
    tiles it can show.
    """
    span = abs(lon[1] - lon[0])
    for z in range(lo, hi + 1):
        if span / 360.0 * 256 * 2 ** z >= px_wide:
            return z
    return hi


def fetch(lat, lon, zoom: int = 12, provider: str = "street", fade: float = 0.0,
          workers: int = 12, cache: pathlib.Path = CACHE):
    """(RGB array on a lon/lat grid, [lon0, lon1, lat0, lat1], attribution).

    `fade` blends the basemap toward white. Some fade is usually right: the map is
    the reference layer, and at full saturation its own colours compete with the
    rainfall field and the flow ramp that are the subject.
    """
    url, attrib = PROVIDERS[provider]
    cache.mkdir(parents=True, exist_ok=True)

    x0f, y1f = _deg2tile(lon[0], lat[0], zoom)
    x1f, y0f = _deg2tile(lon[1], lat[1], zoom)
    xs = list(range(int(math.floor(x0f)), int(math.floor(x1f)) + 1))
    ys = list(range(int(math.floor(y0f)), int(math.floor(y1f)) + 1))

    def one(k):
        x, y = k
        f = cache / f"{provider}_{zoom}_{y}_{x}.jpg"
        if not f.exists():
            r = requests.get(url.format(z=zoom, y=y, x=x), timeout=40, headers=UA)
            r.raise_for_status()
            f.write_bytes(r.content)
        return k, Image.open(f).convert("RGB")

    keys = [(x, y) for y in ys for x in xs]
    n_cached = sum((cache / f"{provider}_{zoom}_{k[1]}_{k[0]}.jpg").exists() for k in keys)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        tiles = dict(ex.map(one, keys))
    log.info("basemap %s z%d: %dx%d tiles (%d already cached)",
             provider, zoom, len(xs), len(ys), n_cached)

    canvas = Image.new("RGB", (256 * len(xs), 256 * len(ys)))
    for (x, y), im in tiles.items():
        canvas.paste(im, ((x - xs[0]) * 256, (y - ys[0]) * 256))
    src = np.asarray(canvas)

    # Stitched image bounds, in Mercator metres.
    west = _tile_bounds(xs[0], ys[0], zoom)[0]
    east = _tile_bounds(xs[-1], ys[-1], zoom)[2]
    south = _tile_bounds(xs[0], ys[-1], zoom)[1]
    north = _tile_bounds(xs[0], ys[0], zoom)[3]
    h, w = src.shape[:2]

    # Columns are linear in lon either way; rows have to come off the Mercator
    # scale, which is what keeps the tiles registered to a lat/lat axis.
    out_h, out_w = h, w
    lons = np.linspace(lon[0], lon[1], out_w)
    lats = np.linspace(lat[1], lat[0], out_h)          # top row first, as imshow wants
    cols = ((R_M * np.radians(lons) - west) / (east - west) * w).astype(int)
    rows = ((north - _merc_y(lats)) / (north - south) * h).astype(int)
    img = src[np.clip(rows, 0, h - 1)][:, np.clip(cols, 0, w - 1)]

    if fade:
        img = (img.astype(np.float32) * (1 - fade) + 255.0 * fade).astype(np.uint8)
    return img, [lon[0], lon[1], lat[0], lat[1]], attrib


def draw(ax, bm, zorder: int = 0) -> None:
    """Put a fetched basemap under everything on one axes.

    aspect="auto" is not optional: imshow otherwise forces an equal aspect and
    silently overrides the 1/cos(lat) correction set_geo applies, which squashes
    every other layer on the panel.
    """
    if bm is None:
        return
    img, extent, _ = bm
    ax.imshow(img, extent=extent, origin="upper", zorder=zorder, aspect="auto",
              interpolation="bilinear")
