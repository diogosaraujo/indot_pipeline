"""visualize_lanesville_mrms_panels.py

Poster figure for the Lanesville, IN flash flood, 9 June 2026: a 3 x 1 stack of
hourly MRMS QPE maps (MultiSensor_QPE_01H_Pass2), files 14, 15, 16 UTC (the
three wet frames of 13-17 UTC in visualize_lanesville_event.py's GIF).

  * MRMS 1-h QPE is labelled by the END of its hour, so the "14 UTC" file is the
    rain that fell 13:00-14:00 UTC; each panel heading states that interval.
  * One fixed colour scale shared by all panels: the GIF's colormap, but 0.01-4.0 in
    (the GIF's 2.0 in cap saturates; MRMS peaks at 3.9 in/h in this view).
  * All bridges are green dots (no scour-critical split); bridge 062-31-07183 is a
    red star (it is discussed in a separate figure). USGS 03294500 precip
    station is the yellow dot, as in the GIF.
  * Maps keep the true ground aspect (1 deg lon = cos(lat) deg lat) over the GIF's
    extent (-86.10..-85.72, 38.12..38.34). Their size is set by the 12.7 in height
    limit; the width is free, so the legend and colour bar sit to the right.
  * Text (Times New Roman): the date once, above the top map; each map's local
    (EDT) hour window in its lower-right corner (Kentucky, no Indiana bridges there). Coordinates are
    labelled only at the extent bounds (latitude rotated to save width).
  * Saved at 600 dpi with a transparent background: printed at 2x it is 300 dpi.

Runs locally (no cfgrib / ecCodes needed): MRMS is fetched over public HTTPS
and its GRIB2 PNG packing (data representation template 5.41) is decoded here.
Bridges come from the pipeline bucket (AWS credentials).

Writes:
    results/lanesville_20260609_mrms_3panel.png

Usage:
    python scripts/visualize_lanesville_mrms_panels.py
    python scripts/visualize_lanesville_mrms_panels.py --hours 13 14 15 16 17
"""
from __future__ import annotations

import argparse
import gzip
import io
import math
import os
import struct
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import requests
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from PIL import Image

import contextily as ctx

# ── Configuration ──────────────────────────────────────────────────────────────
EVENT_DATE = date(2026, 6, 9)
HOURS = [14, 15, 16]                         # MRMS file hours (UTC, end of the 1-h window)
UTC_OFFSET_HR, TZ_LABEL = -4, "EDT"          # Harrison Co., IN in June

# ── Layout (inches) ────────────────────────────────────────────────────────────
MAX_FIG_H = 12.7
TOP_PAD = 0.05
DATE_H = 0.62                                # date heading above the top map
PANEL_GAP = 0.10                             # between maps
XTICK_H = 0.42                               # longitude bound labels under the last map
LEFT_PAD = 0.50                              # rotated latitude bound labels
COL_GAP = 0.25
LEGEND_H = 2.40                              # legend block at the top of the right column
CBAR_W = 0.32
CBAR_LEN = 5.00                              # colour bar length, below the legend
RIGHT_W = 2.00                               # right column (legend / colour bar + labels)

LON_MIN, LON_MAX = -86.10, -85.72            # the GIF's view
LAT_MIN, LAT_MAX = 38.12, 38.34

COORD_FS = 20                                # coordinate bound labels (smallest text)
TICK_FS = 20                                 # colour-bar ticks
SIDE_FS = 22                                 # legend text, colour-bar label
HOUR_FS = 26                                 # hour window inside each map
LARGE_FS = 28                                # date heading

MRMS_HTTP = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"

PIPELINE_BUCKET = "indot-bridge-pipeline"
BRIDGE_KEY = "v1/analysis/bridge_coverage/bridge_coverage_flags.parquet"
ASSET_COL, LAT_COL, LON_COL = "Asset Name", "(16) Latitude:", "(17) Longitude:"
HIGHLIGHT_ASSET = "062-31-07183"

PRECIP_SITE_NO = "03294500"
PRECIP_LAT, PRECIP_LON = 38.280347, -85.799131

BRIDGE_COLOR = "#2ca02c"
HIGHLIGHT_COLOR = "#e31a1c"
PRECIP_DOT_COLOR = "#ffd400"
ALPHA_PRECIP = 0.55
QPE_VMIN, QPE_VMAX = 0.0, 4.0               # event peaks at 3.9 in/h; the GIF's 2.0 saturates

OUT_PNG = os.path.join("results", f"lanesville_{EVENT_DATE:%Y%m%d}_mrms_3panel.png")
DPI = 600

plt.rcParams.update({
    "font.family": "Times New Roman",          # the poster's font
    "mathtext.fontset": "stix",
    "font.size": LARGE_FS,
    "xtick.labelsize": COORD_FS,
    "ytick.labelsize": COORD_FS,
    "legend.fontsize": LARGE_FS,
})


# ── MRMS GRIB2 (template 3.0 grid, 5.41 PNG packing) ────────────────────────────

def _sm16(b: bytes) -> int:
    """GRIB2 sign-magnitude 16-bit integer."""
    v = struct.unpack(">H", b)[0]
    return -(v & 0x7FFF) if v & 0x8000 else v


def _sm32(b: bytes) -> int:
    v = struct.unpack(">I", b)[0]
    return -(v & 0x7FFFFFFF) if v & 0x80000000 else v


def decode_mrms_grib2(buf: bytes):
    """Return (values_mm, lats_desc, lons) for a single-field MRMS GRIB2 message.

    Missing / no-coverage (MRMS -3, and any negative value) becomes NaN."""
    if buf[:4] != b"GRIB" or buf[7] != 2:
        raise ValueError("not a GRIB2 message")
    p, sec = 16, {}
    while buf[p:p + 4] != b"7777":
        n = struct.unpack(">I", buf[p:p + 4])[0]
        sec[buf[p + 4]] = buf[p:p + n]
        p += n
    s3, s5, s6, s7 = sec[3], sec[5], sec[6], sec[7]

    if struct.unpack(">H", s3[12:14])[0] != 0:
        raise ValueError("expected a regular lat/lon grid (template 3.0)")
    ni, nj = struct.unpack(">II", s3[30:38])
    la1, lo1 = _sm32(s3[46:50]) * 1e-6, _sm32(s3[50:54]) * 1e-6
    di, dj = struct.unpack(">II", s3[63:71])
    di, dj = di * 1e-6, dj * 1e-6
    scan = s3[71]
    if scan != 0:
        raise ValueError(f"unexpected scanning mode {scan}")   # 0: W->E, N->S

    if struct.unpack(">H", s5[9:11])[0] != 41:
        raise ValueError("expected PNG packing (template 5.41)")
    ref = struct.unpack(">f", s5[11:15])[0]
    e, d = _sm16(s5[15:17]), _sm16(s5[17:19])
    if s6[5] != 255:
        raise ValueError("bitmap present; not handled")

    x = np.asarray(Image.open(io.BytesIO(s7[5:])), dtype=np.float64)
    if x.shape != (nj, ni):
        raise ValueError(f"PNG shape {x.shape} != grid {(nj, ni)}")
    vals = (ref + x * 2.0 ** e) / 10.0 ** d
    vals[vals < 0] = np.nan

    lons = lo1 + di * np.arange(ni)
    lons = np.where(lons > 180, lons - 360, lons)
    lats = la1 - dj * np.arange(nj)                          # descending (north first)
    return vals, lats, lons


def load_mrms_hour(hour: int):
    """Cropped 1-h QPE (inches) for EVENT_DATE hour, plus its pixel-edge extent."""
    ts = f"{EVENT_DATE:%Y%m%d}-{hour:02d}0000"
    url = (f"{MRMS_HTTP}/CONUS/{MRMS_PRODUCT}/{EVENT_DATE:%Y%m%d}/"
           f"MRMS_{MRMS_PRODUCT}_{ts}.grib2.gz")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    vals, lats, lons = decode_mrms_grib2(gzip.decompress(r.content))
    # Pad one pixel so the edge pixels fill the axes.
    ii = np.where((lats >= LAT_MIN - 0.01) & (lats <= LAT_MAX + 0.01))[0]
    jj = np.where((lons >= LON_MIN - 0.01) & (lons <= LON_MAX + 0.01))[0]
    crop = vals[np.ix_(ii, jj)] / 25.4                       # mm -> in
    dx, dy = abs(lons[1] - lons[0]), abs(lats[0] - lats[1])
    extent = [lons[jj[0]] - dx / 2, lons[jj[-1]] + dx / 2,
              lats[ii[-1]] - dy / 2, lats[ii[0]] + dy / 2]
    return crop, extent, (vals, lats, lons)


def pixel_value(full, lat, lon) -> float:
    vals, lats, lons = full
    return float(vals[np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))]) / 25.4


# ── Bridges ────────────────────────────────────────────────────────────────────

def load_bridges():
    df = pq.read_table(f"{PIPELINE_BUCKET}/{BRIDGE_KEY}",
                       filesystem=pafs.S3FileSystem(region="us-east-1"),
                       columns=[ASSET_COL, LAT_COL, LON_COL]).to_pandas()
    df.columns = ["asset", "lat", "lon"]
    df["asset"] = df["asset"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    hl = df[df["asset"] == HIGHLIGHT_ASSET]
    if hl.empty:
        raise SystemExit(f"highlight bridge {HIGHLIGHT_ASSET} not in {BRIDGE_KEY}")
    inbox = df[df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)
               & (df["asset"] != HIGHLIGHT_ASSET)]
    print(f"Bridges in view: {len(inbox)} + highlighted {HIGHLIGHT_ASSET} "
          f"({hl['lat'].iat[0]:.4f}, {hl['lon'].iat[0]:.4f})")
    return inbox[["lon", "lat"]].to_numpy(float), (float(hl["lon"].iat[0]), float(hl["lat"].iat[0]))


# ── Figure ─────────────────────────────────────────────────────────────────────

def hour_label(hour: int) -> str:
    """Hour window (local time) of the 1-h QPE ending at `hour` UTC."""
    t1 = pd.Timestamp(EVENT_DATE) + pd.Timedelta(hours=hour)
    t0 = t1 - pd.Timedelta(hours=1)
    l0, l1 = t0 + pd.Timedelta(hours=UTC_OFFSET_HR), t1 + pd.Timedelta(hours=UTC_OFFSET_HR)
    return f"{l0:%H}–{l1:%H} {TZ_LABEL}"


def make_figure(frames, bridges_xy, hl_lonlat):
    n = len(frames)
    lat0 = math.radians((LAT_MIN + LAT_MAX) / 2)
    ground_aspect = (LON_MAX - LON_MIN) * math.cos(lat0) / (LAT_MAX - LAT_MIN)   # w / h
    map_h = (MAX_FIG_H - TOP_PAD - DATE_H - (n - 1) * PANEL_GAP - XTICK_H) / n
    map_w = map_h * ground_aspect
    fig_w = LEFT_PAD + map_w + COL_GAP + RIGHT_W
    fig_h = MAX_FIG_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    print(f"Figure {fig_w:.2f} x {fig_h:.2f} in; each map {map_w:.2f} x {map_h:.2f} in")

    def box(x, y_top, w, h):                     # inches from top-left -> figure fraction
        return [x / fig_w, 1 - (y_top + h) / fig_h, w / fig_w, h / fig_h]

    cmap = mcolors.LinearSegmentedColormap.from_list("nws_precip", [
        (1.0, 1.0, 1.0, 0.0), "#b3d9ff", "#6ab4ff", "#1f78b4", "#33a02c", "#b2df8a",
        "#ffff33", "#ff7f00", "#e31a1c", "#fb9a99", "#6a0dad"], N=256)
    # 0 maps to the colormap's first colour, which is fully transparent.
    norm = mcolors.Normalize(vmin=QPE_VMIN, vmax=QPE_VMAX)       # fixed for every panel

    # Basemap once, reprojected to lon/lat, reused on every panel.
    # OpenStreetMap's tile server returns "Access blocked" tiles to contextily and
    # CartoDB now returns "API key required" tiles; Esri World Street Map serves.
    print("Fetching basemap tiles ...")
    img, ext = ctx.bounds2img(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, zoom=13, ll=True,
                              source=ctx.providers.Esri.WorldStreetMap)
    img, ext = ctx.warp_tiles(img, ext, "EPSG:4326")

    first_top = TOP_PAD + DATE_H
    im = None
    for k, (hour, (crop, extent)) in enumerate(frames.items()):
        y = first_top + k * (map_h + PANEL_GAP)
        ax = fig.add_axes(box(LEFT_PAD, y, map_w, map_h))
        ax.imshow(img, extent=ext, interpolation="bilinear", zorder=0)
        im = ax.imshow(crop, origin="upper", extent=extent, cmap=cmap, norm=norm,
                       alpha=ALPHA_PRECIP, interpolation="nearest", zorder=2)
        ax.scatter(bridges_xy[:, 0], bridges_xy[:, 1], marker="o", s=34,
                   facecolor=BRIDGE_COLOR, edgecolors="black", lw=0.5, zorder=5)
        ax.scatter([PRECIP_LON], [PRECIP_LAT], marker="o", s=240,
                   facecolor=PRECIP_DOT_COLOR, edgecolors="black", lw=1.3, zorder=7)
        ax.scatter([hl_lonlat[0]], [hl_lonlat[1]], marker="*", s=750,
                   facecolor=HIGHLIGHT_COLOR, edgecolors="black", lw=1.1, zorder=8)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("auto")                    # the axes box already has the ground aspect

        # Coordinates only at the extent bounds; labels kept inside the axis span.
        ax.set_yticks([LAT_MIN, LAT_MAX])
        ax.set_yticklabels([f"{LAT_MIN:.2f}°N", f"{LAT_MAX:.2f}°N"],
                           rotation=90, rotation_mode="default")
        lo, hi = ax.get_yticklabels()
        lo.set_va("bottom"); hi.set_va("top")
        ax.set_xticks([LON_MIN, LON_MAX])
        ax.set_xticklabels([f"{abs(LON_MIN):.2f}°W", f"{abs(LON_MAX):.2f}°W"])
        lo, hi = ax.get_xticklabels()
        lo.set_ha("left"); hi.set_ha("right")
        ax.tick_params(length=0, pad=4, labelbottom=(k == n - 1))

        ax.text(0.985, 0.03, hour_label(hour), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=HOUR_FS, linespacing=1.15, zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", lw=0.8, alpha=0.9))

    # Date once, above the top map.
    fig.text((LEFT_PAD + map_w / 2) / fig_w, 1 - (TOP_PAD + DATE_H / 2) / fig_h,
             f"{EVENT_DATE:%B} {EVENT_DATE.day}, {EVENT_DATE:%Y}",
             ha="center", va="center", fontsize=LARGE_FS, fontweight="bold")

    # Right column: legend at the top, colour bar below it down to the last map.
    x_r = LEFT_PAD + map_w + COL_GAP
    ax_leg = fig.add_axes(box(x_r, first_top, RIGHT_W, LEGEND_H))
    ax_leg.axis("off")
    handles = [
        Line2D([], [], ls="", marker="o", ms=16, mfc=BRIDGE_COLOR, mec="black", mew=1.0),
        Line2D([], [], ls="", marker="*", ms=32, mfc=HIGHLIGHT_COLOR, mec="black", mew=1.3),
        Line2D([], [], ls="", marker="o", ms=22, mfc=PRECIP_DOT_COLOR, mec="black", mew=1.5),
    ]
    ax_leg.legend(handles, ["Bridge", "Focus\nBridge", "Precip\nStation"],
                  loc="upper left", frameon=False, borderaxespad=0, handletextpad=0.4,
                  labelspacing=0.7, borderpad=0, fontsize=SIDE_FS)

    cb_top = first_top + LEGEND_H + 0.40
    cax = fig.add_axes(box(x_r + 0.15, cb_top, CBAR_W, CBAR_LEN))
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks(np.arange(QPE_VMIN, QPE_VMAX + 1e-9, 1.0))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    cb.set_label("MRMS 1-h rainfall (in)", fontsize=SIDE_FS, labelpad=8)
    cb.ax.tick_params(labelsize=TICK_FS, length=5, width=1)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    # Transparent canvas for the poster; no bbox_inches: keep the exact canvas size.
    fig.savefig(OUT_PNG, dpi=DPI, transparent=True)
    plt.close(fig)
    w_px, h_px = Image.open(OUT_PNG).size
    print(f"Saved {OUT_PNG}: {w_px} x {h_px} px "
          f"({w_px / DPI:.2f} x {h_px / DPI:.2f} in at {DPI} dpi; "
          f"{DPI // 2} dpi when printed at 2x)")


def main():
    global HOURS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=int, nargs="+", default=HOURS,
                    help="MRMS file hours, UTC (default 14 15 16)")
    HOURS = ap.parse_args().hours

    bridges_xy, hl = load_bridges()
    frames = {}
    for h in HOURS:
        crop, extent, full = load_mrms_hour(h)
        frames[h] = (crop, extent)
        print(f"  {h:02d} UTC: view max {np.nanmax(crop):.2f} in | gauge pixel "
              f"{pixel_value(full, PRECIP_LAT, PRECIP_LON):.2f} in | {HIGHLIGHT_ASSET} pixel "
              f"{pixel_value(full, hl[1], hl[0]):.2f} in")
    make_figure(frames, bridges_xy, hl)


if __name__ == "__main__":
    main()
