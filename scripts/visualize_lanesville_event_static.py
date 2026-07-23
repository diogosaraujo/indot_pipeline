"""visualize_lanesville_event_static.py

Static 3-panel PNG (6.5 x 4 in) for the Lanesville, IN flash flood — June 9, 2026.
This is the still-image sibling of visualize_lanesville_event.py (which renders a GIF).

Layout (GridSpec 2x2, map spans both rows on the left):

  Panel 1 (left, full height)  MRMS QPE 1-h (Pass2) for the EVENT PEAK HOUR (15 UTC).
      Basemap + bridges (green), scour-critical bridges (pink) and the USGS precip
      gauge (yellow). The scour-critical bridge 062-31-07183 is HIGHLIGHTED with a
      large red star (sized like the gauge marker).

  Panel 2 (top-right)   Hourly precip readings at USGS 03294500 (the gauge dot).
  Panel 3 (bottom-right) Hourly MRMS 1-h QPE at the grid pixel nearest bridge 062-31-07183.
      Panels 2 & 3 are STACKED (not side by side) but together span the width panels
      2+3 would occupy side by side, and share the event-hour x-axis.

  Both time-series carry horizontal reference lines at the two Atlas-14 1-hour
  precipitation depths bracketing the recorded peak (the two thresholds nearest the
  values recorded — not the whole ladder).

Fonts: as large as fits without crowding; nothing below 7 pt (labels abbreviated).

Reads (public S3, no credentials):
    noaa-mrms-pds  —  MultiSensor_QPE_01H_Pass2_00.00
Reads (pipeline bucket, IAM role):
    precip/usgs/precip_iv.parquet, atlas14/precipitation_frequency.parquet,
    stations/indiana_streamflow_sites.parquet, bridge_coverage_flags.parquet

Writes:
    results/lanesville_20260609_static.png
    s3://indot-bridge-pipeline/v1/events/lanesville_06_09_2026/lanesville_20260609_static.png

Run on EC2 (uploads to S3):
    python scripts/visualize_lanesville_event_static.py
    python scripts/visualize_lanesville_event_static.py --peak-hour 15 --no-upload
"""
from __future__ import annotations

import argparse
import bisect
import io
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import s3fs
from scipy.interpolate import griddata

# Pipeline utilities (script lives alongside utils.py)
sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_units, canonicalize_mrms_grid, decompress_gz, open_mrms_grib

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("WARNING: contextily not installed — basemap tiles will be omitted.")
    print("         Install with:  pip install contextily")

# ── Configuration ──────────────────────────────────────────────────────────────
EVENT_DATE   = date(2026, 6, 9)
PEAK_HOUR    = 15                    # UTC hour shown on the map panel (event peak)
# Zoomed-out view so the precip gauge, bridges and the highlighted bridge all fit.
LON_MIN, LON_MAX = -86.10, -85.72
LAT_MIN, LAT_MAX = 38.12, 38.34
HOURS        = list(range(6, 23))   # 06–22 UTC window for the time-series panels
UTC_OFFSET_HR = -4                  # Harrison Co., IN is Eastern; June → EDT (UTC-4)
TZ_LABEL      = "EDT"               # axes/titles are labelled in local time

MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_1H_FOLDER  = "MultiSensor_QPE_01H_Pass2_00.00"

# Pipeline S3 bucket (output destination + inputs)
PIPELINE_BUCKET  = "indot-bridge-pipeline"
PIPELINE_PREFIX  = "v1/"
S3_EVENT_FOLDER  = f"v1/events/lanesville_{EVENT_DATE:%m_%d_%Y}/"
PNG_FNAME        = f"lanesville_{EVENT_DATE:%Y%m%d}_static.png"

OUT_DIR      = "results"
OUT_PNG      = os.path.join(OUT_DIR, PNG_FNAME)
ALPHA_PRECIP = 0.60     # transparency of the MRMS overlay
QPE_1H_VMAX  = 2.0      # precip colour scale (inches, 1-h)

# ── Precip gauge (top-right panel) + Atlas-14 ───────────────────────────────────
PRECIP_SITE_NO   = "03294500"
PRECIP_REF_LAT   = 38.280347
PRECIP_REF_LON   = -85.799131
USGS_PRECIP_KEY  = "precip/usgs/precip_iv.parquet"
USGS_NODATA      = -999999.0
ATLAS14_KEY      = "atlas14/precipitation_frequency.parquet"
INV_KEY          = "stations/indiana_streamflow_sites.parquet"
ATLAS14_DUR_HR   = 1                                   # hourly readings → 1-h Atlas-14 ladder
PRECIP_RPS       = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
MAX_HOURLY_IN    = 12.0
HYETO_COLOR      = "#1f78b4"                            # blue bars (both time-series)
THRESH_COLOR     = "#e31a1c"                            # red dashed Atlas-14 lines
PRECIP_DOT_COLOR = "#ffd400"                            # yellow gauge dot

# ── Bridges (from the processed bridge_location.csv) ────────────────────────────
BRIDGE_KEY        = "analysis/bridge_coverage/bridge_coverage_flags.parquet"
ASSET_COL         = "Asset Name"
BRIDGE_LAT_COL    = "(16) Latitude:"
BRIDGE_LON_COL    = "(17) Longitude:"
BRIDGE_SCOUR_COL  = "scour_critical"
BRIDGE_COLOR      = "#2ca02c"                           # green — regular bridges
SCOUR_COLOR       = "#e377c2"                           # pink  — scour-critical bridges
HIGHLIGHT_COLOR   = "#e31a1c"                           # red star — the featured bridge
HIGHLIGHT_ASSET   = "062-31-07183"                     # matched in the bridge table
HIGHLIGHT_LABEL   = "Bridge 022310"                    # short label (legend + bottom-panel title)

os.makedirs(OUT_DIR, exist_ok=True)

# Fonts: as large as fits; never below 7.
plt.rcParams.update({
    "font.size":        7.5,
    "axes.titlesize":   8.5,
    "axes.labelsize":   8.0,
    "xtick.labelsize":  7.0,
    "ytick.labelsize":  7.0,
    "legend.fontsize":  7.0,
    "figure.titlesize": 9.0,
})


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _anon_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True)


def upload_png_to_s3(local_path: str) -> str:
    key = S3_EVENT_FOLDER + PNG_FNAME
    s3 = boto3.client("s3", region_name="us-east-1")
    print(f"Uploading to s3://{PIPELINE_BUCKET}/{key} ...")
    s3.upload_file(local_path, PIPELINE_BUCKET, key,
                   ExtraArgs={"ContentType": "image/png"})
    uri = f"s3://{PIPELINE_BUCKET}/{key}"
    print(f"Saved to {uri}")
    return uri


def _read_pipeline_filtered(key: str, columns=None, filters=None) -> pd.DataFrame:
    """Predicate-pushdown read from the pipeline bucket (avoids pulling the whole file)."""
    path = f"{PIPELINE_BUCKET}/{PIPELINE_PREFIX}{key}"
    return pq.read_table(path, filesystem=pafs.S3FileSystem(),
                         columns=columns, filters=filters).to_pandas()


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ── USGS precip gauge (03294500) → hourly readings (top-right panel) ────────────

def load_precip_station() -> tuple[Optional[dict], Optional[pd.Series]]:
    """USGS gauge PRECIP_SITE_NO and its HOURLY precip series (in) over the event day
    ±1 day.  USGS IV param 00045 is already in INCHES and reports INCREMENTAL depth per
    15-min bin → hourly value = SUM of increments in the hour; label = bin END to match
    MRMS QPE_01H.  Returns (info, hourly) or (None, None)."""
    print(f"\n── USGS precip gauge {PRECIP_SITE_NO} (from {USGS_PRECIP_KEY}) ──")
    day0 = pd.Timestamp(EVENT_DATE, tz="UTC") - pd.Timedelta(days=1)
    day1 = pd.Timestamp(EVENT_DATE, tz="UTC") + pd.Timedelta(days=1)
    try:
        df = _read_pipeline_filtered(
            USGS_PRECIP_KEY,
            columns=["site_no", "datetime_utc", "precip_in", "station_nm", "latitude", "longitude"],
            filters=[("site_no", "=", PRECIP_SITE_NO),
                     ("datetime_utc", ">=", day0.to_pydatetime()),
                     ("datetime_utc", "<",  day1.to_pydatetime())],
        )
    except Exception as e:                                   # noqa: BLE001
        print(f"  USGS precip read failed: {e}")
        return None, None
    if df.empty:
        print(f"  No USGS precip for site {PRECIP_SITE_NO} around {EVENT_DATE}. "
              f"Run: python scripts/13_download_usgs_precip.py --site {PRECIP_SITE_NO}")
        return None, None

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["precip_in"]    = pd.to_numeric(df["precip_in"], errors="coerce")
    s = df.set_index("datetime_utc")["precip_in"].astype(float)
    s = s[(s > USGS_NODATA + 1.0) & (s >= 0)].sort_index()
    hourly = s.resample("1h", label="right", closed="right").sum().fillna(0.0)
    hourly = hourly[hourly <= MAX_HOURLY_IN]

    lat = float(pd.to_numeric(df["latitude"], errors="coerce").dropna().iloc[0]) \
        if df["latitude"].notna().any() else PRECIP_REF_LAT
    lon = float(pd.to_numeric(df["longitude"], errors="coerce").dropna().iloc[0]) \
        if df["longitude"].notna().any() else PRECIP_REF_LON
    nm = str(df["station_nm"].dropna().iloc[0]) if df["station_nm"].notna().any() \
        else f"USGS {PRECIP_SITE_NO}"
    info = {"site_no": PRECIP_SITE_NO, "name": nm, "lat": lat, "lon": lon,
            "dist_km": _haversine_km(PRECIP_REF_LAT, PRECIP_REF_LON, lat, lon)}
    print(f"  {info['site_no']} {info['name']}: {len(s)} IV obs → "
          f"{int((hourly > 0).sum())} wet hours; peak 1-h {float(hourly.max()):.2f} in.")
    return info, hourly


# ── Bridges + the highlighted bridge ────────────────────────────────────────────

def load_bridges(highlight_asset: str) -> tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Return (regular_lonlat, scour_lonlat, highlight) for the view bbox.

    `highlight` is {asset, lat, lon} for `highlight_asset` (resolved from the FULL table
    so it is found even if it sits on the bbox edge); it is EXCLUDED from the regular /
    scour point arrays so the big star is the only marker drawn there."""
    try:
        df = _read_pipeline_filtered(
            BRIDGE_KEY,
            columns=[ASSET_COL, BRIDGE_LAT_COL, BRIDGE_LON_COL, BRIDGE_SCOUR_COL])
    except Exception as e:                                   # noqa: BLE001
        print(f"  Bridge read failed: {e}")
        return np.empty((0, 2)), np.empty((0, 2)), None
    df = df.rename(columns={ASSET_COL: "asset", BRIDGE_LAT_COL: "lat",
                            BRIDGE_LON_COL: "lon", BRIDGE_SCOUR_COL: "scour"})
    df["asset"] = df["asset"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    hl_rows = df[df["asset"] == highlight_asset]
    highlight: Optional[dict] = None
    if not hl_rows.empty:
        r = hl_rows.iloc[0]
        highlight = {"asset": highlight_asset, "lat": float(r["lat"]), "lon": float(r["lon"])}
        print(f"  Highlight bridge {highlight_asset} at "
              f"({highlight['lat']:.5f}, {highlight['lon']:.5f})")
    else:
        print(f"  WARNING: highlight asset '{highlight_asset}' not found in {BRIDGE_KEY}")

    inbox = df[df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)]
    scour = inbox["scour"].fillna(False).astype(bool)
    not_hl = inbox["asset"] != highlight_asset
    reg = inbox.loc[~scour & not_hl, ["lon", "lat"]].to_numpy(float)
    sc  = inbox.loc[scour & not_hl,  ["lon", "lat"]].to_numpy(float)
    print(f"  Bridges in view: {len(inbox)} ({int(scour.sum())} scour-critical)")
    return reg, sc, highlight


# ── Atlas-14 1-hour depth ladder at a point ─────────────────────────────────────

def atlas14_1h_depths(lat: float, lon: float) -> dict[int, float]:
    """Atlas-14 ATLAS14_DUR_HR-hour depth (in) interpolated to (lat,lon) per return period."""
    a14 = _read_pipeline_filtered(ATLAS14_KEY)
    a14["site_no"] = a14["site_no"].astype(str)
    a14 = a14[a14["duration_hr"] == ATLAS14_DUR_HR]
    inv = _read_pipeline_filtered(INV_KEY, columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    a14 = a14.merge(inv, on="site_no", how="left").dropna(
        subset=["dec_lat_va", "dec_long_va", "depth_in"])
    depths: dict[int, float] = {}
    for rp in PRECIP_RPS:
        g = a14[a14["return_period_yr"] == rp]
        if len(g) < 3:
            continue
        src = g[["dec_lat_va", "dec_long_va"]].to_numpy(float)
        val = g["depth_in"].to_numpy(float)
        d = griddata(src, val, [[lat, lon]], method="linear")[0]
        if not np.isfinite(d):
            d = griddata(src, val, [[lat, lon]], method="nearest")[0]
        if np.isfinite(d):
            depths[rp] = float(d)
    return depths


def two_nearest_thresholds(depths: dict[int, float], peak: float) -> list[tuple[int, float]]:
    """The two Atlas-14 (RP, depth) pairs bracketing `peak` — largest depth ≤ peak and
    smallest depth > peak.  At the ends of the ladder, returns the two closest."""
    items = sorted(depths.items(), key=lambda kv: kv[1])     # ascending by depth
    if len(items) <= 2:
        return items
    dv = [v for _, v in items]
    j = bisect.bisect_right(dv, peak)                        # first depth strictly > peak
    lo = min(max(j - 1, 0), len(items) - 2)
    return items[lo:lo + 2]


# ── MRMS frames (map peak hour + hourly series at the bridge pixel) ──────────────

def _mrms_key(hour: int) -> str:
    ts = f"{EVENT_DATE:%Y%m%d}-{hour:02d}0000"
    fname = f"MRMS_{MRMS_1H_FOLDER}_{ts}.grib2.gz"
    return f"{MRMS_BUCKET}/CONUS/{MRMS_1H_FOLDER}/{EVENT_DATE:%Y%m%d}/{fname}"


def _open_mrms(fs: s3fs.S3FileSystem, hour: int):
    """Download + decompress one 1-h MRMS GRIB → (arr_in, lats, lons) or None."""
    key = _mrms_key(hour)
    try:
        with fs.open(key, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"  [MRMS h{hour:02d}] file not found: {key}")
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(decompress_gz(raw))
        tmp_path = tmp.name
    try:
        ds  = open_mrms_grib(tmp_path)
        var = list(ds.data_vars)[0]
        arr, lats, lons = canonicalize_mrms_grid(ds[var])
        arr = apply_units(arr, kind="qpe", units="in")
    finally:
        os.unlink(tmp_path)
    return arr, lats, lons


def load_mrms(fs: s3fs.S3FileSystem, bridge_lat: float, bridge_lon: float):
    """Return (peak_frame_in, lats_crop, lons_crop, bridge_hourly_in).

    peak_frame_in is the cropped MRMS field for PEAK_HOUR (map panel).
    bridge_hourly_in is the 1-h QPE at the pixel nearest the bridge over HOURS."""
    print(f"\n── Loading MRMS QPE_01H_Pass2 for {EVENT_DATE} ({HOURS[0]:02d}–{HOURS[-1]:02d} UTC) ──")
    peak_frame = lats_crop = lons_crop = None
    bridge_hourly = np.full(len(HOURS), np.nan)
    bi = bj = None
    for k, h in enumerate(HOURS):
        print(f"  Loading hour {h:02d} UTC...", end="\r")
        got = _open_mrms(fs, h)
        if got is None:
            continue
        arr, lats, lons = got
        lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
        lon_idx = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
        cropped = arr[np.ix_(lat_idx, lon_idx)]
        cropped = np.where(np.isnan(cropped), 0.0, cropped)
        if lats_crop is None:                                # fix crop grid + bridge pixel once
            lats_crop = lats[lat_idx]
            lons_crop = lons[lon_idx]
            bi = int(np.argmin(np.abs(lats_crop - bridge_lat)))
            bj = int(np.argmin(np.abs(lons_crop - bridge_lon)))
            print(f"  Bridge MRMS pixel: lat {lats_crop[bi]:.4f} lon {lons_crop[bj]:.4f}   ")
        bridge_hourly[k] = float(cropped[bi, bj])
        if h == PEAK_HOUR:
            peak_frame = cropped
    n_ok = int(np.isfinite(bridge_hourly).sum())
    print(f"  Loaded {n_ok}/{len(HOURS)} MRMS frames; peak-hour ({PEAK_HOUR}Z) "
          f"frame {'OK' if peak_frame is not None else 'MISSING'}.        ")
    return peak_frame, lats_crop, lons_crop, bridge_hourly


# ── Time-series panel helper ────────────────────────────────────────────────────

def _draw_series(ax, xh, values, title, a14_depths, bar_color, title_fs=None):
    """Hourly bar series + the two bracketing Atlas-14 1-h thresholds."""
    vals = np.nan_to_num(np.asarray(values, float))
    ax.bar(xh, vals, width=0.85, color=bar_color, alpha=0.75, zorder=2)
    peak = float(vals.max()) if vals.size else 0.0
    kpk  = int(np.argmax(vals)) if vals.size else 0

    thresh = two_nearest_thresholds(a14_depths, peak) if a14_depths else []
    top_ref = max((d for _, d in thresh), default=0.0)
    ax.set_ylim(0, max(peak, top_ref) * 1.28 if max(peak, top_ref) > 0 else 0.3)

    for i, (rp, d) in enumerate(thresh):
        ax.axhline(d, color=THRESH_COLOR, ls="--", lw=1.1, zorder=3)
        # Lower threshold labelled below its line, upper above → no overlap when close.
        va = "top" if i == 0 else "bottom"
        ax.text(0.985, d, f"{rp}-yr {d:.2f}\"", transform=ax.get_yaxis_transform(),
                ha="right", va=va, fontsize=7, color=THRESH_COLOR, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    if peak > 0:
        ax.plot([xh[kpk]], [peak], "v", color=THRESH_COLOR, ms=5, zorder=6)

    ax.set_ylabel("Precip (in)")
    ax.set_title(title, pad=3, fontsize=title_fs)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.margins(x=0.01)


# ── Figure ──────────────────────────────────────────────────────────────────────

def make_figure(peak_frame, precip_hourly, bridge_hourly,
                a14_gauge, a14_bridge, reg_xy, sc_xy, highlight, gauge_lonlat) -> None:
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    precip_colors = [
        (1.0, 1.0, 1.0, 0.0),
        "#b3d9ff", "#6ab4ff", "#1f78b4",
        "#33a02c", "#b2df8a",
        "#ffff33", "#ff7f00",
        "#e31a1c", "#fb9a99",
        "#6a0dad",
    ]
    precip_cmap = mcolors.LinearSegmentedColormap.from_list("nws_precip", precip_colors, N=256)
    precip_cmap.set_under("none")
    norm_1h = mcolors.Normalize(vmin=0.01, vmax=QPE_1H_VMAX)

    fig = plt.figure(figsize=(6.5, 4.0), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.05)
    # Left column: map (large) + legend strip + horizontal colorbar strip, stacked so
    # the legend sits right above the colorbar beneath the map.
    gl = outer[0].subgridspec(3, 1, height_ratios=[1.0, 0.22, 0.11], hspace=0.06)
    ax_map = fig.add_subplot(gl[0])
    ax_leg = fig.add_subplot(gl[1]); ax_leg.axis("off")
    cax    = fig.add_subplot(gl[2])
    # Right column: two stacked time-series spanning the full figure height (as tall as
    # the whole left column / map block).
    gr = outer[1].subgridspec(2, 1, hspace=0.34)
    ax_top = fig.add_subplot(gr[0])
    ax_bot = fig.add_subplot(gr[1], sharex=ax_top)

    # ── Panel 1: MRMS peak-hour map ─────────────────────────────────────────────
    ax_map.set_xlim(LON_MIN, LON_MAX)
    ax_map.set_ylim(LAT_MIN, LAT_MAX)
    ax_map.set_aspect("auto")
    ax_map.set_xticks([]); ax_map.set_yticks([])           # declutter the small map
    local_peak = (PEAK_HOUR + UTC_OFFSET_HR) % 24
    ax_map.set_title(f"MRMS 1-h QPE · {local_peak:02d} {TZ_LABEL} {EVENT_DATE:%m-%d}")

    if HAS_CTX:
        try:
            ctx.add_basemap(ax_map, crs="EPSG:4326",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom=12, attribution=False, zorder=0)
        except Exception as e:                              # noqa: BLE001
            print(f"  Basemap warning: {e}")

    if peak_frame is not None:
        im = ax_map.imshow(peak_frame, origin="upper", extent=extent,
                           cmap=precip_cmap, norm=norm_1h, aspect="auto",
                           alpha=ALPHA_PRECIP, zorder=2, interpolation="nearest")
    else:                                                   # keep the colorbar meaningful
        im = ax_map.imshow(np.zeros((2, 2)), origin="upper", extent=extent,
                           cmap=precip_cmap, norm=norm_1h, aspect="auto", zorder=2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("1-h rain (in)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # Bridges, gauge, and the highlighted bridge. The highlight is a HOLLOW red star
    # (facecolor="none") so a nearby/overlapping gauge dot stays visible underneath.
    handles, labels = [], []
    if len(reg_xy):
        h = ax_map.scatter(reg_xy[:, 0], reg_xy[:, 1], marker="o", s=12,
                           facecolor=BRIDGE_COLOR, edgecolors="black", lw=0.3, zorder=5)
        handles.append(h); labels.append("Bridge")
    if len(sc_xy):
        h = ax_map.scatter(sc_xy[:, 0], sc_xy[:, 1], marker="o", s=18,
                           facecolor=SCOUR_COLOR, edgecolors="black", lw=0.4, zorder=6)
        handles.append(h); labels.append("Scour-crit.")
    h = ax_map.scatter([gauge_lonlat[0]], [gauge_lonlat[1]], marker="o", s=70,
                       facecolor=PRECIP_DOT_COLOR, edgecolors="black", lw=0.8, zorder=7)
    handles.append(h); labels.append("Precip gauge")
    if highlight is not None:
        h = ax_map.scatter([highlight["lon"]], [highlight["lat"]], marker="*", s=320,
                           facecolor="none", edgecolors=HIGHLIGHT_COLOR, lw=1.8, zorder=9)
        handles.append(h); labels.append(HIGHLIGHT_LABEL)
    # Legend beneath the map, right above the colorbar, split into 2 columns.
    ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=7, framealpha=0.9,
                  labelspacing=0.3, columnspacing=1.0, handletextpad=0.3, borderpad=0.4)

    # ── Panel 2 (top-right): USGS gauge hourly precip ───────────────────────────
    xh = np.arange(len(HOURS))
    _draw_series(ax_top, xh, precip_hourly,
                 "USGS 03294500 Gauge - Ohio River at Louisville", a14_gauge,
                 HYETO_COLOR, title_fs=8.0)

    # ── Panel 3 (bottom-right): MRMS hourly at the bridge pixel ──────────────────
    _draw_series(ax_bot, xh, bridge_hourly,
                 f"MRMS 1-h At {HIGHLIGHT_LABEL}", a14_bridge, HYETO_COLOR)

    # Shared x-axis in LOCAL time (only the bottom panel shows hour labels).
    ax_bot.set_xticks(xh[::3])
    ax_bot.set_xticklabels([f"{(h + UTC_OFFSET_HR) % 24:02d}" for h in HOURS[::3]])
    ax_bot.set_xlabel(f"Hour ({TZ_LABEL})")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # No bbox_inches="tight" — keep the canvas at exactly 6.5 x 4 in (constrained_layout
    # already packs the axes to fill it), so the physical figure size is as requested.
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"\nSaved locally: {OUT_PNG}  ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    global PEAK_HOUR, HIGHLIGHT_ASSET
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peak-hour", type=int, default=PEAK_HOUR,
                    help=f"UTC hour for the map panel (default {PEAK_HOUR})")
    ap.add_argument("--asset", default=HIGHLIGHT_ASSET,
                    help=f"scour-critical bridge to highlight (default {HIGHLIGHT_ASSET})")
    ap.add_argument("--no-upload", action="store_true", help="skip the S3 upload")
    args = ap.parse_args()

    PEAK_HOUR = args.peak_hour
    HIGHLIGHT_ASSET = args.asset
    if PEAK_HOUR not in HOURS:
        print(f"NOTE: peak hour {PEAK_HOUR} is outside the series window {HOURS[0]}–{HOURS[-1]}Z; "
              "the map frame will still be fetched but no bar aligns to it.")

    fs = _anon_fs()

    # Gauge hourly precip + Atlas-14 at the gauge (top-right panel)
    station_info, hyeto = load_precip_station()
    gauge_lat = station_info["lat"] if station_info else PRECIP_REF_LAT
    gauge_lon = station_info["lon"] if station_info else PRECIP_REF_LON
    hour_ts = [pd.Timestamp(EVENT_DATE, tz="UTC") + pd.Timedelta(hours=h) for h in HOURS]
    precip_hourly = (hyeto.reindex(hour_ts).to_numpy(float)
                     if hyeto is not None else np.full(len(HOURS), np.nan))

    # Bridges + highlighted bridge
    reg_xy, sc_xy, highlight = load_bridges(HIGHLIGHT_ASSET)
    bridge_lat = highlight["lat"] if highlight else gauge_lat
    bridge_lon = highlight["lon"] if highlight else gauge_lon

    # MRMS: peak-hour map frame + hourly series at the bridge pixel
    peak_frame, _lats, _lons, bridge_hourly = load_mrms(fs, bridge_lat, bridge_lon)

    # Atlas-14 1-h ladders at the gauge and at the bridge
    print("\n── Atlas-14 1-h depth ladders ──")
    a14_gauge  = atlas14_1h_depths(gauge_lat, gauge_lon)
    a14_bridge = atlas14_1h_depths(bridge_lat, bridge_lon)
    print(f"  gauge : { {r: round(v,2) for r,v in sorted(a14_gauge.items())} }")
    print(f"  bridge: { {r: round(v,2) for r,v in sorted(a14_bridge.items())} }")

    make_figure(peak_frame, precip_hourly, bridge_hourly,
                a14_gauge, a14_bridge, reg_xy, sc_xy, highlight,
                (gauge_lon, gauge_lat))

    if not args.no_upload:
        upload_png_to_s3(OUT_PNG)
    print("\nDone.")


if __name__ == "__main__":
    main()
