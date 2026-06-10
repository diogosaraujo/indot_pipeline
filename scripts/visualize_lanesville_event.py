"""visualize_lanesville_event.py

3-panel animated GIF for the Lanesville, IN flash flood — June 9, 2026.

  Panel 1: MRMS QPE 1-hour (Pass2) — hourly rainfall depth
  Panel 2: MRMS QPE 3-hour rolling accumulation (Pass2)
  Panel 3: NWM analysis streamflow on NHD flowlines

Before running (if not already installed):
    pip install contextily pillow

Reads (public S3, no credentials needed):
    noaa-mrms-pds  —  MultiSensor_QPE_01H_Pass2_00.00
    noaa-nwm-pds   —  analysis_assim channel_rt

Writes:
    results/lanesville_20260609.gif
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import requests
import s3fs
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

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
LON_MIN, LON_MAX = -86.06, -85.88
LAT_MIN, LAT_MAX = 38.15, 38.31
HOURS        = list(range(24))      # 00–23 UTC

MRMS_BUCKET     = "noaa-mrms-pds"
MRMS_1H_FOLDER  = "MultiSensor_QPE_01H_Pass2_00.00"
NWM_BUCKET      = "noaa-nwm-pds"

OUT_DIR      = "results"
OUT_GIF      = os.path.join(OUT_DIR, f"lanesville_{EVENT_DATE:%Y%m%d}.gif")
FRAME_MS     = 600      # ms per frame in the GIF
ALPHA_PRECIP = 0.55     # transparency of the MRMS overlay

# Precipitation colour scale (inches)
QPE_1H_VMAX  = 2.0
QPE_3H_VMAX  = 5.0

# NWM streamflow colour scale (m³/s, log)
NWM_Q_VMIN   = 0.5
NWM_Q_VMAX   = 500.0

os.makedirs(OUT_DIR, exist_ok=True)


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _anon_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True)


# ── Step 1: MRMS ───────────────────────────────────────────────────────────────

def _mrms_key(hour: int) -> str:
    ts = f"{EVENT_DATE:%Y%m%d}-{hour:02d}0000"
    fname = f"MRMS_{MRMS_1H_FOLDER}_{ts}.grib2.gz"
    return f"{MRMS_BUCKET}/CONUS/{MRMS_1H_FOLDER}/{EVENT_DATE:%Y%m%d}/{fname}"


def load_mrms_frame(fs: s3fs.S3FileSystem, hour: int) -> Optional[np.ndarray]:
    """Download one 1-h MRMS GRIB, decompress, return cropped 2-D array (inches)."""
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

    # Crop to event bounding box
    lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
    lon_idx = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
    cropped = arr[np.ix_(lat_idx, lon_idx)]
    return np.where(np.isnan(cropped), 0.0, cropped)


def load_all_mrms(fs: s3fs.S3FileSystem) -> tuple[list, np.ndarray, np.ndarray]:
    """Return (frames_1h, lats_crop, lons_crop).

    frames_1h is a list of 24 arrays (None where file is missing).
    Also returns the lat/lon vectors for the first successful frame so the
    caller can build the plot extent.
    """
    print(f"\n── Loading MRMS QPE_01H_Pass2 for {EVENT_DATE} ──────────────────────────")

    # Derive crop lat/lon from the first available file to get exact grid coords
    lats_crop = lons_crop = None
    key0 = _mrms_key(0)
    try:
        with fs.open(key0, "rb") as f:
            raw = f.read()
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(decompress_gz(raw))
            tmp_path = tmp.name
        ds   = open_mrms_grib(tmp_path)
        var  = list(ds.data_vars)[0]
        _, lats_full, lons_full = canonicalize_mrms_grid(ds[var])
        os.unlink(tmp_path)
        lats_crop = lats_full[(lats_full >= LAT_MIN) & (lats_full <= LAT_MAX)]
        lons_crop = lons_full[(lons_full >= LON_MIN) & (lons_full <= LON_MAX)]
    except Exception as e:
        print(f"  WARNING: could not probe grid from hour-00 file: {e}")

    frames: list[Optional[np.ndarray]] = []
    for h in HOURS:
        print(f"  Loading hour {h:02d} UTC...", end="\r")
        frames.append(load_mrms_frame(fs, h))
    print(f"  Loaded {sum(f is not None for f in frames)}/24 MRMS frames.      ")

    return frames, lats_crop, lons_crop


def compute_3h_rolling(frames_1h: list) -> list:
    """Return 3-h rolling accumulation: sum of frames[h-2], [h-1], [h]."""
    result = []
    for h in range(len(frames_1h)):
        stack = [frames_1h[i] for i in range(max(0, h - 2), h + 1)
                 if frames_1h[i] is not None]
        result.append(np.sum(stack, axis=0) if stack else None)
    return result


# ── Step 2: NHD flowlines ──────────────────────────────────────────────────────

def fetch_nhd_flowlines() -> gpd.GeoDataFrame:
    """Query USGS National Map ArcGIS REST for NHDPlus flowlines in the bbox."""
    print("\n── Fetching NHD flowlines ───────────────────────────────────────────────")
    url = ("https://hydro.nationalmap.gov/arcgis/rest/services"
           "/nhd/MapServer/6/query")
    params = {
        "geometry":      f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}",
        "geometryType":  "esriGeometryEnvelope",
        "inSR":          "4326",
        "outSR":         "4326",
        "spatialRel":    "esriSpatialRelIntersects",
        "outFields":     "*",
        "returnGeometry": "true",
        "f":             "geojson",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    gdf = gpd.read_file(io.StringIO(r.text))
    if gdf.empty:
        raise RuntimeError("NHD query returned no flowlines — check bbox or service.")

    # Normalise COMID field (varies by service version)
    for candidate in ("COMID", "comid", "NHDPlusID", "nhdplusid"):
        if candidate in gdf.columns:
            gdf = gdf.rename(columns={candidate: "comid"})
            break
    if "comid" not in gdf.columns:
        print("  WARNING: no COMID field found; NWM matching will be skipped.")
        gdf["comid"] = np.nan

    gdf["comid"] = gdf["comid"].astype("Int64")
    gdf = gdf.to_crs(epsg=4326)
    print(f"  {len(gdf)} flowline features retrieved.")
    return gdf


def _geom_segments(geom) -> list:
    """Extract [[x1,y1],[x2,y2]] pairs from a LineString or MultiLineString."""
    segs = []
    if geom is None:
        return segs
    if geom.geom_type == "LineString":
        coords = np.array(geom.coords)
        segs = [coords[i : i + 2] for i in range(len(coords) - 1)]
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            coords = np.array(part.coords)
            segs.extend([coords[i : i + 2] for i in range(len(coords) - 1)])
    return segs


def build_line_collection_data(gdf: gpd.GeoDataFrame):
    """Pre-compute segment arrays and per-segment COMID mapping."""
    all_segs: list = []
    seg_comids: list = []
    row_seg_counts: list = []  # how many segments belong to each GDF row
    for _, row in gdf.iterrows():
        segs = _geom_segments(row.geometry)
        all_segs.extend(segs)
        comid = int(row["comid"]) if not (row["comid"] is None or row["comid"] != row["comid"]) else -1
        seg_comids.extend([comid] * len(segs))
        row_seg_counts.append(len(segs))
    return all_segs, np.array(seg_comids), row_seg_counts


# ── Step 3: NWM streamflow ─────────────────────────────────────────────────────

def _nwm_key(hour: int) -> str:
    d = EVENT_DATE.strftime("%Y%m%d")
    return (f"nwm.{d}/analysis_assim/"
            f"nwm.t{hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc")


def load_nwm_frame(
    fs: s3fs.S3FileSystem,
    hour: int,
    target_comids: np.ndarray,
) -> dict[int, float]:
    """Read streamflow (m³/s) for target COMIDs from one NWM analysis file."""
    key = _nwm_key(hour)
    try:
        with fs.open(f"{NWM_BUCKET}/{key}", "rb") as fobj:
            with h5py.File(fobj, "r") as h:
                all_ids = h["feature_id"][:].astype(np.int64)
                q_arr   = h["streamflow"][:].astype(np.float32)
                # scale factor / fill — NWM stores raw float32, no extra scale needed
                q_arr   = np.where(q_arr < 0, np.nan, q_arr)
                idx = np.where(np.isin(all_ids, target_comids))[0]
                return {int(all_ids[i]): float(q_arr[i]) for i in idx}
    except Exception as e:
        print(f"  [NWM h{hour:02d}] error: {e}")
        return {}


def load_all_nwm(
    fs: s3fs.S3FileSystem,
    target_comids: np.ndarray,
) -> list[dict[int, float]]:
    """Load one streamflow dict per hour; missing hours return {}."""
    print(f"\n── Loading NWM analysis streamflow for {EVENT_DATE} ─────────────────────")
    frames: list[dict[int, float]] = []
    for h in HOURS:
        print(f"  Loading hour {h:02d} UTC...", end="\r")
        frames.append(load_nwm_frame(fs, h, target_comids))
    n_ok = sum(len(f) > 0 for f in frames)
    print(f"  Loaded {n_ok}/24 NWM frames with data.        ")
    return frames


# ── Step 4: Build animation ────────────────────────────────────────────────────

def make_animation(
    frames_1h: list,
    frames_3h: list,
    lats: Optional[np.ndarray],
    lons: Optional[np.ndarray],
    flowlines: gpd.GeoDataFrame,
    nwm_frames: list[dict[int, float]],
    all_segs: list,
    seg_comids: np.ndarray,
) -> None:
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # Precipitation colormap: white → yellow → orange → red → purple
    precip_colors = [
        (1.0, 1.0, 1.0, 0.0),   # transparent for zero
        "#b3d9ff", "#6ab4ff", "#1f78b4",
        "#33a02c", "#b2df8a",
        "#ffff33", "#ff7f00",
        "#e31a1c", "#fb9a99",
        "#6a0dad",
    ]
    precip_cmap = mcolors.LinearSegmentedColormap.from_list(
        "nws_precip", precip_colors, N=256
    )
    precip_cmap.set_under("none")

    norm_1h  = mcolors.Normalize(vmin=0.01, vmax=QPE_1H_VMAX)
    norm_3h  = mcolors.Normalize(vmin=0.01, vmax=QPE_3H_VMAX)
    norm_nwm = mcolors.LogNorm(vmin=NWM_Q_VMIN, vmax=NWM_Q_VMAX)
    nwm_cmap = plt.get_cmap("Blues")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
        gridspec_kw={"wspace": 0.08},
    )
    titles = [
        "MRMS QPE — 1-h (in)",
        "MRMS QPE — 3-h rolling (in)",
        "NWM streamflow (m³/s)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")

    # Basemap tiles (added once; won't change between frames)
    if HAS_CTX:
        print("\nAdding basemap tiles (requires internet)...")
        for ax in axes:
            try:
                ctx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=12,
                    attribution=False,
                    zorder=0,
                )
            except Exception as e:
                print(f"  Basemap warning: {e}")

    # ── Initialise artists ─────────────────────────────────────────────────────
    blank = np.zeros((2, 2))

    # MRMS panels: imshow (origin='upper' because lats are descending)
    im_1h = axes[0].imshow(
        blank, origin="upper", extent=extent,
        cmap=precip_cmap, norm=norm_1h,
        alpha=ALPHA_PRECIP, zorder=2, interpolation="nearest",
    )
    im_3h = axes[1].imshow(
        blank, origin="upper", extent=extent,
        cmap=precip_cmap, norm=norm_3h,
        alpha=ALPHA_PRECIP, zorder=2, interpolation="nearest",
    )

    # Colourbars for precipitation
    fig.colorbar(im_1h, ax=axes[0], fraction=0.035, pad=0.02,
                 label="Rainfall (in)")
    fig.colorbar(im_3h, ax=axes[1], fraction=0.035, pad=0.02,
                 label="Rainfall (in)")

    # NWM: LineCollection (segments pre-computed)
    lc = LineCollection(
        all_segs if all_segs else [[[LON_MIN, LAT_MIN], [LON_MAX, LAT_MAX]]],
        linewidths=2.0, zorder=3,
    )
    axes[2].add_collection(lc)
    sm_nwm = plt.cm.ScalarMappable(cmap=nwm_cmap, norm=norm_nwm)
    sm_nwm.set_array([])
    fig.colorbar(sm_nwm, ax=axes[2], fraction=0.035, pad=0.02,
                 label="Streamflow (m³/s)")

    # Timestamp text (centred above all panels)
    time_text = fig.text(
        0.5, 0.97,
        "",
        ha="center", va="top", fontsize=13, fontweight="bold",
    )

    # ── Update function ────────────────────────────────────────────────────────
    def update(h: int):
        # -- MRMS 1h --
        f1 = frames_1h[h]
        if f1 is not None:
            im_1h.set_data(f1)
            im_1h.set_extent(extent)
        else:
            im_1h.set_data(np.zeros((2, 2)))

        # -- MRMS 3h rolling --
        f3 = frames_3h[h]
        if f3 is not None:
            im_3h.set_data(f3)
            im_3h.set_extent(extent)
        else:
            im_3h.set_data(np.zeros((2, 2)))

        # -- NWM streamflow --
        q_map = nwm_frames[h]
        if q_map and all_segs:
            q_vals = np.array(
                [q_map.get(int(c), NWM_Q_VMIN) if c >= 0 else NWM_Q_VMIN
                 for c in seg_comids],
                dtype=float,
            )
            q_vals = np.clip(q_vals, NWM_Q_VMIN, NWM_Q_VMAX)
            colors = nwm_cmap(norm_nwm(q_vals))
            lc.set_colors(colors)
        else:
            lc.set_colors(["#aaaaaa"] * max(len(all_segs), 1))

        time_text.set_text(
            f"{EVENT_DATE:%Y-%m-%d}  {h:02d}:00 UTC"
        )
        return [im_1h, im_3h, lc, time_text]

    # ── Render ─────────────────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update,
        frames=HOURS,
        interval=FRAME_MS,
        blit=False,
    )

    print(f"\nRendering {len(HOURS)} frames → {OUT_GIF}")
    writer = PillowWriter(fps=int(1000 / FRAME_MS))
    anim.save(OUT_GIF, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {OUT_GIF}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fs_anon = _anon_fs()

    # 1. MRMS
    frames_1h, lats, lons = load_all_mrms(fs_anon)
    frames_3h = compute_3h_rolling(frames_1h)

    # 2. NHD flowlines
    try:
        flowlines = fetch_nhd_flowlines()
    except Exception as e:
        print(f"  WARNING: NHD query failed ({e}) — NWM panel will show grey lines.")
        flowlines = gpd.GeoDataFrame({"comid": [], "geometry": []},
                                      crs="EPSG:4326")

    all_segs, seg_comids, _ = build_line_collection_data(flowlines)
    target_comids = seg_comids[seg_comids >= 0]

    # 3. NWM
    if len(target_comids) > 0:
        nwm_frames = load_all_nwm(fs_anon, target_comids)
    else:
        print("  No COMIDs found — skipping NWM download.")
        nwm_frames = [{} for _ in HOURS]

    # 4. Animate
    make_animation(
        frames_1h, frames_3h, lats, lons,
        flowlines, nwm_frames,
        all_segs, seg_comids,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
