"""09c_pipeline_flowchart.py

Reads station counts live from S3 and draws a flowchart showing how
stations move through the pipeline from Script 01 to Script 10,
including active/inactive breakdowns and exclusion reasons.

Writes:
    s3://<bucket>/<prefix>analysis/figures/pipeline_flowchart.png
"""
from __future__ import annotations

import io
import logging

import boto3
import botocore.exceptions
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from utils import load_config, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("09c_flowchart")

OUT_KEY = "analysis/figures/pipeline_flowchart.png"

# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def read_sites(bucket: str, key: str, col: str = "site_no") -> set:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        return set(
            pq.read_table(io.BytesIO(obj["Body"].read()), columns=[col])
            .to_pandas()[col].astype(str).str.strip().unique()
        )
    except botocore.exceptions.ClientError:
        log.warning("Not found: %s", key)
        return set()


def count_unique(bucket: str, key: str, col: str = "site_no") -> int | None:
    s = read_sites(bucket, key, col)
    return len(s) if s else None


def list_geojson_sites(bucket: str, prefix: str) -> set:
    pag = _s3().get_paginator("list_objects_v2")
    sites: set = set()
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".geojson"):
                sites.add(k.split("/")[-1].replace(".geojson", ""))
    return sites


def fmt(n: int | None) -> str:
    return f"{n:,}" if n is not None else "—"


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, title, lines, color):
    """Rounded rectangle with title and variable-height line list."""
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#444444", linewidth=1.6, zorder=2,
    )
    ax.add_patch(rect)
    n_lines = len(lines)
    # Distribute lines evenly inside the box
    step = h / (n_lines + 2)
    for i, (txt, kw) in enumerate([(title, {"size": 11, "weight": "bold",
                                             "color": "#1a1a2e"})] + lines):
        y = cy + h / 2 - step * (i + 1)
        ax.text(cx, y, txt, ha="center", va="center", zorder=3,
                fontsize=kw.get("size", 9.5),
                fontweight=kw.get("weight", "normal"),
                color=kw.get("color", "#2c2c2c"),
                style=kw.get("style", "normal"))


def draw_arrow(ax, x0, y0, x1, y1, style="arc3,rad=0"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.5, mutation_scale=15,
                                connectionstyle=style),
                zorder=1)


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(c: dict) -> plt.Figure:
    FW, FH = 26, 20
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    # colour palette
    C_INV  = "#D6EAF8"   # blue   – inventory / streamflow
    C_SPA  = "#D5F5E3"   # green  – spatial / watershed
    C_MRMS = "#FDEBD0"   # orange – precipitation
    C_ANAL = "#E8DAEF"   # purple – analysis
    C_NWM  = "#D5F5E3"   # green  – NWM (same family as spatial)
    C_PREC = "#FCF3CF"   # yellow – supplementary precip

    MX  = 8.5     # main column centre-x
    NX  = 20.0    # NWM column centre-x
    BW  = 13.0    # main box width
    NW  = 9.5     # NWM/precip box width

    # Y positions (top → bottom)
    Y01   = 18.5
    Y02   = 15.3
    Y03   = 12.1
    Y04   = 9.0
    Y0506 = 5.9
    Y07   = 3.0
    Y08   = 0.4
    Y10   = 13.5
    Y1213 = 4.5

    BH = 2.3   # standard box height

    ax.text(FW / 2, 19.6,
            "INDOT Bridge Pipeline — Station Counts and Data Flow",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1a1a2e")

    # ── Script 01 ─────────────────────────────────────────────────────────────
    draw_box(ax, MX, Y01, BW, BH, "Script 01 — Station Inventory",
             [
                 (f"All gauges: {fmt(c['01_total'])}   |   "
                  f"Active (end_date ≥ 2020-10-14): {fmt(c['01_active'])}   |   "
                  f"Inactive: {fmt(c['01_total'] - c['01_active'] if c['01_total'] and c['01_active'] else None)}",
                  {"size": 11, "weight": "bold"}),
                 ("USGS NWIS API · Indiana discharge gauges (parameter 00060)",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_INV)

    # ── Script 02 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y01 - BH / 2, MX, Y02 + BH / 2)
    draw_box(ax, MX, Y02, BW, BH, "Script 02 — Streamflow Download",
             [
                 (f"Stations with discharge records: {fmt(c['02'])}",
                  {"size": 11, "weight": "bold"}),
                 ("USGS Water Data v3 · 15-min instantaneous · "
                  "clipped to each gauge's begin / end dates",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_INV)

    # ── Script 03 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y02 - BH / 2, MX, Y03 + BH / 2)
    draw_box(ax, MX, Y03, BW, BH + 0.3, "Script 03 — Watershed Delineation",
             [
                 (f"Delineated watersheds: {fmt(c['03'])}   "
                  f"({fmt(c['03_active'])} from active stations  |  "
                  f"{fmt(c['03_inactive'])} from inactive stations)",
                  {"size": 11, "weight": "bold"}),
                 (f"{fmt(c['active_no_ws'])} active gauges could not be delineated "
                  "(NLDI lookup failed — ungauged or non-stream sites)",
                  {"size": 9, "color": "#c0392b"}),
                 ("USGS NLDI · per-gauge upstream basin GeoJSON polygons",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_SPA)

    # ── Script 04 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y03 - (BH + 0.3) / 2, MX, Y04 + BH / 2)
    draw_box(ax, MX, Y04, BW, BH, "Script 04 — Flow Statistics",
             [
                 (f"Stations with Q10 / Q50: {fmt(c['04'])}   "
                  f"(includes {fmt(c['fs_no_ws'])} stations without watershed polygon)",
                  {"size": 11, "weight": "bold"}),
                 ("USGS Gage Statistics Service · published flood-frequency curves",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_SPA)

    # ── Scripts 05+06 ─────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y04 - BH / 2, MX, Y0506 + (BH + 0.3) / 2)
    draw_box(ax, MX, Y0506, BW, BH + 0.3,
             "Scripts 05 + 06 — MRMS Precipitation Extraction",
             [
                 (f"Nearest pixel: {fmt(c['05'])} stations   |   "
                  f"Watershed mean: {fmt(c['06'])} stations",
                  {"size": 11, "weight": "bold"}),
                 (f"Watershed mean = exactly {fmt(c['06'])} active stations with a valid watershed polygon",
                  {"size": 9}),
                 ("NOAA MRMS QPE_01H_Pass2 · hourly · 2020-10-14 → present · active stations only",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_MRMS)

    # ── Script 07 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y0506 - (BH + 0.3) / 2, MX, Y07 + BH / 2)
    draw_box(ax, MX, Y07, BW, BH, "Script 07 — Atlas-14 Precipitation Frequency",
             [
                 (f"Stations with PF data: {fmt(c['07'])}  (= all active stations)",
                  {"size": 11, "weight": "bold"}),
                 ("NOAA PFDS API · durations 1 h – 60 d · return periods 2 – 1000 yr · active stations only",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_MRMS)

    # ── Script 08 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, MX, Y07 - BH / 2, MX, Y08 + (BH + 0.6) / 2)
    draw_box(ax, MX, Y08, BW, BH + 0.6, "Script 08 — Trigger Analysis",
             [
                 (f"Stations in analysis: {fmt(c['08'])}   "
                  f"(from {fmt(c['06'])} watershed-mean stations)",
                  {"size": 12, "weight": "bold", "color": "#1a1a2e"}),
                 (f"{fmt(c['qc_excluded'])} stations excluded by QC: "
                  "regulated / artificial channels "
                  "(negative recorded flow  OR  published Q10 < observed median flow)",
                  {"size": 9, "color": "#c0392b"}),
                 ("MRMS precipitation × USGS streamflow × Atlas-14 × Flow Stats intersection",
                  {"style": "italic", "color": "#555", "size": 9}),
             ], C_ANAL)

    # ── Script 10 — NWM (right branch) ────────────────────────────────────────
    NH = 5.5
    draw_box(ax, NX, Y10, NW, NH, "Script 10 — NWM Data Download",
             [
                 ("Retrospective v3.0  (1979 – 2023):",
                  {"weight": "bold", "size": 10}),
                 (f"  {fmt(c['10_retro'])} stations  "
                  f"({fmt(c['retro_active'])} active  |  {fmt(c['retro_inactive'])} inactive)",
                  {"size": 10}),
                 ("Analysis & Assimilation  (2018 – present):",
                  {"weight": "bold", "size": 10}),
                 (f"  {fmt(c['10_assim'])} stations  "
                  f"({fmt(c['assim_active'])} active  |  {fmt(c['assim_inactive'])} inactive)",
                  {"size": 10}),
                 ("Open Loop  (2021 – present):",
                  {"weight": "bold", "size": 10}),
                 (f"  {fmt(c['10_loop'])} stations  "
                  f"({fmt(c['loop_active'])} active  |  {fmt(c['loop_inactive'])} inactive)",
                  {"size": 10}),
                 ("Uses ALL stations (active + inactive);  "
                  "excl. if COMID not found in NWM domain",
                  {"style": "italic", "color": "#555", "size": 8.5}),
             ], C_NWM)

    # Arrow: Script 01 → NWM
    draw_arrow(ax, MX + BW / 2, Y01,
               NX - NW / 2, Y10 + NH * 0.35,
               style="arc3,rad=-0.18")

    # ── Scripts 12+13 — Supplementary precip ──────────────────────────────────
    PH = 3.2
    draw_box(ax, NX, Y1213, NW, PH,
             "Scripts 12 + 13 — Supplementary Precipitation",
             [
                 (f"NOAA ISD / LCD stations: {fmt(c['12_isd'])}",
                  {"size": 10}),
                 (f"NOAA GHCNh stations: {fmt(c['12_ghcnh'])}",
                  {"size": 10}),
                 (f"USGS IV precip stations (param 00045): {fmt(c['13_usgs'])}",
                  {"size": 10}),
                 ("Spatial filter: full watershed union bbox "
                  "(IN + IL + OH + MI + KY + NY + PA + VA + WV + NC)",
                  {"style": "italic", "color": "#555", "size": 8.5}),
             ], C_PREC)

    # Arrow: Script 03 watersheds → supplementary precip (watershed union used for filtering)
    draw_arrow(ax, NX - NW / 2, Y10 - NH / 2,
               NX - NW / 2, Y1213 + PH / 2,
               style="arc3,rad=0")
    ax.text(NX - NW / 2 - 0.2,
            (Y10 - NH / 2 + Y1213 + PH / 2) / 2,
            "watershed\npolygons\nfor filter",
            ha="right", va="center", fontsize=8, color="#888", style="italic")

    fig.tight_layout(pad=0.5)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Reading station sets from S3...")

    all_sites    = read_sites(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    active_sites = read_sites(bucket, f"{prefix}stations/indiana_streamflow_sites_active.parquet")
    ws_sites     = list_geojson_sites(bucket, f"{prefix}watersheds/per_gauge/")
    fs_sites     = read_sites(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    wm_sites     = read_sites(bucket, f"{prefix}mrms/QPE_01H_Pass2/watershed_mean.parquet")
    ta_sites     = read_sites(bucket, f"{prefix}analysis/trigger_analysis.parquet")
    nwm_r        = read_sites(bucket, f"{prefix}nwm/retrospective.parquet")
    nwm_a        = read_sites(bucket, f"{prefix}nwm/analysis_assim.parquet")
    nwm_l        = read_sites(bucket, f"{prefix}nwm/open_loop.parquet")
    isd_sites    = read_sites(bucket, f"{prefix}precip/noaa/stations_isd.parquet",   col="station_id")
    ghcnh_sites  = read_sites(bucket, f"{prefix}precip/noaa/stations_ghcnh.parquet", col="station_id")
    usgs_p_sites = read_sites(bucket, f"{prefix}precip/usgs/stations.parquet")

    active_with_ws = active_sites & ws_sites

    c: dict = {
        "01_total":       len(all_sites)    or None,
        "01_active":      len(active_sites) or None,
        "02":             count_unique(bucket, f"{prefix}streamflow/instantaneous/all_gauges_long.parquet"),
        "03":             len(ws_sites)     or None,
        "03_active":      len(active_with_ws),
        "03_inactive":    len(ws_sites - active_sites),
        "active_no_ws":   len(active_sites - ws_sites),
        "04":             len(fs_sites)     or None,
        "fs_no_ws":       len(fs_sites - ws_sites),
        "05":             count_unique(bucket, f"{prefix}mrms/QPE_01H_Pass2/nearest_pixel.parquet"),
        "06":             len(wm_sites)     or None,
        "07":             count_unique(bucket, f"{prefix}atlas14/precipitation_frequency.parquet"),
        "08":             len(ta_sites)     or None,
        "qc_excluded":    len(wm_sites - ta_sites),
        "10_retro":       len(nwm_r)        or None,
        "retro_active":   len(nwm_r & active_sites),
        "retro_inactive": len(nwm_r - active_sites),
        "10_assim":       len(nwm_a)        or None,
        "assim_active":   len(nwm_a & active_sites),
        "assim_inactive": len(nwm_a - active_sites),
        "10_loop":        len(nwm_l)        or None,
        "loop_active":    len(nwm_l & active_sites),
        "loop_inactive":  len(nwm_l - active_sites),
        "12_isd":         len(isd_sites)    or None,
        "12_ghcnh":       len(ghcnh_sites)  or None,
        "13_usgs":        len(usgs_p_sites) or None,
    }

    for k, v in sorted(c.items()):
        log.info("  %-22s = %s", k, fmt(v))

    log.info("Building figure...")
    fig = build_figure(c)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    key = f"{prefix}{OUT_KEY}"
    write_bytes_to_s3(buf.getvalue(), bucket, key)
    log.info("Saved s3://%s/%s", bucket, key)


if __name__ == "__main__":
    main()
