"""09c_pipeline_flowchart.py

Reads station counts live from S3 and draws a flowchart showing how
stations move through the pipeline from Script 01 to Script 10.

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
import pandas as pd

from utils import load_config, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("09c_flowchart")

OUT_KEY = "analysis/figures/pipeline_flowchart.png"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def count_unique(bucket: str, key: str, col: str = "site_no") -> int | None:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        df = pq.read_table(io.BytesIO(obj["Body"].read()),
                           columns=[col]).to_pandas()
        return int(df[col].nunique())
    except botocore.exceptions.ClientError:
        return None


def count_rows(bucket: str, key: str) -> int | None:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        pf = pq.ParquetFile(io.BytesIO(obj["Body"].read()))
        return int(pf.metadata.num_rows)
    except botocore.exceptions.ClientError:
        return None


def count_geojsons(bucket: str, prefix: str) -> int:
    paginator = _s3().get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        n += sum(1 for obj in page.get("Contents", [])
                 if obj["Key"].endswith(".geojson"))
    return n


def fmt(n: int | None) -> str:
    return f"{n:,}" if n is not None else "—"


# ── Figure ────────────────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, title, lines, color, title_color="#1a1a2e"):
    """Rounded box with a bold title and additional text lines."""
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="#555555", linewidth=1.4, zorder=2,
    )
    ax.add_patch(rect)
    n = len(lines) + 1
    step = h / (n + 1)
    top_y = cy + h / 2 - step
    ax.text(cx, top_y, title,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=title_color, zorder=3)
    for i, (txt, kw) in enumerate(lines):
        ax.text(cx, top_y - step * (i + 1), txt,
                ha="center", va="center", zorder=3,
                fontsize=kw.get("size", 8.5), fontweight=kw.get("weight", "normal"),
                color=kw.get("color", "#2c2c2c"), style=kw.get("style", "normal"))


def arrow(ax, x0, y0, x1, y1, **kw):
    defaults = dict(arrowstyle="-|>", color="#555555", lw=1.4,
                    mutation_scale=14, zorder=1)
    defaults.update(kw)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=defaults.pop("arrowstyle"),
                                color=defaults.pop("color"),
                                lw=defaults.pop("lw"),
                                mutation_scale=defaults.pop("mutation_scale")),
                zorder=defaults.get("zorder", 1))


def build_figure(counts: dict) -> plt.Figure:
    W, H = 15, 20
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── Colour palette ────────────────────────────────────────────────────────
    C_INVENTORY  = "#D6EAF8"   # light blue  – station data
    C_SPATIAL    = "#D5F5E3"   # light green – spatial/watershed
    C_MRMS       = "#FDEBD0"   # light orange – MRMS/Atlas14
    C_ANALYSIS   = "#E8DAEF"   # light purple – final analysis
    C_NWM        = "#D5F5E3"   # light green – NWM (right branch)
    C_PRECIP     = "#FCF3CF"   # light yellow – supplementary precip

    # Main column x, NWM column x
    MX, NX, PX = 5.5, 12.0, 5.5

    # ── Y positions (top to bottom of main pipeline) ──────────────────────────
    Y = {
        "01":    18.5,
        "02":    15.5,
        "03":    12.5,
        "04":    9.5,
        "0506":  6.5,
        "07":    3.8,
        "08":    1.2,
        "10":    12.5,
        "1213":  3.8,
    }
    BW, BH = 6.2, 2.0   # box width, height (main)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(W / 2, 19.6, "INDOT Bridge Pipeline — Station Flow",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")

    # ── Script 01 ─────────────────────────────────────────────────────────────
    draw_box(ax, MX, Y["01"], BW, BH,
             "Script 01 — Station Inventory",
             [
                 (f"All gauges: {fmt(counts['01_total'])} sites", {"weight": "bold", "size": 9.5}),
                 (f"Active (end ≥ 2020-10-14): {fmt(counts['01_active'])} sites", {"size": 8.5}),
                 ("USGS NWIS API · Indiana discharge (00060)", {"style": "italic", "color": "#555"}),
             ], C_INVENTORY)

    # ── Script 02 ─────────────────────────────────────────────────────────────
    arrow(ax, MX, Y["01"] - BH / 2, MX, Y["02"] + BH / 2)
    draw_box(ax, MX, Y["02"], BW, BH,
             "Script 02 — Streamflow Download",
             [
                 (f"Stations with records: {fmt(counts['02'])}", {"weight": "bold", "size": 9.5}),
                 ("USGS Water Data v3 · 15-min instantaneous discharge", {"style": "italic", "color": "#555"}),
                 ("Clipped to each gauge's begin/end dates", {"style": "italic", "color": "#555"}),
             ], C_INVENTORY)

    # ── Script 03 ─────────────────────────────────────────────────────────────
    arrow(ax, MX, Y["02"] - BH / 2, MX, Y["03"] + BH / 2)
    draw_box(ax, MX, Y["03"], BW, BH,
             "Script 03 — Watershed Delineation",
             [
                 (f"Delineated watersheds: {fmt(counts['03'])}", {"weight": "bold", "size": 9.5}),
                 ("NLDI / USGS StreamStats · per-gauge GeoJSON polygons", {"style": "italic", "color": "#555"}),
                 ("Exclusions: NLDI 404 (non-stream / ungauged sites)", {"style": "italic", "color": "#555"}),
             ], C_SPATIAL)

    # ── Script 04 ─────────────────────────────────────────────────────────────
    arrow(ax, MX, Y["03"] - BH / 2, MX, Y["04"] + BH / 2)
    draw_box(ax, MX, Y["04"], BW, BH,
             "Script 04 — Flow Statistics",
             [
                 (f"Stations with Q10/Q50: {fmt(counts['04'])}", {"weight": "bold", "size": 9.5}),
                 ("USGS Gage Statistics Service (published flood-frequency)", {"style": "italic", "color": "#555"}),
                 ("Exclusions: sites with no published stats", {"style": "italic", "color": "#555"}),
             ], C_SPATIAL)

    # ── Scripts 05+06 ─────────────────────────────────────────────────────────
    arrow(ax, MX, Y["04"] - BH / 2, MX, Y["0506"] + BH / 2)
    draw_box(ax, MX, Y["0506"], BW, BH + 0.3,
             "Scripts 05+06 — MRMS Precipitation Extraction",
             [
                 (f"Nearest pixel: {fmt(counts['05'])} stations  |  Watershed mean: {fmt(counts['06'])} stations",
                  {"weight": "bold", "size": 9.5}),
                 ("NOAA MRMS QPE_01H_Pass2 · 2020-10-14 → present", {"style": "italic", "color": "#555"}),
                 ("Active stations only (end_date ≥ 2020-10-14)", {"style": "italic", "color": "#555"}),
             ], C_MRMS)

    # ── Script 07+08 ──────────────────────────────────────────────────────────
    arrow(ax, MX, Y["0506"] - (BH + 0.3) / 2, MX, Y["07"] + BH / 2)
    draw_box(ax, MX, Y["07"], BW, BH,
             "Script 07 — Atlas-14 Precipitation Frequency",
             [
                 (f"Stations with PF data: {fmt(counts['07'])}", {"weight": "bold", "size": 9.5}),
                 ("NOAA PFDS API · 1–60 h durations · 2–1000 yr return periods", {"style": "italic", "color": "#555"}),
             ], C_MRMS)

    arrow(ax, MX, Y["07"] - BH / 2, MX, Y["08"] + BH / 2)
    draw_box(ax, MX, Y["08"], BW, BH,
             "Script 08 — Trigger Analysis",
             [
                 (f"Stations in analysis: {fmt(counts['08'])}", {"weight": "bold", "size": 9.5, "color": "#1a1a2e"}),
                 ("MRMS × Streamflow × Atlas-14 × Flow Stats intersection", {"style": "italic", "color": "#555"}),
                 ("Exclusions: missing any input, QC failures (neg. flow, Q10 outliers)", {"style": "italic", "color": "#555"}),
             ], C_ANALYSIS)

    # ── Script 10 (NWM, right branch) ─────────────────────────────────────────
    NW, NH = 5.0, 3.8
    draw_box(ax, NX, Y["10"], NW, NH,
             "Script 10 — NWM Data Download",
             [
                 ("Retrospective (1979–2023):", {"weight": "bold", "size": 8.5}),
                 (f"   {fmt(counts['10_retro'])} stations", {"size": 9, "color": "#1a1a2e"}),
                 ("Analysis & Assimilation (2018–present):", {"weight": "bold", "size": 8.5}),
                 (f"   {fmt(counts['10_assim'])} stations", {"size": 9, "color": "#1a1a2e"}),
                 ("Open Loop (2021–present):", {"weight": "bold", "size": 8.5}),
                 (f"   {fmt(counts['10_loop'])} stations", {"size": 9, "color": "#1a1a2e"}),
                 ("NLDI COMID lookup; excl. if COMID not in NWM domain", {"style": "italic", "color": "#555", "size": 7.5}),
             ], C_NWM)

    # Arrow: Script 01 → NWM box (curved right)
    ax.annotate("",
                xy=(NX - NW / 2, Y["10"]),
                xytext=(MX + BW / 2, Y["01"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#555555", lw=1.4,
                    mutation_scale=14,
                    connectionstyle="arc3,rad=-0.25",
                ),
                zorder=1)

    # ── Scripts 12+13 (supplementary precip, bottom right) ───────────────────
    PW, PH = 5.0, 2.8
    draw_box(ax, NX, Y["1213"], PW, PH,
             "Scripts 12+13 — Supplementary Precipitation",
             [
                 (f"NOAA ISD stations: {fmt(counts['12_isd'])}", {"size": 8.5}),
                 (f"NOAA GHCNh stations: {fmt(counts['12_ghcnh'])}", {"size": 8.5}),
                 (f"USGS IV precip stations: {fmt(counts['13_usgs'])}", {"size": 8.5}),
                 ("Watershed-union spatial filter (all neighboring states)", {"style": "italic", "color": "#555", "size": 7.5}),
             ], C_PRECIP)

    # Arrow: Script 03 → Scripts 12+13 (watershed union used for filtering)
    ax.annotate("",
                xy=(NX - PW / 2, Y["1213"] + PH / 2),
                xytext=(NX - PW / 2, Y["10"] - NH / 2),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.1,
                                mutation_scale=12),
                zorder=1)
    ax.text(NX - PW / 2 - 0.15, (Y["1213"] + PH / 2 + Y["10"] - NH / 2) / 2,
            "watershed\npolygons", ha="right", va="center", fontsize=7,
            color="#888", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C_INVENTORY, edgecolor="#555", label="USGS Station / Streamflow data"),
        mpatches.Patch(facecolor=C_SPATIAL,   edgecolor="#555", label="Spatial / Watershed data"),
        mpatches.Patch(facecolor=C_MRMS,      edgecolor="#555", label="Precipitation data (MRMS, Atlas-14)"),
        mpatches.Patch(facecolor=C_NWM,       edgecolor="#555", label="NWM Streamflow model"),
        mpatches.Patch(facecolor=C_PRECIP,    edgecolor="#555", label="Supplementary precipitation"),
        mpatches.Patch(facecolor=C_ANALYSIS,  edgecolor="#555", label="Analysis output"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9, ncol=2, bbox_to_anchor=(0.01, 0.0))

    fig.tight_layout(pad=0.3)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Reading station counts from S3...")

    counts: dict = {}

    # Script 01
    counts["01_total"]  = count_unique(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    counts["01_active"] = count_unique(bucket, f"{prefix}stations/indiana_streamflow_sites_active.parquet")

    # Script 02
    counts["02"] = count_unique(bucket, f"{prefix}streamflow/instantaneous/all_gauges_long.parquet")

    # Script 03 — count GeoJSON files
    counts["03"] = count_geojsons(bucket, f"{prefix}watersheds/per_gauge/")

    # Script 04
    counts["04"] = count_unique(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")

    # Scripts 05+06
    counts["05"] = count_unique(bucket, f"{prefix}mrms/QPE_01H_Pass2/nearest_pixel.parquet")
    counts["06"] = count_unique(bucket, f"{prefix}mrms/QPE_01H_Pass2/watershed_mean.parquet")

    # Script 07
    counts["07"] = count_unique(bucket, f"{prefix}atlas14/precipitation_frequency.parquet")

    # Script 08
    counts["08"] = count_unique(bucket, f"{prefix}analysis/trigger_analysis.parquet")

    # Script 10
    counts["10_retro"] = count_unique(bucket, f"{prefix}nwm/retrospective.parquet")
    counts["10_assim"] = count_unique(bucket, f"{prefix}nwm/analysis_assim.parquet")
    counts["10_loop"]  = count_unique(bucket, f"{prefix}nwm/open_loop.parquet")

    # Scripts 12+13
    counts["12_isd"]   = count_unique(bucket, f"{prefix}precip/noaa/stations_isd.parquet",   col="station_id")
    counts["12_ghcnh"] = count_unique(bucket, f"{prefix}precip/noaa/stations_ghcnh.parquet", col="station_id")
    counts["13_usgs"]  = count_unique(bucket, f"{prefix}precip/usgs/stations.parquet")

    for k, v in sorted(counts.items()):
        log.info("  %-20s = %s", k, fmt(v))

    log.info("Building flowchart figure...")
    fig = build_figure(counts)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    key = f"{prefix}{OUT_KEY}"
    write_bytes_to_s3(buf.getvalue(), bucket, key)
    log.info("Saved s3://%s/%s", bucket, key)


if __name__ == "__main__":
    main()
