"""visualize_lanesville_bridge_vs_station.py

Poster companion to visualize_lanesville_mrms_panels.py (Lanesville, IN flash
flood, 9 June 2026). Two stacked hourly precipitation panels for the local day
of June 9 (00-24 EDT), on the SAME y-axis so they compare directly:

  top     MRMS 1-h QPE (Pass2) at the pixel nearest the Focus Bridge 062-31-07183
          (red star, as on the maps)
  bottom  USGS 03294500 precip station (yellow dot, as on the maps): IV param
          00045, 15-min increments summed into hours ending on the hour, so the
          bins match the MRMS 1-h accumulations

Each panel follows visualize_bridge_event.py: the peak 1-h depth, its Atlas-14
1-h ARI by log-log interpolation of the RP -> depth curve, and a dashed line at
the Atlas-14 depth of the standard return period nearest that ARI (log space).

ATLAS-14 SOURCE. The 1-h depths come from NOAA's labelled PFDS csv endpoint (PDS
series, the same series the pipeline uses), NOT atlas14/precipitation_frequency.parquet:
that table still carries the duration off-by-one (see 07d_relabel_atlas14_durations.py;
it still has the duration_hr == 120 fingerprint), so its "1 h" row holds the 30-min
depth. visualize_bridge_event.py reads that table, which is why its figure for this
bridge shows "P1000 1-h = 2.79 in"; 2.79 in is the 30-min P1000.

Layout: <= 5.90 x 8.30 in, Times New Roman, transparent background, 600 dpi
(300 dpi when printed at 2x). Tick labels 20 pt (the smallest text), other
text 22 pt.

Runs locally: MRMS over public HTTPS (decoder shared with
visualize_lanesville_mrms_panels.py), gauge precip from the pipeline bucket
(AWS credentials), Atlas-14 from the PFDS web service.

Writes:
    results/lanesville_20260609_bridge_vs_station.png

Usage:
    python scripts/visualize_lanesville_bridge_vs_station.py
"""
from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from visualize_lanesville_mrms_panels import (  # noqa: E402
    BRIDGE_COLOR, HIGHLIGHT_COLOR, PRECIP_DOT_COLOR, decode_mrms_grib2)

# ── Configuration ──────────────────────────────────────────────────────────────
EVENT_DATE = date(2026, 6, 9)                # local (EDT) day shown
UTC_OFFSET_HR, TZ_LABEL = -4, "EDT"

BRIDGE_ASSET = "062-31-07183"
BRIDGE_LAT, BRIDGE_LON = 38.2218, -86.01091  # bridge_coverage_flags.parquet
GAUGE_SITE = "03294500"
GAUGE_LAT, GAUGE_LON = 38.280347, -85.799131

MRMS_HTTP = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"
PIPELINE = "indot-bridge-pipeline/v1/"
USGS_PRECIP_KEY = "precip/usgs/precip_iv.parquet"
USGS_NODATA = -999999.0

PFDS_CSV = "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
PRECIP_RPS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

BAR_COLOR = "#1f78b4"
LINE_COLOR = "#e31a1c"

FIG_W, FIG_H = 5.90, 8.30
TICK_FS = 20                                 # smallest text
TEXT_FS = 22                                 # axis labels, annotations
TITLE_FS = 28                                # bold panel headings (= the maps' date heading)
DPI = 600
OUT_PNG = os.path.join("results", f"lanesville_{EVENT_DATE:%Y%m%d}_bridge_vs_station.png")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": TEXT_FS,
    "axes.titlesize": TEXT_FS,
    "axes.labelsize": TEXT_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})


# ── Data ───────────────────────────────────────────────────────────────────────

def hour_ends_utc() -> pd.DatetimeIndex:
    """UTC end times of the 24 one-hour bins making up the local day."""
    start_local = pd.Timestamp(EVENT_DATE)
    start_utc = (start_local - pd.Timedelta(hours=UTC_OFFSET_HR)).tz_localize("UTC")
    return pd.date_range(start_utc + pd.Timedelta(hours=1), periods=24, freq="1h")


def mrms_at_point(ends: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    out = np.full(len(ends), np.nan)
    for k, t in enumerate(ends):
        url = (f"{MRMS_HTTP}/CONUS/{MRMS_PRODUCT}/{t:%Y%m%d}/"
               f"MRMS_{MRMS_PRODUCT}_{t:%Y%m%d-%H}0000.grib2.gz")
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f"  MRMS {t:%Y-%m-%d %H}Z missing ({r.status_code})")
            continue
        vals, lats, lons = decode_mrms_grib2(gzip.decompress(r.content))
        out[k] = vals[np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))] / 25.4
        print(f"  MRMS {t:%m-%d %H}Z: {out[k]:.2f} in", end="\r")
    print()
    return out


def gauge_hourly(ends: pd.DatetimeIndex) -> np.ndarray:
    lo, hi = ends[0] - pd.Timedelta(hours=1), ends[-1]
    df = pq.read_table(PIPELINE + USGS_PRECIP_KEY,
                       filesystem=pafs.S3FileSystem(region="us-east-1"),
                       columns=["site_no", "datetime_utc", "precip_in"],
                       filters=[("site_no", "=", GAUGE_SITE),
                                ("datetime_utc", ">", lo.to_pydatetime()),
                                ("datetime_utc", "<=", hi.to_pydatetime())]).to_pandas()
    if df.empty:
        raise SystemExit(f"no USGS precip for {GAUGE_SITE} in {lo}..{hi}")
    s = df.set_index(pd.to_datetime(df["datetime_utc"], utc=True))["precip_in"].astype(float)
    s = s[(s > USGS_NODATA + 1) & (s >= 0)].sort_index()
    hourly = s.resample("1h", label="right", closed="right").sum()
    print(f"  gauge {GAUGE_SITE}: {len(s)} IV obs; day total {hourly.sum():.2f} in")
    return hourly.reindex(ends).fillna(0.0).to_numpy(float)


def atlas14_1h(lat: float, lon: float) -> dict[int, float]:
    """Atlas-14 60-min PDS depths (in) by return period, from the labelled csv."""
    r = requests.get(PFDS_CSV, params=dict(lat=f"{lat:.4f}", lon=f"{lon:.4f}", type="pf",
                                           data="depth", units="english", series="pds"),
                     timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    head = next(l for l in lines if l.startswith("by duration for ARI"))
    rps = [int(x) for x in head.split(":,")[1].split(",")]
    row = next(l for l in lines if l.startswith("60-min:"))       # first table = estimates
    depths = [float(x) for x in row.split(":,")[1].split(",")]
    out = dict(zip(rps, depths))
    assert sorted(out) == PRECIP_RPS, out
    return out


# ── ARI helpers (as in visualize_bridge_event.py) ──────────────────────────────

def ari_from_value(value: float, a14: dict[int, float]) -> tuple[float, str]:
    rps = np.array(sorted(a14)); dv = np.array([a14[r] for r in rps])
    ari = 10.0 ** float(np.interp(np.log10(value), np.log10(dv), np.log10(rps)))
    cmp = "≥" if value >= dv[-1] else ("<" if value < dv[0] else "≈")
    return ari, cmp


def nearest_standard_rp(ari: float) -> int:
    la = np.log10(max(ari, 1.0001))
    return min(PRECIP_RPS, key=lambda r: abs(np.log10(r) - la))


# ── Figure ─────────────────────────────────────────────────────────────────────

def title_with_marker(ax, text, marker, ms, color):
    """Bold axes title followed by the map symbol of the point it describes.

    The title starts at the figure's left margin (not the axes edge) so the 28 pt
    heading fits the 5.9 in width."""
    t = ax.set_title(text, loc="left", pad=8, fontsize=TITLE_FS, fontweight="bold")
    fig = ax.figure
    x_ax = ax.transAxes.inverted().transform(fig.transFigure.transform((0.015, 0)))[0]
    t.set_x(x_ax)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transAxes.inverted())
    ax.plot([bb.x1 + 0.065],[(bb.y0 + bb.y1) / 2], marker=marker, ms=ms, mfc=color,
            mec="black", mew=1.0, ls="", transform=ax.transAxes, clip_on=False)


def draw_panel(ax, x, vals, a14, ymax):
    ax.bar(x, vals, width=0.85, color=BAR_COLOR, alpha=0.8, zorder=2)
    peak = float(np.nanmax(vals))
    k = int(np.nanargmax(vals))
    ari, cmp = ari_from_value(peak, a14)
    rp = nearest_standard_rp(ari)
    ax.axhline(a14[rp], color=LINE_COLOR, ls="--", lw=2.0, zorder=3)
    ax.text(0.3, a14[rp] + 0.08, f"Atlas-14 P{rp} = {a14[rp]:.2f} in",
            ha="left", va="bottom", fontsize=TICK_FS, color=LINE_COLOR, zorder=5)
    ax.plot([x[k]], [peak], "v", color=LINE_COLOR, ms=12, zorder=6)
    ari_txt = f"< {PRECIP_RPS[0]}" if cmp == "<" else f"{cmp} {ari:.0f}"
    ax.text(0.97, 0.95, f"Peak 1-h = {peak:.2f} in",
            transform=ax.transAxes, ha="right", va="top", fontsize=TEXT_FS,
            color=LINE_COLOR, linespacing=1.15, zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LINE_COLOR, alpha=0.9))
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 24)
    ax.set_ylabel("1-h precip (in)")
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    print(f"  peak {peak:.2f} in at bin {k:02d}-{k + 1:02d} {TZ_LABEL}; ARI {ari_txt} yr; "
          f"line P{rp} = {a14[rp]:.2f} in")


def make_figure(mrms, gauge, a14_bridge, a14_gauge):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.925, bottom=0.10, hspace=0.30)
    x = np.arange(24) + 0.5                       # bar centred in its local hour
    ymax = 1.6 * max(np.nanmax(mrms), np.nanmax(gauge), 1.0)   # headroom for the peak box

    print("Focus Bridge (MRMS):"); draw_panel(ax1, x, mrms, a14_bridge, ymax)
    print(f"Precip Station {GAUGE_SITE}:"); draw_panel(ax2, x, gauge, a14_gauge, ymax)

    ax2.set_xticks(range(0, 25, 3))
    ax2.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)])
    ax2.set_xlabel(f"Hour ({TZ_LABEL}), "
                   f"{EVENT_DATE:%B} {EVENT_DATE.day}, {EVENT_DATE:%Y}")

    title_with_marker(ax1, "MRMS at Focus Bridge", "*", 22, HIGHLIGHT_COLOR)
    title_with_marker(ax2, f"Precip Station USGS {GAUGE_SITE}", "o", 15, PRECIP_DOT_COLOR)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    # Transparent canvas, white plotting area.
    fig.patch.set_alpha(0.0)
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="none")
    plt.close(fig)
    w, h = Image.open(OUT_PNG).size
    print(f"Saved {OUT_PNG}: {w} x {h} px ({w / DPI:.2f} x {h / DPI:.2f} in at {DPI} dpi)")


def main():
    ends = hour_ends_utc()
    print(f"Local day {EVENT_DATE} {TZ_LABEL} = hour bins ending "
          f"{ends[0]:%m-%d %H}Z .. {ends[-1]:%m-%d %H}Z")
    a14_bridge = atlas14_1h(BRIDGE_LAT, BRIDGE_LON)
    a14_gauge = atlas14_1h(GAUGE_LAT, GAUGE_LON)
    print(f"  Atlas-14 1-h bridge: {a14_bridge}\n  Atlas-14 1-h gauge : {a14_gauge}")
    gauge = gauge_hourly(ends)
    mrms = mrms_at_point(ends, BRIDGE_LAT, BRIDGE_LON)
    make_figure(mrms, gauge, a14_bridge, a14_gauge)


if __name__ == "__main__":
    main()
