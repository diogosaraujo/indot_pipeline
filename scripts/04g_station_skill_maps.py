"""Per-station NWM-vs-gauge skill, mapped — is poor performance regional?

Two maps, one metric each, from 04d's annual_maxima_station_stats:

    station_skill_spearman.png   Spearman rho of the paired annual maxima
    station_skill_r2.png         R2 of log10(NWM) against log10(USGS)

Spearman and R2 rather than r and R2: R2 is just Pearson r squared, so mapping
both would be the same picture under a stretched colour scale. Rank correlation
and explained variance in log space are genuinely different views — a station
can order its floods correctly while getting their sizes wrong.

Reading the maps
────────────────
Colour is DIVERGING about a stated threshold (rho 0.70, R2 0.50, each close to
the fleet median), so the question "which stations are worse than typical, and
are they next to each other" is answered by the red cluster rather than by
comparing shades. Red is below the threshold, blue above, grey at it.

Marker SIZE is the number of paired water years. This matters more than it
looks: 44 of the 244 stations have fewer than 10 paired years, and a rank
correlation on 6 years is close to meaningless. Small dots should not be read
as evidence of anything. --min-years drops them outright.

Both metrics come from annual maxima paired by water year, so they measure
whether NWM reproduces the size and ordering of each year's biggest flood — not
whether it reproduces the hydrograph.

Usage
    python scripts/04g_station_skill_maps.py
    python scripts/04g_station_skill_maps.py --min-years 10
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("skillmap")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "monitor"))

STATS_KEY = "analysis/nwm_vs_usgs/annual_maxima_station_stats.parquet"
COMID_KEY = "nwm/comid_locations.parquet"

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, LAND = "#e1e0d9", "#f4f3ef"
# documented diverging pair: blue <-> red through a neutral grey
DIV = LinearSegmentedColormap.from_list("skill", ["#d03b3b", "#f0efec", "#2a78d6"])

METRICS = {
    "spearman": dict(col="spearman_r", centre=0.70, lo=0.0, hi=1.0,
                     label="Spearman ρ of paired annual maxima",
                     title="Does NWM rank each year's flood like the gauge?",
                     fname="station_skill_spearman.png"),
    "r2": dict(col="r2_log", centre=0.50, lo=0.0, hi=1.0,
               label="R² of log₁₀(NWM) vs log₁₀(USGS)",
               title="How much of the log-space variance does NWM explain?",
               fname="station_skill_r2.png"),
}


def size_for(n: np.ndarray) -> np.ndarray:
    return 18 + 120 * np.clip(n, 0, 45) / 45.0


def neighbour_corr(lat, lon, v, k: int = 5) -> float:
    """Correlation between a station's value and the mean of its k nearest.

    A plain spatial-autocorrelation check, so "are the poor ones clustered" is
    answered by a number and not only by how the dots look. Near 0 means skill
    is scattered; strongly positive means neighbours resemble each other.
    """
    ok = np.isfinite(v)
    if ok.sum() < k + 2:
        return float("nan")
    xy = np.c_[lon[ok] * np.cos(np.radians(39.9)), lat[ok]] * 111.0   # ~km
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1)[:, :k]
    return float(np.corrcoef(v[ok], v[ok][nb].mean(axis=1))[0, 1])


def draw(metric: str, d: pd.DataFrame, counties, out_dir: pathlib.Path,
         figsize, dpi: int) -> None:
    from monitor_common import maps
    m = METRICS[metric]
    v = d[m["col"]].to_numpy(float)
    ok = np.isfinite(v)

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    maps.draw_counties(ax, counties, lw=0.45, fc=LAND)

    norm = TwoSlopeNorm(vmin=m["lo"], vcenter=m["centre"], vmax=m["hi"])
    sc = ax.scatter(d.lon[ok], d.lat[ok], c=v[ok], cmap=DIV, norm=norm,
                    s=size_for(d.n_paired_wy.to_numpy(float))[ok],
                    ec="white", lw=0.7, zorder=8)

    ax.set_aspect(1.0 / np.cos(np.radians(39.9)))
    ax.set_xlim(-88.2, -84.6); ax.set_ylim(37.7, 41.9)
    ax.axis("off")
    ax.set_title(m["title"], fontsize=13.5, color=INK, loc="left", pad=18)
    ax.annotate(m["label"], (0.0, 1.014), xycoords="axes fraction",
                fontsize=9.5, color=INK_2, va="bottom")

    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02,
                      ticks=[m["lo"], m["centre"], m["hi"]])
    cb.ax.set_yticklabels([f"{m['lo']:.2f}", f"{m['centre']:.2f}\n(threshold)",
                           f"{m['hi']:.2f}"])
    cb.ax.tick_params(colors=MUTED, labelsize=8.5); cb.outline.set_visible(False)

    below = int((v[ok] < m["centre"]).sum())
    nc = neighbour_corr(d.lat.to_numpy(float), d.lon.to_numpy(float), v)
    verdict = ("clustered" if nc > 0.20 else
               "weakly clustered" if nc > 0.10 else "scattered")
    ax.annotate(
        f"{int(ok.sum())} stations   median {np.nanmedian(v[ok]):.2f}\n"
        f"{below} below {m['centre']:.2f}  ({100*below/ok.sum():.0f}%)\n"
        f"neighbour corr {nc:+.2f} — {verdict}",
        (0.015, 0.985), xycoords="axes fraction", va="top", fontsize=9.5,
        color=INK, bbox=dict(fc="white", ec=GRID, lw=0.7,
                             boxstyle="round,pad=0.45"), zorder=10)

    handles = [Line2D([], [], marker="o", ls="", mfc="#b9b7b0", mec="white",
                      mew=0.7, ms=np.sqrt(size_for(np.array([k]))[0]),
                      label=f"{k} yr") for k in (5, 20, 43)]
    # Figure-level, not axes-level: inside the axes this sat on the state outline.
    fig.legend(handles=handles, title="paired water years", frameon=False,
               fontsize=8.5, title_fontsize=8.5, labelcolor=INK_2,
               loc="lower left", bbox_to_anchor=(0.015, 0.035),
               borderpad=0.5, labelspacing=0.85)

    fig.text(0.985, 0.022,
             "Marker size is the paired record length — a correlation on 6 years "
             "says little.\nColour diverges about the threshold, not about the "
             "data range.",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    out = out_dir / m["fname"]
    fig.subplots_adjust(left=0.02, right=0.88, top=0.91, bottom=0.11)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s  (%d stations, median %.3f, %d below %.2f)",
             out, int(ok.sum()), float(np.nanmedian(v[ok])), below, m["centre"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-years", type=int, default=0,
                    help="drop stations with fewer paired water years than this")
    ap.add_argument("--out-dir", default=str(REPO / "results"))
    ap.add_argument("--figsize", default="7.4x9.6")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    from monitor_common import config as mcfg
    from monitor_common.s3io import read_parquet
    b, p = mcfg.bucket_prefix()

    d = read_parquet(b, f"{p}{STATS_KEY}")
    d["site_no"] = d.site_no.astype(str)
    d["r2_log"] = d.pearson_r_log ** 2
    cl = read_parquet(b, f"{p}{COMID_KEY}")[["site_no", "usgs_lat", "usgs_lon"]]
    cl["site_no"] = cl.site_no.astype(str)
    d = d.merge(cl, on="site_no", how="left").rename(
        columns={"usgs_lat": "lat", "usgs_lon": "lon"})
    miss = d.lat.isna().sum()
    if miss:
        log.warning("%d stations have no coordinates and are dropped", miss)
    d = d.dropna(subset=["lat", "lon"])
    if args.min_years:
        n0 = len(d); d = d[d.n_paired_wy >= args.min_years]
        log.info("dropped %d stations with < %d paired years", n0 - len(d), args.min_years)
    log.info("mapping %d stations", len(d))

    counties = read_parquet(b, f"{p}monitor/assets/in_counties.parquet")
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    w, h = (float(x) for x in args.figsize.lower().split("x"))
    for k in METRICS:
        draw(k, d, counties, out_dir, (w, h), args.dpi)


if __name__ == "__main__":
    main()
