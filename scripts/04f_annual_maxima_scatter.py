"""Paired annual maxima, NWM retrospective vs USGS observed peaks — one panel.

A standalone version of panel (a) of 04d's fig_annual_maxima, with the fit
statistics on it. Reads analysis/nwm_vs_usgs/annual_maxima_paired.parquet, which
04d writes, so it needs no scipy and runs anywhere.

On the statistics shown
───────────────────────
R2 and r are quoted in LOG10 space, because that is the space the plot is drawn
in and the data span five orders of magnitude: on raw values a handful of the
largest floods set the number almost by themselves (raw R2 0.849 against log10
R2 0.796, from the same points). Spearman rho is on ranks and so is free of both
scales.

Correlation measures whether the two move together, NOT whether they agree. Two
series can correlate at 0.99 and be a factor of three apart. So the box also
carries the median ratio and the share of station-years inside a factor of two,
which are agreement, and the Nash-Sutcliffe efficiency against the 1:1 line,
which is the closest single number to "how far from identity".

The fitted line is an OLS fit in log space and is NOT the 1:1 line; both are
drawn so the difference is visible. Its slope above 1 means NWM's deficit
narrows as floods get bigger — about 0.73x at 100 cfs against 0.99x at
100,000 cfs — which is the opposite of the usual worry about models clipping
peaks.

Usage
    python scripts/04f_annual_maxima_scatter.py
    python scripts/04f_annual_maxima_scatter.py --exclude-regulated --out x.png
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("am_scatter")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "monitor"))
PAIRED_KEY = "analysis/nwm_vs_usgs/annual_maxima_paired.parquet"

# dataviz reference palette, light surface
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson on ranks — pandas routes method='spearman' through scipy."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-regulated", action="store_true",
                    help="drop Rule-B regulated stations (they carry no 04b fit, "
                         "but their annual maxima are in this table)")
    ap.add_argument("--out", default=str(REPO / "results" / "annual_maxima_scatter.png"))
    ap.add_argument("--figsize", default="7.8x7.2", help="inches, WxH")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    from monitor_common import config as mcfg
    from monitor_common.s3io import read_parquet
    b, p = mcfg.bucket_prefix()
    d = read_parquet(b, f"{p}{PAIRED_KEY}")
    if args.exclude_regulated and "is_regulated" in d.columns:
        n0 = len(d); d = d[~d.is_regulated.astype(bool)]
        log.info("dropped %d regulated station-years", n0 - len(d))
    d = d[(d.usgs_peak_cfs > 0) & (d.nwm_peak_cfs > 0)]

    x = d.usgs_peak_cfs.to_numpy(float); y = d.nwm_peak_cfs.to_numpy(float)
    lx, ly = np.log10(x), np.log10(y)
    r = float(np.corrcoef(lx, ly)[0, 1])
    rho = spearman(x, y)
    slope, icept = np.polyfit(lx, ly, 1)
    ratio = y / x
    med = float(np.median(ratio))
    within2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    nse = 1 - np.sum((ly - lx) ** 2) / np.sum((ly - ly.mean()) ** 2)
    n_st = d.site_no.nunique() if "site_no" in d.columns else np.nan
    log.info("n=%d  stations=%s  R2(log)=%.3f  rho=%.3f  slope=%.3f  median=%.2fx",
             len(d), n_st, r ** 2, rho, slope, med)

    w, h = (float(v) for v in args.figsize.lower().split("x"))
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, ls="-"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)

    lim = (min(x.min(), y.min()) * 0.6, max(x.max(), y.max()) * 1.7)
    xs = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 100)
    ax.plot(lim, lim, color=AXIS, lw=1.4, zorder=3, label="1:1")
    ax.plot(xs, 10 ** (slope * np.log10(xs) + icept), color=SERIES_2, lw=2.0,
            zorder=4, label=f"OLS fit (log space), slope {slope:.2f}")
    ax.scatter(x, y, s=9, color=SERIES_1, alpha=0.20, lw=0, zorder=2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=8))
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=8))
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("USGS instantaneous annual peak (cfs)", fontsize=10.5, color=INK_2)
    ax.set_ylabel("NWM retrospective water-year maximum (cfs)", fontsize=10.5, color=INK_2)
    ax.set_title("Annual maxima — NWM retrospective vs USGS observed peaks",
                 fontsize=12.5, color=INK, loc="left", pad=10)

    box = "\n".join([
        f"n = {len(d):,} station-years, {n_st} stations",
        "",
        f"R²  = {r ** 2:.3f}       (log₁₀)",
        f"r   = {r:.3f}       Pearson, log₁₀",
        f"ρ   = {rho:.3f}       Spearman, rank",
        "",
        f"median NWM/USGS = {med:.2f}×",
        f"within a factor of 2 = {within2 * 100:.0f}%",
        f"NSE vs 1:1 (log₁₀) = {nse:.2f}",
    ])
    ax.text(0.035, 0.965, box, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.2, color=INK, family="monospace", zorder=6,
            bbox=dict(fc="white", ec=GRID, lw=0.8, boxstyle="round,pad=0.55"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.text(0.5, 0.012,
             "R² and r are log₁₀ — on raw values a few of the largest floods set them. "
             "Correlation is not agreement: the ratio and NSE speak to that.",
             ha="center", fontsize=8, color=MUTED)

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    log.info("wrote %s  (%.1f x %.1f in)", out, w, h)


if __name__ == "__main__":
    main()
