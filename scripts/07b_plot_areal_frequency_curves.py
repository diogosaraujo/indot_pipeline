#!/usr/bin/env python3
"""07b_plot_areal_frequency_curves.py

QC plots for the areal (watershed-mean) GEV precipitation-frequency fits from
07a — analogous to the LP3 frequency-curve figures for streamflow (04b).

For each station it renders a 2x4 panel (one panel per duration) showing the
observed annual maxima at Weibull plotting positions against the fitted GEV
curve, with per-panel goodness-of-fit (PPCC, KS). It also writes a
gof_summary.csv across ALL station x duration fits so bad fits are easy to find.

Fits are recomputed with 07a's OWN functions, so the plotted curve is exactly
the one written to areal_precip_frequency.parquet.

Run on EC2:
    python scripts/07b_plot_areal_frequency_curves.py                 # all 106, panel per station
    python scripts/07b_plot_areal_frequency_curves.py --site 05522500 # just one station
"""
from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

from utils import load_config, s3_client

# ── import 07a (digit-prefixed) to reuse its fit + loaders ────────────────────
_spec = importlib.util.spec_from_file_location(
    "areal07a", Path(__file__).with_name("07a_areal_precip_frequency.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

OUT_PREFIX = "analysis/areal_frequency_curves"


def gev_cdf(x, xi, alpha, k):
    """GEV CDF consistent with 07a's gev_quantile (Hosking parameterization)."""
    x = np.asarray(x, float)
    if abs(k) < 1e-6:
        return np.exp(-np.exp(-(x - xi) / alpha))
    arg = 1.0 - k * (x - xi) / alpha
    with np.errstate(invalid="ignore"):
        F = np.where(arg > 0, np.exp(-np.power(arg, 1.0 / k)),
                     np.where(k > 0, 1.0, 0.0))
    return F


def gof(ams, xi, alpha, k):
    """PPCC, RMSE (in), KS stat + p vs the fitted GEV. Weibull plotting positions."""
    x = np.sort(ams)
    n = x.size
    p = np.arange(1, n + 1) / (n + 1.0)
    fit = m.gev_quantile(p, xi, alpha, k)
    ppcc = float(np.corrcoef(x, fit)[0, 1])
    rmse = float(np.sqrt(np.mean((x - fit) ** 2)))
    ks, ksp = stats.kstest(x, lambda v: gev_cdf(v, xi, alpha, k))
    return ppcc, rmse, float(ks), float(ksp)


def plot_panel(ax, ams, duration):
    """One duration's frequency curve on ax; returns a gof dict row."""
    l1, l2, t, t3, t4 = m.sample_lmoments(ams)
    xi, alpha, k = m.gev_params_from_lmom(l1, l2, t3)
    ppcc, rmse, ks, ksp = gof(ams, xi, alpha, k)

    x = np.sort(ams)
    n = x.size
    T_emp = (n + 1) / (n + 1 - np.arange(1, n + 1))          # Weibull return period
    T_fit = np.logspace(np.log10(1.05), np.log10(1000), 300)
    Q_fit = m.gev_quantile(1.0 - 1.0 / T_fit, xi, alpha, k)

    ax.scatter(T_emp, x, s=14, color="steelblue", zorder=5,
               edgecolors="white", lw=0.3)
    ax.plot(T_fit, Q_fit, color="firebrick", lw=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 5, 10, 25, 100, 1000]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25, ls="--")
    flag = "  (!)" if (ppcc < 0.98 or ksp < 0.05) else ""
    ax.set_title(f"D={duration}h  (n={n}){flag}", fontsize=8)
    ax.text(0.03, 0.97, f"k={k:.2f}\nPPCC={ppcc:.3f}\nKS p={ksp:.2f}",
            transform=ax.transAxes, va="top", fontsize=6.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="lightgrey"))
    return {"duration_hr": duration, "n": n, "mean_in": round(l1, 4),
            "gev_xi": round(xi, 4), "gev_alpha": round(alpha, 4), "gev_k": round(k, 4),
            "ppcc": round(ppcc, 4), "rmse_in": round(rmse, 4),
            "ks": round(ks, 4), "ks_pval": round(ksp, 4)}


def make_station_figure(site, series, min_years, min_frac):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    rows = []
    for ax, d in zip(axes.ravel(), m.DURATIONS_HR):
        ams = m.annual_maxima(series, d, min_frac)
        if len(ams) < min_years:
            ax.set_title(f"D={d}h  (n={len(ams)} < {min_years})", fontsize=8)
            ax.set_axis_off()
            continue
        row = plot_panel(ax, ams, d)
        row["site_no"] = site
        rows.append(row)
    for ax in axes[1, :]:
        ax.set_xlabel("Return period (yr)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Areal precip depth (in)", fontsize=8)
    fig.suptitle(f"Areal GEV frequency curves — Station {site}  "
                 f"(watershed-mean MRMS; (!) = PPCC<0.98 or KS p<0.05)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="QPE_01H_Pass2")
    ap.add_argument("--universe-key", default="analysis/event_confusion_matrix_tc.parquet")
    ap.add_argument("--site", default=None, help="render only this station")
    ap.add_argument("--min-years", type=int, default=m.MIN_YEARS)
    ap.add_argument("--min-frac", type=float, default=m.MIN_FRAC)
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    keep = {args.site} if args.site else m.load_universe(bucket, prefix, args.universe_key)
    series = m.load_watershed_series(bucket, prefix, args.product, keep)
    print(f"Plotting {len(series)} station(s)")

    import pandas as pd
    all_rows = []
    for i, (site, s) in enumerate(sorted(series.items()), 1):
        fig, rows = make_station_figure(site, s, args.min_years, args.min_frac)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        s3_client().put_object(Bucket=bucket,
                               Key=f"{prefix}{OUT_PREFIX}/{site}_panels.png",
                               Body=buf.getvalue(), ContentType="image/png")
        all_rows.extend(rows)
        if i % 20 == 0:
            print(f"  {i}/{len(series)}")

    gof_df = pd.DataFrame(all_rows)[
        ["site_no", "duration_hr", "n", "mean_in", "gev_xi", "gev_alpha", "gev_k",
         "ppcc", "rmse_in", "ks", "ks_pval"]].sort_values(["ppcc"])
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{OUT_PREFIX}/gof_summary.csv",
                           Body=gof_df.to_csv(index=False).encode(), ContentType="text/csv")

    bad = gof_df[(gof_df.ppcc < 0.98) | (gof_df.ks_pval < 0.05)]
    print(f"\nWrote {len(series)} panel figures + gof_summary.csv to "
          f"s3://{bucket}/{prefix}{OUT_PREFIX}/")
    print(f"Fits flagged (PPCC<0.98 or KS p<0.05): {len(bad)} / {len(gof_df)}")
    if len(bad):
        print("Worst 10 by PPCC:")
        print(bad.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
