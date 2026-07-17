#!/usr/bin/env python3
"""regenerate_lp3_figure_manuscript.py

Re-render the 04b LP3 frequency-curve figure at MANUSCRIPT size (default
6.5in x 4.0in) with legible fonts, and with the goodness-of-fit stats block
moved to sit just ABOVE the legend (lower-right) instead of the upper-left.

The fit itself is reproduced EXACTLY by importing 04b's own functions
(clean_peaks_for_site / fit_lp3 / compute_gof / lp3_quantile), so the curve,
skews, and GoF match the production figure — only the layout/sizing changes.

Run on EC2 (instance role must read the private output bucket) and upload to S3:
    python scripts/regenerate_lp3_figure_manuscript.py                 # station 05522500, PNG@600
    python scripts/regenerate_lp3_figure_manuscript.py --site 05522500 --format pdf
    python scripts/regenerate_lp3_figure_manuscript.py --local ./fig.png   # also write local copy
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

# ── Import 04b as a module (filename starts with a digit → load via importlib) ──
_04B_PATH = Path(__file__).resolve().parent / "04b_regression_flows.py"
_spec = importlib.util.spec_from_file_location("mod04b", _04B_PATH)
m04b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m04b)                       # defines fit funcs + constants

from utils import load_config, s3_client            # noqa: E402


def build_fit(bucket: str, prefix: str, site_no: str):
    """Reproduce 04b's fit for one station. Returns (annual_max, params, q_est, gof)."""
    peaks = m04b._read_parquet_s3(
        bucket, f"{prefix}{m04b.PEAKS_KEY}",
        columns=["site_no", "water_year", "peak_va", "peak_cd"],
    )
    peaks["site_no"] = peaks["site_no"].astype(str)
    site_peaks = peaks[peaks["site_no"] == site_no]
    if site_peaks.empty:
        raise SystemExit(f"No annual-peak record for site {site_no}.")

    peaks_s, is_regulated = m04b.clean_peaks_for_site(site_peaks)
    if is_regulated:
        raise SystemExit(f"{site_no} is flagged regulated (Rule B) — 04b would not fit it.")
    if len(peaks_s) < m04b.MIN_YEARS:
        raise SystemExit(f"{site_no} has only {len(peaks_s)} clean peaks (< {m04b.MIN_YEARS}).")

    inv = m04b._read_parquet_s3(
        bucket, f"{prefix}stations/indiana_streamflow_sites.parquet",
        columns=["site_no", "dec_lat_va"],
    )
    inv["site_no"] = inv["site_no"].astype(str)
    lat = float(inv.set_index("site_no")["dec_lat_va"].to_dict().get(site_no, 39.8))

    log_q  = np.log10(peaks_s.values)
    params = m04b.fit_lp3(log_q, lat)
    gof    = m04b.compute_gof(log_q, params)
    q_est  = {rp: m04b.lp3_quantile(rp, params["mean_log"], params["std_log"], params["skew"])
              for rp in m04b.RETURN_PERIODS}
    return peaks_s, params, q_est, gof


def plot_manuscript(site_no, annual_max, params, q_estimates, gof) -> plt.Figure:
    n_total   = params["n"]
    n_c       = params.get("n_censored", 0)
    n_u       = params.get("n_uncensored", n_total)
    threshold = params.get("threshold_log")
    mean_y, std_y, skew = params["mean_log"], params["std_log"], params["skew"]
    method    = params.get("fitting_method", "MOM")
    skew_site = params.get("skew_at_site", skew)
    skew_reg  = params.get("skew_regional", skew)
    w_s       = params.get("weight_at_site", 1.0)
    w_r       = params.get("weight_regional", 0.0)

    all_q   = np.sort(annual_max.values)
    log_all = np.log10(all_q)
    uncensored_q = all_q[log_all >= threshold] if threshold is not None else all_q
    censored_q   = all_q[log_all <  threshold] if threshold is not None else np.array([])

    m_u   = np.arange(1, n_u + 1)
    T_emp = (n_total + 1) / (n_total + 1 - (n_c + m_u))
    T_fit = np.logspace(np.log10(1.01), np.log10(2000), 400)
    Q_fit = np.array([m04b.lp3_quantile(T, mean_y, std_y, skew) for T in T_fit])

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    ax.scatter(T_emp, uncensored_q, s=16, color="steelblue", zorder=5,
               label="Observed — uncensored (Weibull)", edgecolors="white", lw=0.3)
    if len(censored_q) > 0:
        T_cens = 10 ** threshold
        ax.scatter([1.5] * len(censored_q), [T_cens] * len(censored_q),
                   marker="v", s=26, color="none", edgecolors="grey", lw=1.0,
                   zorder=5, label=f"Censored (GB low outlier, n={n_c})")
        ax.axhline(T_cens, color="grey", ls=":", lw=0.9,
                   label=f"Censoring threshold = {T_cens:,.0f} cfs")
    ax.plot(T_fit, Q_fit, color="firebrick", lw=1.4, label="Fitted LP3")

    rp_colors = {10: "goldenrod", 25: "darkorange", 50: "coral", 100: "red"}
    for rp in m04b.RETURN_PERIODS:
        q_val = q_estimates.get(rp)
        if q_val is None:
            continue
        c = rp_colors.get(rp, "grey")
        ax.axvline(rp, color=c, ls="--", lw=0.8, alpha=0.7)
        ax.scatter([rp], [q_val], marker="D", s=24, color=c, zorder=6)
        # Label offset above the point cloud, drawn ON TOP of the dots (high
        # zorder) with a white backing box so the value stays legible.
        ax.annotate(
            f"Q{rp}\n{q_val:,.0f} cfs",
            xy=(rp, q_val), xytext=(rp * 1.08, q_val * 1.22),
            fontsize=8, fontweight="bold", color=c, va="bottom", ha="left",
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Return Period (years)", fontsize=10)
    ax.set_ylabel("Peak Discharge (cfs)", fontsize=10)
    ax.set_title(
        f"LP3 Frequency Curve — Station {site_no}\n"
        f"Bulletin 17C {method}, weighted skew (at-site w={w_s:.2f} / regional w={w_r:.2f})",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # Y-axis: force consistent plain (non-scientific) labels on the log scale.
    # matplotlib's default labels the decade tick plainly ("1000") but the
    # intermediate ticks in sci notation ("2×10³", "6×10²") — hence the mix.
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 3, 4, 6), numticks=15))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.25, ls="--")

    leg = ax.legend(loc="lower right", fontsize=6, framealpha=0.9)
    fig.tight_layout()

    # ── Stats block just ABOVE the legend, right-aligned to it ────────────────
    cens_line = (f"GB censored     = {n_c} below {10**threshold:,.0f} cfs\n"
                 if threshold is not None else "")
    txt = (
        f"n = {n_total} water-years  [{method}]\n"
        + cens_line
        + f"Mean(log Q)     = {mean_y:.3f}\n"
        f"Std(log Q)      = {std_y:.3f}\n"
        f"{'─' * 26}\n"
        f"Skew at-site    = {skew_site:.3f}  (w={w_s:.2f})\n"
        f"Skew regional   = {skew_reg:.3f}  (w={w_r:.2f})\n"
        f"Skew weighted   = {skew:.3f}\n"
        f"{'─' * 26}\n"
        f"PPCC            = {gof['ppcc']:.4f}\n"
        f"RMSE (log)      = {gof['rmse_log']:.4f}\n"
        f"KS stat         = {gof['ks_stat']:.4f}  (p = {gof['ks_pval']:.3f})"
    )
    fig.canvas.draw()                                # realize legend geometry
    inv = ax.transAxes.inverted()
    bb  = leg.get_window_extent()
    (lx0, ly0) = inv.transform((bb.x0, bb.y0))
    (lx1, ly1) = inv.transform((bb.x1, bb.y1))
    ax.text(lx1, ly1 + 0.025, txt, transform=ax.transAxes,
            va="bottom", ha="right", fontsize=6, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="lightgrey"))
    return fig


def main() -> None:
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="05522500")
    ap.add_argument("--width",  type=float, default=6.5)
    ap.add_argument("--height", type=float, default=4.0)
    ap.add_argument("--format", choices=["png", "pdf", "tif"], default="png",
                    help="png/tif = raster at --dpi; pdf = vector (crispest for manuscripts)")
    ap.add_argument("--dpi", type=int, default=600, help="raster DPI (png/tif)")
    ap.add_argument("--s3-key", default=None, help="override S3 key")
    ap.add_argument("--local", default=None, help="also write a local copy to this path")
    args = ap.parse_args()

    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    annual_max, params, q_est, gof = build_fit(bucket, prefix, args.site)
    fig = plot_manuscript(args.site, annual_max, params, q_est, gof)

    buf = io.BytesIO()
    # No bbox_inches='tight' → the saved canvas is EXACTLY --width x --height.
    fmt = "tiff" if args.format == "tif" else args.format
    save_kw = {"format": fmt}
    if args.format in ("png", "tif"):
        save_kw["dpi"] = args.dpi
    fig.savefig(buf, **save_kw)
    buf.seek(0)
    data = buf.getvalue()

    ctype = {"png": "image/png", "pdf": "application/pdf", "tif": "image/tiff"}[args.format]
    key = args.s3_key or (f"{prefix}analysis/lp3_frequency_curves/"
                          f"{args.site}_lp3_manuscript_{args.width}x{args.height}in.{args.format}")
    s3_client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=ctype)
    print(f"Uploaded s3://{bucket}/{key}  ({len(data):,} bytes, {args.width}x{args.height}in)")

    if args.local:
        Path(args.local).write_bytes(data)
        print(f"Also wrote local copy: {args.local}")


if __name__ == "__main__":
    main()
