"""compare_iv_vs_peak.py

Validate that the USGS annual peak (peak_va) and our IV-derived annual maximum
are the SAME physical quantity, by comparing them for matching water years at
each station (their common period).

If they measure the same thing, matched pairs should cluster tightly on the 1:1
line with peak_va >= IV max (the official crest can sit a hair above the highest
IV sample for flashy streams).  A systematic divergence would signal a units or
time-scale problem.

Daily-average peaks (peak_cd contains '1') are EXCLUDED from the comparison —
those are not instantaneous and would (correctly) plot below the 1:1 line.

Outputs (to s3://<bucket>/<prefix>analysis/iv_vs_peak_comparison/):
    scatter.png                 – log-log scatter of all matched pairs + metrics
    per_station_metrics.csv     – per-station median ratio, n_pairs, etc.

Usage:
    python scripts/compare_iv_vs_peak.py
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from utils import load_config, s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("cmp_iv_peak")

MIN_OBS_PER_YEAR = 100   # match 04b: IV water-year counts only if >= 100 readings
S3_OUT = "analysis/iv_vs_peak_comparison/"


def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _upload_png(fig: plt.Figure, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf, ContentType="image/png")
    log.info("  -> s3://%s/%s", bucket, key)


def _upload_csv(df: pd.DataFrame, bucket: str, key: str) -> None:
    s3_client().put_object(Bucket=bucket, Key=key,
                           Body=df.to_csv(index=False).encode(),
                           ContentType="text/csv")
    log.info("  -> s3://%s/%s", bucket, key)


def iv_annual_max(sf: pd.DataFrame) -> pd.DataFrame:
    """IV annual maximum per (site_no, water_year), with the >=100-reading filter."""
    sf = sf[sf["value_cfs"] > 0].copy()
    sf["water_year"] = sf["datetime"].dt.year + (sf["datetime"].dt.month >= 10).astype(int)
    g = sf.groupby(["site_no", "water_year"])["value_cfs"]
    out = g.agg(iv_max="max", n_obs="count").reset_index()
    return out[out["n_obs"] >= MIN_OBS_PER_YEAR]


def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading USGS annual peaks...")
    peaks = _read_parquet_s3(bucket, f"{prefix}streamflow/annual_peaks/usgs_annual_peaks.parquet")
    peaks["site_no"] = peaks["site_no"].astype(str)

    # ── Exclude non-instantaneous (code 1) and regulated/diverted (codes 5,6) ──
    cd = peaks["peak_cd"].fillna("").astype(str)
    peaks["is_daily"]     = cd.str.contains("1").values
    peaks["is_regulated"] = (cd.str.contains("5") | cd.str.contains("6")).values
    peaks["is_clean"]     = ~peaks["is_daily"] & ~peaks["is_regulated"]
    n_total_sites = peaks["site_no"].nunique()

    # Per-station eligibility BEFORE filtering pairs, to answer "how many left?"
    per_site_all = peaks.groupby("site_no").agg(
        n_peaks=("peak_va", "size"),
        n_regulated=("is_regulated", "sum"),
        n_clean=("is_clean", "sum"),
    )
    # Rule A: drop regulated peaks, keep station if >=10 clean peaks remain
    keep_A = per_site_all[per_site_all["n_clean"] >= 10]
    # Rule B: drop the whole station if it has ANY regulated peak
    fully_unreg = per_site_all[per_site_all["n_regulated"] == 0]
    keep_B = fully_unreg[fully_unreg["n_clean"] >= 10]

    n_any_reg     = int((per_site_all["n_regulated"] > 0).sum())
    n_fully_unreg = len(fully_unreg)
    log.info("Station eligibility:")
    log.info("  total sites with peaks                         : %d", n_total_sites)
    log.info("  sites with ANY regulated peak                  : %d", n_any_reg)
    log.info("  fully unregulated sites                        : %d  (%d - %d)",
             n_fully_unreg, n_total_sites, n_any_reg)
    log.info("  -- applying the >=10 instantaneous-peak record filter --")
    log.info("  Rule A (drop reg. peaks; keep station >=10)    : %d sites", len(keep_A))
    log.info("  Rule B (fully unregulated AND >=10 peaks)      : %d sites", len(keep_B))

    # Apply exclusion for the comparison itself: instantaneous + unregulated peaks
    n_drop = int((~peaks["is_clean"]).sum())
    peaks = peaks[peaks["is_clean"]]
    log.info("Excluded %d peaks (daily-avg + regulated); %d clean peaks remain",
             n_drop, len(peaks))

    log.info("Loading IV streamflow...")
    sf = _read_parquet_s3(bucket, f"{prefix}streamflow/instantaneous/all_gauges_long.parquet",
                          columns=["site_no", "datetime", "value_cfs"])
    sf["site_no"]   = sf["site_no"].astype(str)
    sf["datetime"]  = pd.to_datetime(sf["datetime"], utc=True)
    sf["value_cfs"] = pd.to_numeric(sf["value_cfs"], errors="coerce")
    iv = iv_annual_max(sf)
    log.info("IV annual maxima: %d (site, water_year) pairs", len(iv))

    # Match on the common period: same site AND same water year
    merged = iv.merge(
        peaks[["site_no", "water_year", "peak_va"]],
        on=["site_no", "water_year"], how="inner",
    )
    merged = merged[(merged["iv_max"] > 0) & (merged["peak_va"] > 0)]
    merged["ratio"] = merged["peak_va"] / merged["iv_max"]
    log.info("Matched pairs (common period): %d across %d stations",
             len(merged), merged["site_no"].nunique())

    # ── Overall metrics (log space) ───────────────────────────────────────────
    x = np.log10(merged["iv_max"].values)
    y = np.log10(merged["peak_va"].values)
    r2      = float(np.corrcoef(x, y)[0, 1] ** 2)
    rmse    = float(np.sqrt(np.mean((y - x) ** 2)))
    nse     = float(1 - np.sum((y - x) ** 2) / np.sum((y - np.mean(y)) ** 2))
    med_r   = float(merged["ratio"].median())
    mean_r  = float(merged["ratio"].mean())
    within5  = float((merged["ratio"].between(0.95, 1.05)).mean() * 100)
    within10 = float((merged["ratio"].between(0.90, 1.10)).mean() * 100)

    log.info("R^2(log)=%.4f  RMSE(log)=%.4f  NSE=%.4f", r2, rmse, nse)
    log.info("ratio peak/IV: median=%.3f mean=%.3f | within 5%%=%.1f%% within 10%%=%.1f%%",
             med_r, mean_r, within5, within10)

    # ── Per-station metrics ───────────────────────────────────────────────────
    per = merged.groupby("site_no").agg(
        n_pairs=("ratio", "size"),
        median_ratio=("ratio", "median"),
        min_ratio=("ratio", "min"),
        max_ratio=("ratio", "max"),
    ).reset_index().sort_values("median_ratio")
    _upload_csv(per, bucket, f"{prefix}{S3_OUT}per_station_metrics.csv")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    lo = min(merged["iv_max"].min(), merged["peak_va"].min()) * 0.8
    hi = max(merged["iv_max"].max(), merged["peak_va"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--", label="1:1 line", zorder=3)
    ax.scatter(merged["iv_max"], merged["peak_va"], s=12, alpha=0.35,
               color="steelblue", edgecolors="none", zorder=2,
               label=f"{len(merged):,} matched (site, water-year) pairs")

    txt = (
        f"Stations      = {merged['site_no'].nunique()}\n"
        f"Matched pairs = {len(merged):,}\n"
        f"{'-'*26}\n"
        f"R^2 (log)     = {r2:.4f}\n"
        f"RMSE (log)    = {rmse:.4f}\n"
        f"NSE           = {nse:.4f}\n"
        f"{'-'*26}\n"
        f"peak/IV median= {med_r:.3f}\n"
        f"peak/IV mean  = {mean_r:.3f}\n"
        f"within  5%    = {within5:.1f}%\n"
        f"within 10%    = {within10:.1f}%"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="lightgrey"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("IV annual maximum (cfs)", fontsize=11)
    ax.set_ylabel("USGS annual peak  peak_va (cfs)", fontsize=11)
    ax.set_title("USGS annual peak vs IV-derived annual maximum\n"
                 "matched water years, instantaneous peaks only (code 1 excluded)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25, ls="--")
    fig.tight_layout()
    _upload_png(fig, bucket, f"{prefix}{S3_OUT}scatter.png")
    plt.close(fig)

    # Flag stations that disagree the most
    worst = per[(per["median_ratio"] < 0.9) | (per["median_ratio"] > 1.1)]
    if len(worst):
        log.info("Stations with median peak/IV ratio outside [0.9, 1.1]: %d", len(worst))
        log.info("\n%s", worst.to_string(index=False))
    else:
        log.info("All stations agree within 10%% on the median ratio.")


if __name__ == "__main__":
    main()
