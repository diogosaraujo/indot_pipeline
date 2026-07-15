"""08f_nwm_scatter.py

Diagnostic scatters for the 08f NWM-streamflow trigger — why is the confusion
matrix so false-alarm heavy?

Two questions, two figures:

  1. FLOW MAGNITUDE  (nwm_vs_usgs_peak_scatter):  per station, the peak hourly
     NWM operational streamflow vs the peak hourly USGS observed streamflow over
     the SAME common window, one panel each for A&A and open-loop.  Reveals
     whether the operational model over/under-predicts the biggest flows — and why
     A&A (with gauge DA) and open-loop (no DA) trip the trigger so differently.

  2. THRESHOLD CLIMATOLOGY  (nwm_vs_usgs_threshold_scatter):  per station, the 04c
     NWM-retrospective LP3 flood quantile (the TRIGGER threshold) vs the 04b
     USGS-peak LP3 flood quantile (the TRUTH threshold) for Q10/Q50/Q100.  If the
     04c thresholds sit below the 04b thresholds (points under the 1:1 line), the
     alarm is calibrated too low and fires on flows that never reach a real flood
     → the false-alarm explosion, independent of source.

Everything is reused from 08f (loaders, window logic, unit conversion) so the
peaks match exactly what the 08f confusion matrix saw.

Writes:
    s3://<bucket>/<prefix>analysis/figures/nwm_vs_usgs_peak_scatter.{png,svg}
    s3://<bucket>/<prefix>analysis/figures/nwm_vs_usgs_threshold_scatter.{png,svg}
    s3://<bucket>/<prefix>analysis/nwm_vs_usgs_peaks.csv   (per station × source)

Usage:
    python scripts/08f_nwm_scatter.py
"""
from __future__ import annotations

import importlib.util
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import load_config, s3_client, write_bytes_to_s3

# Reuse 08f (which itself reuses 08) — loaders, window logic, constants.
_spec = importlib.util.spec_from_file_location(
    "nwm_trigger_08f", Path(__file__).with_name("08f_nwm_trigger_analysis.py"))
f08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f08)
m = f08.m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08f_scatter")

FLOW_RPS = f08.FLOW_RPS
PEAK_CSV_KEY  = "analysis/nwm_vs_usgs_peaks.csv"
PEAK_FIG_KEY  = "analysis/figures/nwm_vs_usgs_peak_scatter"
THR_FIG_KEY   = "analysis/figures/nwm_vs_usgs_threshold_scatter"

SRC_LABEL = {"nwm_analysis_assim": "A&A (gauged, w/ DA)",
             "nwm_open_loop": "Open-loop (ungaged proxy)"}
SRC_COLOR = {"nwm_analysis_assim": "#1f78b4", "nwm_open_loop": "#e6674a"}
RP_COLOR  = {10: "#1a9850", 50: "#f46d43", 100: "#762a83"}


# ---------- shared log-log scatter ----------

def loglog_scatter(ax, x, y, color, xlabel, ylabel, title, label=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10.5)
        return
    ax.scatter(x, y, s=30, color=color, alpha=0.72, edgecolors="white", lw=0.35,
               zorder=3, label=label)
    lo = min(x.min(), y.min()) * 0.7
    hi = max(x.max(), y.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color="0.35", ls="--", lw=1.1, zorder=2, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ratio = float(np.median(y / x))
    frac_above = float(np.mean(y > x))
    try:
        r = float(spearmanr(x, y).correlation)
    except Exception:                                     # noqa: BLE001
        r = np.nan
    ax.text(0.03, 0.97,
            f"n = {len(x)}\nmedian y/x = {ratio:.2f}\nSpearman r = {r:.2f}\n"
            f"y > x: {100*frac_above:.0f}%",
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="0.8"))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.legend(loc="lower right", fontsize=8)


def _save(fig, bucket, prefix, key_stem, dpi=185):
    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=dpi, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{key_stem}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, key_stem, ext)
    plt.close(fig)


# ---------- per-station peak flows over the common window ----------

def compute_peaks(bucket, prefix, stations_08c, flow_by_site, usgs_fs, nwm_fs) -> pd.DataFrame:
    """One row per (station, source): peak hourly NWM-op cfs and USGS cfs over the
    USGS∩NWM-op window, plus the 04c/04b Q thresholds for reference."""
    rows: list[dict] = []
    for source_name, key in f08.NWM_SOURCES:
        log.info("Loading NWM operational %s ...", source_name)
        try:
            nwm_df = f08.load_nwm_operational(bucket, prefix, key)
        except Exception as e:                            # noqa: BLE001
            log.warning("Could not load %s (%s) — skipping.", key, e)
            continue
        nwm_df = nwm_df[nwm_df["site_no"].isin(stations_08c)]
        nwm_by_site: dict[str, pd.Series] = {}
        for s, gdf in nwm_df.groupby("site_no"):
            ser = gdf.set_index("datetime_utc")["value_cfs"]
            nwm_by_site[s] = ser.groupby(ser.index.floor("h")).max().sort_index()

        universe = sorted(stations_08c & set(nwm_by_site) & set(flow_by_site)
                          & set(usgs_fs.index) & set(nwm_fs.index))
        for site in universe:
            usgs_ser = flow_by_site[site]
            nwm_ser  = nwm_by_site[site]
            if usgs_ser.empty or nwm_ser.empty:
                continue
            ws = max(usgs_ser.index.min(), nwm_ser.index.min())
            we = min(usgs_ser.index.max(), nwm_ser.index.max())
            if pd.isna(ws) or pd.isna(we) or ws >= we:
                continue
            u = usgs_ser[(usgs_ser.index >= ws) & (usgs_ser.index <= we)]
            v = nwm_ser[(nwm_ser.index >= ws) & (nwm_ser.index <= we)]
            usgs_peak = float(u.max()) if len(u) and np.isfinite(u.max()) else np.nan
            nwm_peak  = float(v.max()) if len(v) and np.isfinite(v.max()) else np.nan
            rows.append({
                "site_no": site, "source": source_name,
                "usgs_peak_cfs": round(usgs_peak, 1) if np.isfinite(usgs_peak) else np.nan,
                "nwm_peak_cfs":  round(nwm_peak, 1)  if np.isfinite(nwm_peak) else np.nan,
                "q_usgs10": usgs_fs.loc[site].get("Q10"),
                "q_nwm10":  nwm_fs.loc[site].get("Q10"),
                "common_start": ws, "common_end": we, "n_hours": len(u),
            })
    return pd.DataFrame(rows)


# ---------- figures ----------

def peak_figure(peaks: pd.DataFrame, bucket, prefix):
    srcs = [s for s, _ in f08.NWM_SOURCES if (peaks["source"] == s).any()]
    fig, axes = plt.subplots(1, len(srcs), figsize=(6.3 * len(srcs), 6.0), squeeze=False)
    for ax, src in zip(axes[0], srcs):
        d = peaks[peaks["source"] == src]
        # Each product covers a different operational era (A&A ~2018→, open-loop
        # ~2021→); label it so the differing windows are explicit.
        t0 = pd.to_datetime(d["common_start"]).min()
        t1 = pd.to_datetime(d["common_end"]).max()
        title = f"{SRC_LABEL.get(src, src)}\n{t0:%Y-%m} → {t1:%Y-%m}"
        loglog_scatter(
            ax, d["usgs_peak_cfs"].to_numpy(), d["nwm_peak_cfs"].to_numpy(),
            SRC_COLOR.get(src, "steelblue"),
            "USGS observed peak (cfs)", "NWM operational peak (cfs)",
            title, label="stations")
    fig.suptitle("Per-station PEAK streamflow over the common window — NWM operational vs USGS observed\n"
                 "(points above 1:1 = NWM runs high → over-triggers; below = NWM runs low → misses)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, bucket, prefix, PEAK_FIG_KEY)


def threshold_figure(nwm_fs: pd.DataFrame, usgs_fs: pd.DataFrame, bucket, prefix):
    both = sorted(set(nwm_fs.index) & set(usgs_fs.index))
    fig, axes = plt.subplots(1, len(FLOW_RPS), figsize=(5.6 * len(FLOW_RPS), 5.6), squeeze=False)
    for ax, rp in zip(axes[0], FLOW_RPS):
        qu = usgs_fs.loc[both, f"Q{rp}"].to_numpy(float)
        qn = nwm_fs.loc[both, f"Q{rp}"].to_numpy(float)
        loglog_scatter(
            ax, qu, qn, RP_COLOR.get(rp, "steelblue"),
            f"USGS-peak LP3  Q{rp} (04b, cfs)", f"NWM-retro LP3  Q{rp} (04c, cfs)",
            f"Q{rp} trigger vs truth threshold", label="stations")
    fig.suptitle("Flood-threshold climatology — 04c NWM-retrospective (trigger) vs 04b USGS-peak (truth)\n"
                 "(points near 1:1 → thresholds agree; the false alarms are NOT a threshold artifact)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, bucket, prefix, THR_FIG_KEY)


# ---------- main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading thresholds (04b, 04c) + USGS streamflow...")
    usgs_q = m.load_flow_stats(bucket, prefix)
    nwm_q  = f08.load_nwm_thresholds(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)

    tc = m._read_parquet_s3(bucket, f"{prefix}{f08.TC_KEY}", columns=["site_no"])
    stations_08c = set(tc["site_no"].astype(str))

    flow_by_site = {s: g.set_index("datetime_utc")["value_cfs"].sort_index()
                    for s, g in streamflow.groupby("site_no")}
    usgs_fs = usgs_q.drop_duplicates("site_no").set_index("site_no")
    nwm_fs  = nwm_q.drop_duplicates("site_no").set_index("site_no")
    # Threshold panel: only gauges with a valid 04c fit AND in the 08c set.
    nwm_thr = nwm_fs[nwm_fs[[f"Q{rp}" for rp in FLOW_RPS]].notna().any(axis=1)]
    nwm_thr = nwm_thr.loc[[s for s in nwm_thr.index if s in stations_08c]]
    # Align to the SAME index (order + membership) so element-wise ratio/compare work.
    usgs_thr = usgs_fs.reindex(nwm_thr.index)

    peaks = compute_peaks(bucket, prefix, stations_08c, flow_by_site, usgs_fs, nwm_fs)
    if peaks.empty:
        log.error("No peak rows computed.")
        return

    # ── Diagnostics to the log (the same story the figures tell) ──────────────
    log.info("Median NWM/USGS PEAK ratio by source:\n%s",
             peaks.assign(ratio=peaks["nwm_peak_cfs"] / peaks["usgs_peak_cfs"])
                  .groupby("source")["ratio"].median().round(3).to_string())
    for rp in FLOW_RPS:
        ratio = (nwm_thr[f"Q{rp}"] / usgs_thr[f"Q{rp}"]).median()
        frac_low = float((nwm_thr[f"Q{rp}"] < usgs_thr[f"Q{rp}"]).mean())
        log.info("Q%d threshold  median 04c/04b ratio = %.2f  |  04c below 04b at %.0f%% of gauges",
                 rp, ratio, 100 * frac_low)

    # ── CSV + figures ─────────────────────────────────────────────────────────
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{PEAK_CSV_KEY}",
                           Body=peaks.to_csv(index=False).encode(), ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s", bucket, prefix, PEAK_CSV_KEY)

    peak_figure(peaks, bucket, prefix)
    threshold_figure(nwm_thr, usgs_thr, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
