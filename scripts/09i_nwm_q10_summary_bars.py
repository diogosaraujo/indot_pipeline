"""09i_nwm_q10_summary_bars.py

09a-style skill bars for the NWM-streamflow trigger (08f), Q10 ONLY, comparing the
two chosen operating points:

    A&A  /  USGS-Q (04b)     gauged upper bound   (source=nwm_analysis_assim, thresh_src=usgs_peak_04b)
    Open-loop  /  NWM-Q (04c) gauge-free           (source=nwm_open_loop,      thresh_src=nwm_retro_04c)

Truth is USGS observed >= Q10 (04b) in both.  Like 09a, each scenario shows four bars:

    POD, FAR, CSI  -> left y-axis   (bounded skill scores)
    FAF            -> right y-axis  (False Alarm Frequency = false alarms per
                     station-year, log scale)

    POD = TP/(TP+FN)   FAR = FP/(TP+FP)   CSI = TP/(TP+FP+FN)
    FAF = sum(FP) / sum(n_common_hours / 8766)   [false alarms per station-year]

Reads:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_nwm.parquet
        (08f output — REQUIRES the 4-scenario version with the `thresh_src` column;
         re-run 08f_nwm_trigger_analysis.py first if that column is absent.)

Writes:
    s3://<bucket>/<prefix>analysis/figures/nwm_q10_summary_bars.{png,svg}

Usage:
    python scripts/09i_nwm_q10_summary_bars.py
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
from matplotlib.patches import Patch

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09i_nwm_q10")

FLOW_RP        = 10
HOURS_PER_YEAR = 8766.0
EPS            = 1e-9
INPUT_KEY      = "analysis/event_confusion_matrix_nwm.parquet"
OUTPUT_STEM    = "analysis/figures/nwm_q10_summary_bars"

# (legend label, source, thresh_src, short tag)
SCENARIOS = [
    ("Analysis & Assimilation (A&A)", "nwm_analysis_assim", "usgs_peak_04b", "gauged"),
    ("Open-Loop",                     "nwm_open_loop",      "nwm_retro_04c", "gauge-free"),
]
LEFT_METRICS = [("POD", "#2c7fb8"), ("FAR", "#d95f0e"), ("CSI", "#31a354")]
FAF_COLOR    = "#6a51a3"


def _read_parquet(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _metrics(df: pd.DataFrame, source: str, thresh_src: str) -> dict | None:
    sub = df[(df["source"] == source) & (df["thresh_src"] == thresh_src)
             & (df["flow_rp_yr"] == FLOW_RP)]
    if sub.empty:
        return None
    # one row per station for a fixed (source, thresh_src, flow_rp); guard against dups
    per_stn = sub.drop_duplicates("site_no")
    tp, fp, fn = float(per_stn["tp"].sum()), float(per_stn["fp"].sum()), float(per_stn["fn"].sum())
    station_years = float(per_stn["n_common_hours"].sum()) / HOURS_PER_YEAR
    return {
        "POD": tp / (tp + fn + EPS),
        "FAR": fp / (tp + fp + EPS),
        "CSI": tp / (tp + fp + fn + EPS),
        "FAF": fp / (station_years + EPS),
        "n_events": int(per_stn["n_flow_events"].sum()),
        "n_stations": int(per_stn["site_no"].nunique()),
    }


def make_figure(results: list[dict], bucket: str, prefix: str) -> None:
    """Two products per metric, formatted like 09g: metric names as the x labels and
    09g's 0.62 group footprint / bar spacing.  The two products are distinguished by
    fill (solid vs hatched) with an explicit legend of neutral grey patches."""
    metrics = [name for name, _ in LEFT_METRICS] + ["FAF"]      # POD, FAR, CSI, FAF
    colors  = [c for _, c in LEFT_METRICS] + [FAF_COLOR]
    hatches = [None, "////"]                                    # product A solid, product B hatched
    x = np.arange(len(metrics))
    n = len(results)
    width = 0.62 / n                                            # group footprint == 09g's single-bar width
    offs  = [(i - (n - 1) / 2) * width for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    for gi, (metric, color) in enumerate(zip(metrics, colors)):
        for pi, r in enumerate(results):
            axis = ax2 if metric == "FAF" else ax
            v = r["FAF"] if metric == "FAF" else r[metric]
            xpos = x[gi] + offs[pi]
            axis.bar(xpos, v, width, color=color, hatch=hatches[pi],
                     edgecolor="white" if hatches[pi] else "none", linewidth=0.0)
            axis.text(xpos, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12)

    left_max = max(r[k] for r in results for k in ("POD", "FAR", "CSI"))
    faf_max  = max(r["FAF"] for r in results)
    ax.set_ylim(0, (left_max if left_max > 0 else 1.0) * 1.22)
    ax2.set_ylim(0, (faf_max if faf_max > 0 else 1.0) * 1.30)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.set_ylabel("FAF (per station-yr)", fontsize=16)

    # Legend keys the fill pattern (product), not colour (which encodes the metric),
    # so use neutral grey patches: solid = product A, hatched = product B.  Placed
    # above the axes so it never overlaps the bars whatever the values.
    handles = [
        Patch(facecolor="0.62", label=results[0]["label_flat"]),
        Patch(facecolor="0.62", hatch="////", edgecolor="white",
              label=results[1]["label_flat"]),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=12, frameon=True, handlelength=1.9, handleheight=1.5,
              columnspacing=1.8, borderaxespad=0.0)

    ax.set_title(f"Q{FLOW_RP} NWM streamflow trigger", fontsize=15, pad=36)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}.{ext}")
        log.info("Saved s3://%s/%s%s.%s", bucket, prefix, OUTPUT_STEM, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    df = _read_parquet(bucket, f"{prefix}{INPUT_KEY}")
    if "thresh_src" not in df.columns:
        log.error("%s has no `thresh_src` column — re-run 08f_nwm_trigger_analysis.py "
                  "(4-scenario version) first.", INPUT_KEY)
        return

    results = []
    for label, source, thresh_src, tag in SCENARIOS:
        m = _metrics(df, source, thresh_src)
        if m is None:
            log.error("No rows for scenario %s (source=%s, thresh_src=%s, flow_rp=%d) — "
                      "is 08f's 4-scenario output present?", label, source, thresh_src, FLOW_RP)
            return
        m.update(label=label, tag=tag, label_flat=label.replace("\n", " / "))
        results.append(m)
        log.info("%-22s POD=%.3f FAR=%.3f CSI=%.3f FAF=%.2f/stn-yr (%d events, %d stns)",
                 label.replace("\n", " "), m["POD"], m["FAR"], m["CSI"], m["FAF"],
                 m["n_events"], m["n_stations"])

    make_figure(results, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
