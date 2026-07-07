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

# (x-axis label, source, thresh_src, short tag)
SCENARIOS = [
    ("A&A\nUSGS-Q (04b)",       "nwm_analysis_assim", "usgs_peak_04b", "gauged"),
    ("Open-loop\nNWM-Q (04c)",  "nwm_open_loop",      "nwm_retro_04c", "gauge-free"),
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
    names = [r["label"] for r in results]
    x = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax2 = ax.twinx()

    # left-axis skill bars: POD, FAR, CSI
    for k, (metric, color) in enumerate(LEFT_METRICS):
        off = (k - 1.5) * width
        vals = [r[metric] for r in results]
        ax.bar(x + off, vals, width, color=color)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=12, rotation=90)

    # right-axis FAF bar (log), drawn from a small floor so bars are visible
    faf = np.array([r["FAF"] for r in results])
    pos = faf[faf > 0]
    floor = max(pos.min() / 5.0, 1e-3) if pos.size else 1e-3
    ax2.set_yscale("log")
    off_faf = 1.5 * width
    ax2.bar(x + off_faf, np.where(faf > 0, faf, floor) - floor, width, bottom=floor,
            color=FAF_COLOR)
    for xi, v in zip(x + off_faf, faf):
        if v > 0:
            ax2.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=11, color=FAF_COLOR, rotation=90)
    ax2.set_ylim(floor, (pos.max() * 4 if pos.size else 1))

    ax.set_ylim(0, 1.18)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("Skill score", fontsize=14)
    ax2.set_ylabel("FAF (false alarms / station-yr)", fontsize=13, color=FAF_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12, colors=FAF_COLOR)
    ax.grid(axis="y", ls=":", alpha=0.4)

    handles = [Patch(color=c, label=m) for m, c in LEFT_METRICS] + [Patch(color=FAF_COLOR, label="FAF")]
    ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=12, framealpha=0.9)
    sub = "  |  ".join(f"{r['tag']}: {r['n_events']} events / {r['n_stations']} stns" for r in results)
    ax.set_title(f"Q10 flood target — NWM streamflow trigger (truth: USGS ≥ Q10 04b)\n{sub}",
                 fontsize=13)
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
        m.update(label=label, tag=tag)
        results.append(m)
        log.info("%-22s POD=%.3f FAR=%.3f CSI=%.3f FAF=%.2f/stn-yr (%d events, %d stns)",
                 label.replace("\n", " "), m["POD"], m["FAR"], m["CSI"], m["FAF"],
                 m["n_events"], m["n_stations"])

    make_figure(results, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
