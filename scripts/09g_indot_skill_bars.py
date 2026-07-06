"""09g_indot_skill_bars.py

Skill-metric bar charts for the CURRENT INDOT trigger (08d output).

One figure per flood target (Q10, Q50, Q100).  Each shows four bars:
    POD, FAR, CSI  → left y-axis (bounded skill scores)
    FAF            → right y-axis (False Alarm Frequency, false alarms per
                     station-year — an unbounded rate)
Counts are pooled GLOBALLY across all stations, then scored once per target.

    POD = TP / (TP+FN)      FAR = FP / (TP+FP)      CSI = TP / (TP+FP+FN)
    FAF = Σ FP / Σ (n_common_hours / 8766)      [false alarms per station-year]

Reads:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_indot.parquet
Writes:
    s3://<bucket>/<prefix>analysis/figures/indot_skill_Q{10,50,100}.{png,svg}

Usage:
    python scripts/09g_indot_skill_bars.py
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09g_indot")

FLOW_RPS = [10, 50, 100]
EPS = 1e-9
HOURS_PER_YEAR = 8766.0
INPUT_KEY = "analysis/event_confusion_matrix_indot.parquet"
OUTPUT_STEM = "analysis/figures/indot_skill_Q"

# (label, color) — POD/FAR/CSI on the left axis, FAF on the right axis.
LEFT_METRICS = [("POD", "#2c7fb8"), ("FAR", "#d95f0e"), ("CSI", "#31a354")]
FAF_COLOR = "#6a51a3"


def _read_parquet(bucket: str, key: str, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def make_figure(rp: int, pod: float, far: float, csi: float, faf: float,
                bucket: str, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    x = [0, 1, 2, 3]
    left_vals = [pod, far, csi]
    width = 0.62

    # left-axis bars
    for xi, v, (_, color) in zip(x[:3], left_vals, LEFT_METRICS):
        ax.bar(xi, v, width, color=color)
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=17)

    # right-axis bar (FAF)
    ax2.bar(x[3], faf, width, color=FAF_COLOR)
    ax2.text(x[3], faf, f"{faf:.2f}", ha="center", va="bottom", fontsize=17)

    top = max(left_vals) if max(left_vals) > 0 else 1.0
    ax.set_ylim(0, top * 1.22)
    ax2.set_ylim(0, (faf if faf > 0 else 1.0) * 1.30)

    ax.set_xticks(x)
    ax.set_xticklabels(["POD", "FAR", "CSI", "FAF"], fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.set_ylabel("FAF (per station-yr)", fontsize=16)

    ax.set_title(f"Skill Metrics of existing INDOT methods to trigger Q{rp}",
                 fontsize=17)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}{rp}.{ext}")
        log.info("Saved s3://%s/%s%s%d.%s", bucket, prefix, OUTPUT_STEM, rp, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    df = _read_parquet(bucket, f"{prefix}{INPUT_KEY}",
                       ["site_no", "flow_rp_yr", "tp", "fp", "fn", "n_common_hours"])

    for rp in FLOW_RPS:
        sub = df[df["flow_rp_yr"] == rp]
        if sub.empty:
            log.warning("No rows for Q%d — skipping", rp)
            continue
        tp, fp, fn = sub["tp"].sum(), sub["fp"].sum(), sub["fn"].sum()
        pod = tp / (tp + fn + EPS)
        far = fp / (tp + fp + EPS)
        csi = tp / (tp + fp + fn + EPS)
        station_years = (sub["n_common_hours"] / HOURS_PER_YEAR).sum()
        faf = fp / (station_years + EPS)          # false alarms per station-year
        log.info("Q%d: POD=%.2f FAR=%.2f CSI=%.2f FAF=%.2f (n=%d stations)",
                 rp, pod, far, csi, faf, sub["site_no"].nunique())
        make_figure(rp, pod, far, csi, faf, bucket, prefix)


if __name__ == "__main__":
    main()
