"""09h_tc_performance_diagram.py

Performance (Roebber) diagram of the fixed-Tc trigger (08c), one figure per flood
target, with the precipitation ARI thresholds as a connected, labelled path — an
alternative to the 09a bar charts for the same pooled skill.

Background: CSI shading + frequency-bias lines.  Each point is one ARI threshold
(pooled TP/FP/FN over all stations, MRMS nearest):
    x = success ratio  SR = TP/(TP+FP) = 1 - FAR
    y = detection      POD = TP/(TP+FN)
The path runs from loose ARI (top-left: high POD, many false alarms) to strict
ARI (bottom-right).  NOTE: a performance diagram shows the false-alarm RATIO (via
SR), not the per-year FAF rate the bar charts carry.

Output:
    s3://<bucket>/<prefix>analysis/figures/tc_performance_Q{10,50,100}.{png,svg}

Usage:
    python scripts/09h_tc_performance_diagram.py
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09h_tc")

SOURCE = "nearest"
FLOW_RPS = [10, 50, 100]
EPS = 1e-9
INPUT_KEY = "analysis/event_confusion_matrix_tc.parquet"
OUTPUT_STEM = "analysis/figures/tc_performance_Q"


def _read_parquet(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def draw_background(ax):
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    cl = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=8, inline=True)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:
        xx = np.linspace(0, 1, 10)
        ax.plot(xx, np.minimum(b * xx, 1), ls="--", color="0.5", lw=0.7)
        if b <= 1:
            ax.text(1.008, b, f"{b:g}", fontsize=7.5, color="0.4", va="center")
        else:
            ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7.5, color="0.4", ha="center")
    return cf


def make_figure(flow_rp, sub, precip_rps, bucket, prefix):
    pooled = (sub.groupby("precip_rp_yr")[["tp", "fp", "fn"]].sum().reindex(precip_rps).fillna(0.0))
    tp, fp, fn = pooled["tp"].to_numpy(), pooled["fp"].to_numpy(), pooled["fn"].to_numpy()
    sr = tp / (tp + fp + EPS)
    pod = tp / (tp + fn + EPS)
    n_events = int(sub.groupby("site_no")["n_flow_events"].first().sum())
    n_stations = int(sub["site_no"].nunique())

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    cf = draw_background(ax)
    ax.plot(sr, pod, color="0.35", lw=1.2, zorder=5)
    sc = ax.scatter(sr, pod, c=precip_rps, cmap="plasma", norm=LogNorm(),
                    s=95, edgecolor="k", linewidth=0.6, zorder=6)
    for rp, sx, sy in zip(precip_rps, sr, pod):
        ax.annotate(f"P{int(rp)}", (sx, sy), textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color="0.15", zorder=7)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Success ratio  (1 − FAR)", fontsize=13)
    ax.set_ylabel("Probability of detection (POD)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.text(0.985, 0.03, "dashed = frequency bias", fontsize=8, color="0.4",
            ha="right", style="italic")
    ax.set_title(f"Q{flow_rp} flood target  —  {n_events} events across "
                 f"{n_stations} stations (MRMS nearest)", fontsize=14)
    cb = plt.colorbar(sc, ax=ax, label="Precipitation ARI (yr)", shrink=0.85, pad=0.02)
    cb.set_label("Precipitation ARI (yr)", fontsize=12)
    plt.colorbar(cf, ax=ax, label="CSI", shrink=0.85, pad=0.09)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=160, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_STEM}{flow_rp}.{ext}")
        log.info("Saved s3://%s/%s%s%d.%s", bucket, prefix, OUTPUT_STEM, flow_rp, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    df = _read_parquet(bucket, f"{prefix}{INPUT_KEY}",
                       ["site_no", "source", "precip_rp_yr", "flow_rp_yr",
                        "tp", "fp", "fn", "n_flow_events"])
    df = df[df["source"] == SOURCE].copy()
    if df.empty:
        log.error("No %s rows in %s", SOURCE, INPUT_KEY)
        return
    precip_rps = sorted(df["precip_rp_yr"].unique())
    for flow_rp in FLOW_RPS:
        sub = df[df["flow_rp_yr"] == flow_rp]
        if not sub.empty:
            make_figure(flow_rp, sub, precip_rps, bucket, prefix)


if __name__ == "__main__":
    main()
