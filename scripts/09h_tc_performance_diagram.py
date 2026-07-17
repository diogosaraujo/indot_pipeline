"""09h_tc_performance_diagram.py

Combined performance (Roebber) diagrams comparing the three Tc-fixed precipitation
triggers on the SAME 106 gauges:
    circles   = watershed-mean MRMS      (08g, event_confusion_matrix_tc_areal)
    triangles = station gauge (ISD/GHCNh) (08h, event_confusion_matrix_tc_station)
    stars     = nearest-pixel MRMS       (08c, event_confusion_matrix_tc)

Two figures, each 6.5 x 6.5 in, three panels in one row (Q10 / Q50 / Q100), CSI
shading + frequency-bias lines behind, markers coloured by precipitation ARI, with
two horizontal colorbars (ARI, CSI) below the panels:
    Figure 1  tc_performance_3source   — all three sources
    Figure 2  tc_performance_2source   — watershed + station only (no nearest-pixel)

Each point is one ARI threshold, pooled TP/FP/FN over all stations:
    x = success ratio  SR = TP/(TP+FP) = 1 - FAR
    y = detection      POD = TP/(TP+FN)

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
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09h_tc")

FLOW_RPS = [10, 50, 100]
EPS      = 1e-9
CMAP     = "plasma"
ARI_NORM = LogNorm(vmin=1, vmax=1000)

# source key → parquet, marker, marker size, legend label
SOURCES = {
    "watershed":       {"key": "analysis/event_confusion_matrix_tc_areal.parquet",
                        "marker": "o", "size": 26, "label": "Watershed-mean MRMS"},
    "station_nearest": {"key": "analysis/event_confusion_matrix_tc_station.parquet",
                        "marker": "^", "size": 32, "label": "Station gauge"},
    "nearest":         {"key": "analysis/event_confusion_matrix_tc.parquet",
                        "marker": "*", "size": 90, "label": "Nearest-pixel MRMS"},
}
FIG1_ORDER = ["watershed", "station_nearest", "nearest"]
FIG2_ORDER = ["watershed", "station_nearest"]
OUT1 = "analysis/figures/tc_performance_3source"
OUT2 = "analysis/figures/tc_performance_2source"


def _read(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def draw_background(ax):
    g = np.linspace(0.001, 1, 300)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.4)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:                    # frequency-bias lines
        xx = np.linspace(0, 1, 10)
        ax.plot(xx, np.minimum(b * xx, 1), ls="--", color="0.5", lw=0.5)
    return cf


def pod_sr(sub):
    """Pooled (ARI, SR, POD) for one source + flood target."""
    pooled = sub.groupby("precip_rp_yr")[["tp", "fp", "fn"]].sum().sort_index()
    rps = pooled.index.to_numpy(float)
    tp, fp, fn = (pooled[c].to_numpy(float) for c in ("tp", "fp", "fn"))
    return rps, tp / (tp + fp + EPS), tp / (tp + fn + EPS)


def load_sources(bucket, prefix, order):
    out = {}
    for s in order:
        key = SOURCES[s]["key"]
        try:
            df = _read(bucket, f"{prefix}{key}",
                       ["site_no", "source", "precip_rp_yr", "flow_rp_yr", "tp", "fp", "fn"])
            df = df[df["source"] == s]
            out[s] = df if not df.empty else None
            log.info("%s: %d rows", s, 0 if out[s] is None else len(df))
        except Exception as e:                               # noqa: BLE001
            log.warning("%s unavailable (%s): %s", s, key, e)
            out[s] = None
    return out


def make_combined(loaded, order, out_stem, bucket, prefix):
    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(1, 3, left=0.09, right=0.975, top=0.85, bottom=0.24, wspace=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    cf = None
    for ax, flow_rp in zip(axes, FLOW_RPS):
        cf = draw_background(ax)
        for s in order:
            df = loaded.get(s)
            if df is None:
                continue
            sub = df[df["flow_rp_yr"] == flow_rp]
            if sub.empty:
                continue
            rps, sr, pod = pod_sr(sub)
            ax.plot(sr, pod, color="0.45", lw=0.7, zorder=4)
            ax.scatter(sr, pod, c=rps, cmap=CMAP, norm=ARI_NORM, marker=SOURCES[s]["marker"],
                       s=SOURCES[s]["size"], edgecolor="k", linewidth=0.4, zorder=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Q{flow_rp}", fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    fig.text(0.53, 0.205, "Success ratio  (1 − FAR)", ha="center", fontsize=8)
    fig.text(0.022, 0.545, "Probability of detection (POD)", rotation=90, va="center", fontsize=8)

    # ── Two horizontal colorbars below the panels ─────────────────────────────
    cax_ari = fig.add_axes([0.17, 0.150, 0.68, 0.022])
    cax_csi = fig.add_axes([0.17, 0.070, 0.68, 0.022])
    sm = cm.ScalarMappable(norm=ARI_NORM, cmap=CMAP)
    sm.set_array([])
    cb_ari = fig.colorbar(sm, cax=cax_ari, orientation="horizontal", ticks=[1, 10, 100, 1000])
    cb_ari.ax.tick_params(labelsize=7)
    cb_ari.ax.set_xticklabels(["1", "10", "100", "1000"])
    fig.text(0.155, 0.161, "ARI (yr)", ha="right", va="center", fontsize=7)
    cb_csi = fig.colorbar(cf, cax=cax_csi, orientation="horizontal")
    cb_csi.ax.tick_params(labelsize=7)
    fig.text(0.155, 0.081, "CSI", ha="right", va="center", fontsize=7)

    # ── Shape legend (grey; shape = source, colour = ARI) ─────────────────────
    handles = [Line2D([0], [0], marker=SOURCES[s]["marker"], color="0.3", ls="",
                      ms=7, mec="k", mew=0.4, label=SOURCES[s]["label"]) for s in order]
    fig.legend(handles=handles, loc="upper center", ncol=len(order), fontsize=7,
               frameon=False, bbox_to_anchor=(0.53, 0.965),
               handletextpad=0.3, columnspacing=1.1)
    fig.text(0.975, 0.028, "dashed = frequency bias", ha="right", fontsize=7,
             color="0.4", style="italic")

    # Exact 6.5x6.5 (no bbox_inches='tight').
    for ext in ("png", "pdf"):
        buf = io.BytesIO()
        kw = {"format": ext}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(buf, **kw)
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{out_stem}.{ext}")
        log.info("Saved s3://%s/%s%s.%s", bucket, prefix, out_stem, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    loaded = load_sources(bucket, prefix, FIG1_ORDER)
    make_combined(loaded, FIG1_ORDER, OUT1, bucket, prefix)   # all three sources
    make_combined(loaded, FIG2_ORDER, OUT2, bucket, prefix)   # no nearest-pixel


if __name__ == "__main__":
    main()
