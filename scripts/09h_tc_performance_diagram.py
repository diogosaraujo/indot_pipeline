"""09h_tc_performance_diagram.py

Combined performance (Roebber) diagrams comparing the three Tc-fixed precipitation
triggers on the SAME 106 gauges:
    circles   = watershed-mean MRMS      (08g, event_confusion_matrix_tc_areal)
    triangles = station gauge (ISD/GHCNh) (08h, event_confusion_matrix_tc_station)
    stars     = nearest-pixel MRMS       (08c, event_confusion_matrix_tc)

Four figures, each 6.5 x 6.5 in, three panels in one row (Q10 / Q50 / Q100), CSI
shading + frequency-bias lines behind, markers coloured by precipitation ARI. The
success-ratio axis is zoomed to 0–0.4 to separate the (clustered) points. Below
the panels the bottom band holds the two horizontal colorbars (ARI, CSI) on the
left and the symbol legend on the right:
    tc_performance_3source            — all three sources
    tc_performance_watershed_station  — watershed + station
    tc_performance_station_nearest    — station + nearest-pixel
    tc_performance_watershed_nearest  — watershed + nearest-pixel

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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09h_tc")

FLOW_RPS   = [10, 50, 100]
EPS        = 1e-9
SR_MAX     = 0.3       # success-ratio axis zoom (points cluster at low SR)
CSI_MAX    = 0.3       # max CSI reachable within the shown box (SR=0.3, POD=1)
BIAS       = [0.3, 0.5, 1, 1.5, 2, 3, 5]
CSI_LEVELS = np.arange(0.05, 0.301, 0.05)      # CSI contour lines within the shown range

# 10 discrete, distinct colours for the precipitation ARI thresholds.
ARIS       = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
ARI_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#000000"]
ARI_IDX    = {a: i for i, a in enumerate(ARIS)}

# source key → parquet, default marker, default legend label
SOURCES = {
    "watershed":       {"key": "analysis/event_confusion_matrix_tc_areal.parquet",
                        "marker": "o", "label": "Watershed-mean MRMS"},
    "station_nearest": {"key": "analysis/event_confusion_matrix_tc_station.parquet",
                        "marker": "^", "label": "Station gauge"},
    "nearest":         {"key": "analysis/event_confusion_matrix_tc.parquet",
                        "marker": "*", "label": "Nearest-pixel MRMS"},
}


def marker_size(m):
    return {"*": 130, "^": 52, "o": 42}.get(m, 46)


# Per-figure config; optional `markers`/`labels`/`ari_label` override the defaults.
FIGS = [
    dict(order=["watershed", "station_nearest", "nearest"],
         stem="analysis/figures/tc_performance_3source"),
    dict(order=["watershed", "station_nearest"],
         stem="analysis/figures/tc_performance_watershed_station"),
    dict(order=["station_nearest", "nearest"],
         stem="analysis/figures/tc_performance_station_nearest",
         markers={"nearest": "o"},                        # circles for MRMS, not stars
         labels={"station_nearest": "Precipitation Station"},
         ari_label="Precipitation ARI"),
    dict(order=["watershed", "nearest"],
         stem="analysis/figures/tc_performance_watershed_nearest"),
]


def _read(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def draw_background(ax):
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    # CSI shading + contours only up to CSI_MAX (the most reachable in the box).
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, CSI_MAX + 1e-9, 0.05),
                     cmap="Blues", alpha=0.75)
    cs = ax.contour(SR, POD, CSI, levels=CSI_LEVELS, colors="0.45", linewidths=0.4)
    for b in BIAS:                                           # frequency-bias lines
        xx = np.linspace(0, 1, 60)
        ax.plot(xx, np.minimum(b * xx, 1), ls="--", color="0.5", lw=0.5)
    return cf, cs


def label_bias(ax):
    """Label each frequency-bias line with its value at the line end, just INSIDE
    the panel (works on every panel). Labels over the dark CSI shading (b >= 1.5)
    use a white bold font; the lighter ones keep the grey font."""
    for b in BIAS:
        white = b >= 1.5
        color, weight = ("white", "bold") if white else ("0.35", "normal")
        if b * SR_MAX <= 1.0:                                # exits the right edge
            ax.text(SR_MAX * 0.98, b * SR_MAX, f"{b:g}", fontsize=7, color=color,
                    fontweight=weight, ha="right", va="center", zorder=5)
        else:                                                # exits the top → placed lower along its
            ax.text(0.75 / b, 0.75, f"{b:g}", fontsize=7, color=color,        # line to clear the
                    fontweight=weight, ha="right", va="center", zorder=5)     # CSI 0.20 label


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


def make_combined(loaded, fig_cfg, bucket, prefix):
    order     = fig_cfg["order"]
    out_stem  = fig_cfg["stem"]
    markers   = fig_cfg.get("markers", {})                  # per-source marker overrides
    labels    = fig_cfg.get("labels", {})                   # per-source legend-label overrides
    ari_label = fig_cfg.get("ari_label", "ARI (yr)")
    mk = lambda s: markers.get(s, SOURCES[s]["marker"])

    fig = plt.figure(figsize=(6.5, 6.5))
    # Panels fill the upper area; the bottom band holds the ARI colorbar (full
    # width) over a row with the CSI colorbar (left) beside the symbol legend.
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.98, top=0.955, bottom=0.30, wspace=0.13)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    cf = None
    for ax, flow_rp in zip(axes, FLOW_RPS):
        cf, cs = draw_background(ax)
        for t in ax.clabel(cs, fmt="%.2f", fontsize=7, inline=True, inline_spacing=2):
            try:
                lvl = float(t.get_text())
            except ValueError:
                lvl = 0.0
            if lvl >= 0.15:                              # over the dark blue → white bold
                t.set_color("white")
                t.set_fontweight("bold")
        for s in order:
            df = loaded.get(s)
            if df is None:
                continue
            sub = df[df["flow_rp_yr"] == flow_rp]
            if sub.empty:
                continue
            rps, sr, pod = pod_sr(sub)
            colors = [ARI_COLORS[ARI_IDX[int(round(r))]] for r in rps]
            ax.plot(sr, pod, color="black", lw=1.4, zorder=5)
            ax.scatter(sr, pod, c=colors, marker=mk(s), s=marker_size(mk(s)),
                       edgecolor="k", linewidth=0.4, zorder=6)
        label_bias(ax)                                       # bias values on EVERY panel
        ax.set_xlim(0, SR_MAX)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        # Drop the rightmost "0.3" except on the last panel so it doesn't collide
        # with the next panel's "0" when packed tight.
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3" if ax is axes[-1] else ""])
        ax.set_title(f"Q{flow_rp}", fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    # Shared axis labels — x-label sits clearly BELOW the tick labels.
    fig.text(0.53, 0.255, "Success ratio  (1 − FAR)", ha="center", fontsize=8)
    fig.text(0.020, 0.63, "Probability of detection (POD)", rotation=90, va="center", fontsize=8)

    # ── ARI discrete colorbar (full width) ────────────────────────────────────
    cax_ari = fig.add_axes([0.17, 0.155, 0.70, 0.028])
    sm = cm.ScalarMappable(cmap=ListedColormap(ARI_COLORS),
                           norm=BoundaryNorm(np.arange(len(ARIS) + 1), len(ARIS)))
    sm.set_array([])
    cb_ari = fig.colorbar(sm, cax=cax_ari, orientation="horizontal",
                          ticks=np.arange(len(ARIS)) + 0.5)
    cb_ari.ax.set_xticklabels([str(a) for a in ARIS])
    cb_ari.ax.tick_params(labelsize=7, length=0)
    fig.text(0.16, 0.169, ari_label, ha="right", va="center", fontsize=7)

    # ── CSI colorbar (left) beside the symbol legend (right) ──────────────────
    cax_csi = fig.add_axes([0.17, 0.065, 0.34, 0.028])
    cb_csi = fig.colorbar(cf, cax=cax_csi, orientation="horizontal")
    cb_csi.ax.tick_params(labelsize=7)
    fig.text(0.16, 0.079, "CSI", ha="right", va="center", fontsize=7)

    handles = [Line2D([0], [0], marker=mk(s), color="0.3", ls="", ms=7, mec="k", mew=0.4,
                      label=labels.get(s, SOURCES[s]["label"])) for s in order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.58, 0.08),
               ncol=1, fontsize=7, frameon=False, handletextpad=0.3, labelspacing=0.6)
    fig.text(0.985, 0.015, "dashed = frequency bias", ha="right", fontsize=7,
             color="0.4", style="italic")

    for ext in ("png", "pdf"):                               # exact 6.5x6.5 (no bbox_tight)
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
    loaded = load_sources(bucket, prefix, ["watershed", "station_nearest", "nearest"])
    for fig_cfg in FIGS:
        make_combined(loaded, fig_cfg, bucket, prefix)


if __name__ == "__main__":
    main()
