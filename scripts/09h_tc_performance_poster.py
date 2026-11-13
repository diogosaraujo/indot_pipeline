"""09h_tc_performance_poster.py

Poster version of the 09h `tc_performance_station_nearest` performance diagram:
Precipitation Station (triangles) vs nearest-pixel MRMS (circles), Q10 / Q50 /
Q100 panels, CSI shading + frequency-bias lines, markers coloured by
precipitation ARI.

Layout: 14.9 x 6.4 in, Times New Roman, 600 dpi. Text sizes 16 pt (ticks,
contour / bias labels), 18 pt (axis labels, colorbar titles, legend), 24 pt
(panel headings). Colorbars and the symbol legend sit in a column to the right of
the panels so the panels keep the full figure height.

ATLAS-14 LABEL CORRECTION (default ON). The 08c / 08h thresholds came from the
Atlas-14 table that 07_extract_atlas14.py mislabelled by one duration (see
07d_relabel_atlas14_durations.py). The pooled TP/FP/FN, and so every point's
SR / POD, are correct: each row was counted against the depth it actually used.
Only the ARI NAME on each threshold is wrong. Both sources share the gauge-site
Atlas-14 depths, so the same relabel applies to both. EFFECTIVE_ARI is the median
over the 106 gauges of the true ARI of each stored depth at the gauge's
accumulation duration, read off the 07d-relabelled table:
    nominal  1    2    5    10   25   50   100  200  500  1000
    true    <1   1.2  2.9  5.7  14   27   54   110  290  580
(p10-p90 spread across gauges is about +-35 %). Pass --nominal to draw the old
labels. The script decides automatically (utils.atlas14_label_state): while the
Atlas-14 tables are still buggy it applies EFFECTIVE_ARI; once 07d has been run
and 08c / 08h re-run with --fresh, the sweep sits on the true ARIs and the plain
1..1000 labels are drawn; if the tables are fixed but the matrices are stale it
refuses.

Writes results/tc_performance_station_nearest_poster.{png,pdf}

Usage:
    python scripts/09h_tc_performance_poster.py
    python scripts/09h_tc_performance_poster.py --nominal
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from utils import atlas14_label_state, load_config

_spec = importlib.util.spec_from_file_location(
    "p09h", Path(__file__).with_name("09h_tc_performance_diagram.py"))
p09h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p09h)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09h_poster")

FIG_W, FIG_H, DPI = 14.9, 6.4, 600
FS_S, FS_M, FS_L = 16, 18, 24
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
STEM = "tc_performance_station_nearest_poster"

ORDER   = ["station_nearest", "nearest"]
MARKERS = {"station_nearest": "^", "nearest": "o"}
LABELS  = {"station_nearest": "Precipitation Station", "nearest": "Nearest-pixel MRMS"}
MSIZE   = {"^": 230, "o": 190}                 # scatter area (pt^2); 09h used 52 / 42

# median true ARI behind each nominal 08c/08h label (see docstring)
EFFECTIVE_ARI = {1: "<1", 2: "1.2", 5: "2.9", 10: "5.7", 25: "14", 50: "27",
                 100: "54", 200: "110", 500: "290", 1000: "580"}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": FS_S,
})


def label_bias(ax):
    """09h.label_bias at poster size."""
    for b in p09h.BIAS:
        dark = b >= 1.5 and b * p09h.SR_MAX <= 1.0           # right-edge labels on dark CSI
        color, weight = ("white", "bold") if dark else ("0.30", "normal")
        if b * p09h.SR_MAX <= 1.0:
            ax.text(p09h.SR_MAX * 0.98, b * p09h.SR_MAX, f"{b:g}", fontsize=FS_S,
                    color=color, fontweight=weight, ha="right", va="center", zorder=5)
        else:                          # exits the top: sit lower on the line, between
            ax.text(0.62 / b, 0.62, f"{b:g}", fontsize=FS_S, color=color,  # CSI labels
                    fontweight=weight, ha="right", va="center", zorder=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", action="store_true",
                    help="label the ARI colorbar with the stored (mislabelled) ARIs")
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    state = atlas14_label_state(bucket, prefix, [p09h.SOURCES[s]["key"] for s in ORDER])
    relabel = state == "buggy" and not args.nominal
    log.info("Atlas-14 tables: %s -> %s ARI labels", state,
             "corrected" if relabel else "stored")
    loaded = p09h.load_sources(bucket, prefix, ORDER)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 3, left=0.063, right=0.745, top=0.90, bottom=0.135,
                          wspace=0.07)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    cf = None
    for ax, flow_rp in zip(axes, p09h.FLOW_RPS):
        g = np.linspace(0.001, 1, 400)
        SR, POD = np.meshgrid(g, g)
        CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
        cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, p09h.CSI_MAX + 1e-9, 0.05),
                         cmap="Blues", alpha=0.75)
        cs = ax.contour(SR, POD, CSI, levels=p09h.CSI_LEVELS, colors="0.45", linewidths=0.8)
        for b in p09h.BIAS:
            xx = np.linspace(0, 1, 60)
            ax.plot(xx, np.minimum(b * xx, 1), ls="--", color="0.5", lw=0.9)
        for t in ax.clabel(cs, fmt="%.2f", fontsize=FS_S, inline=True, inline_spacing=3):
            try:
                lvl = float(t.get_text())
            except ValueError:
                lvl = 0.0
            if lvl >= 0.15:
                t.set_color("white")
                t.set_fontweight("bold")

        for s in ORDER:
            df = loaded.get(s)
            if df is None:
                continue
            sub = df[df["flow_rp_yr"] == flow_rp]
            if sub.empty:
                continue
            rps, sr, pod = p09h.pod_sr(sub)
            colors = [p09h.ARI_COLORS[p09h.ARI_IDX[int(round(r))]] for r in rps]
            ax.plot(sr, pod, color="black", lw=2.6, zorder=5)
            ax.scatter(sr, pod, c=colors, marker=MARKERS[s], s=MSIZE[MARKERS[s]],
                       edgecolor="k", linewidth=0.9, zorder=6, clip_on=False)

        label_bias(ax)
        ax.set_xlim(0, p09h.SR_MAX)
        ax.set_ylim(0, 1)
        ax.set_facecolor("white")                            # panels stay opaque
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3" if ax is axes[-1] else ""])
        ax.set_title(f"Q{flow_rp}", fontsize=FS_L, pad=8)
        ax.tick_params(labelsize=FS_S, length=5, width=0.9)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    fig.text(0.404, 0.035, "Success ratio  (1 − FAR)", ha="center", fontsize=FS_M)
    fig.text(0.010, 0.517, "Probability of detection (POD)", rotation=90,
             ha="left", va="center", fontsize=FS_M)

    # ── right column: symbol legend on top, ARI + CSI colorbars below ──────────
    handles = [Line2D([0], [0], marker=MARKERS[s], color="0.35", ls="", ms=15,
                      mec="k", mew=0.9, label=LABELS[s]) for s in ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.765, 0.935),
               fontsize=FS_M, frameon=False, handletextpad=0.4, labelspacing=0.5,
               borderaxespad=0)

    cb_bottom, cb_h = 0.135, 0.535
    cax_ari = fig.add_axes([0.785, cb_bottom, 0.018, cb_h])
    sm = cm.ScalarMappable(cmap=ListedColormap(p09h.ARI_COLORS),
                           norm=BoundaryNorm(np.arange(len(p09h.ARIS) + 1), len(p09h.ARIS)))
    sm.set_array([])
    cb_ari = fig.colorbar(sm, cax=cax_ari, orientation="vertical",
                          ticks=np.arange(len(p09h.ARIS)) + 0.5)
    tick_txt = [EFFECTIVE_ARI[a] for a in p09h.ARIS] if relabel \
        else [str(a) for a in p09h.ARIS]
    cb_ari.ax.set_yticklabels(tick_txt)
    cb_ari.ax.tick_params(labelsize=FS_S, length=0)
    fig.text(0.785, cb_bottom + cb_h + 0.035, "Precip.\nARI (yr)", ha="left",
             va="bottom", fontsize=FS_M, linespacing=1.0)

    cax_csi = fig.add_axes([0.895, cb_bottom, 0.018, cb_h])
    cb_csi = fig.colorbar(cf, cax=cax_csi, orientation="vertical")
    cb_csi.set_ticks(np.arange(0, 0.301, 0.05))
    cb_csi.ax.tick_params(labelsize=FS_S)
    fig.text(0.895, cb_bottom + cb_h + 0.035, "CSI", ha="left", va="bottom",
             fontsize=FS_M)

    fig.text(0.995, 0.012, "dashed = frequency bias", ha="right", fontsize=FS_S,
             color="0.35", style="italic")

    OUT_DIR.mkdir(exist_ok=True)
    stem = STEM + ("_nominal" if args.nominal and state == "buggy" else "")
    for ext in ("png", "pdf"):                               # exact size, no bbox_tight
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=DPI, facecolor="none")         # transparent outside the panels
        log.info("Saved %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
