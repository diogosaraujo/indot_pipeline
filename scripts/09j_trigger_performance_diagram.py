"""09j_trigger_performance_diagram.py

Combined Roebber performance diagram comparing every flood-inspection trigger family
on one plot.  Eight operating points:

  Stations — INDOT current rule (nearest ISD/GHCNh gauge, 24-h >= 2.5 in)   Q10, Q50, Q100
             from analysis/indot_trigger_metrics.csv (08d)
  MRMS     — nearest-pixel Atlas-14 trigger, ARI matched to the flood RP
             (P10->Q10, P50->Q50, P100->Q100), source='nearest'               Q10, Q50, Q100
             pooled from analysis/event_confusion_matrix_tc.parquet (08c)
  NWM      — operational streamflow trigger, Q10 only:                          Q10 (x2)
               A&A  vs USGS-Q (04b)      [gauged best case]
               Open-loop vs NWM-Q (04c)  [gauge-free]
             from analysis/event_confusion_matrix_nwm.parquet (08f, 4-scenario)

Encoding (so a trigger is identifiable at a glance):
    symbol  = data source group   (square = stations, circle = MRMS, star = NWM)
    colour  = flood return period  (green = Q10, orange = Q50, purple = Q100)
    the two NWM Q10 stars (both green) are split by fill: A&A filled, Open-loop open.

POD = TP/(TP+FN)   SR = TP/(TP+FP) = 1 - FAR   CSI = TP/(TP+FP+FN)   (all pooled first, then scored).

Writes:
    s3://<bucket>/<prefix>analysis/figures/combined_performance_diagram.{png,svg}

Usage:
    python scripts/09j_trigger_performance_diagram.py
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
from matplotlib.lines import Line2D

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09j_perf")

FLOW_RPS   = [10, 50, 100]
INDOT_KEY  = "analysis/indot_trigger_metrics.csv"
TC_KEY     = "analysis/event_confusion_matrix_tc.parquet"
NWM_KEY    = "analysis/event_confusion_matrix_nwm.parquet"
FIG_KEY    = "analysis/figures/combined_performance_diagram"

# symbol = group; colour = flood return period
GROUP_MARKER = {"stations": ("s", 150), "mrms": ("o", 150), "nwm": ("*", 460)}
Q_COLOR      = {10: "#1a9850", 50: "#f46d43", 100: "#762a83"}
# NWM Q10 scenarios (label, source, thresh_src, filled)
NWM_SCENARIOS = [
    ("A&A",       "nwm_analysis_assim", "usgs_peak_04b", True),
    ("Open-loop", "nwm_open_loop",      "nwm_retro_04c", False),
]


def _get(key: str) -> bytes:
    return s3_client().get_object(Bucket=_BUCKET, Key=f"{_PREFIX}{key}")["Body"].read()


def _read_parquet(key: str, columns=None) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(_get(key)), columns=columns).to_pandas()


def skill(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    sr  = tp / (tp + fp) if (tp + fp) else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    return pod, sr, csi


# ── Collect the operating points ──────────────────────────────────────────────

def collect_points() -> list[dict]:
    pts: list[dict] = []

    # Stations — INDOT current method (per flow RP)
    idf = pd.read_csv(io.BytesIO(_get(INDOT_KEY)))
    for rp in FLOW_RPS:
        r = idf[idf["flow_rp_yr"] == rp]
        if r.empty:
            continue
        r = r.iloc[0]
        pod, sr, csi = skill(float(r["tp"]), float(r["fp"]), float(r["fn"]))
        pts.append(dict(group="stations", rp=rp, pod=pod, sr=sr, csi=csi, filled=True, label=None))

    # MRMS — Atlas-14 matched P=Q, nearest pixel (pool across stations)
    tc = _read_parquet(TC_KEY, columns=["source", "precip_rp_yr", "flow_rp_yr", "tp", "fp", "fn"])
    tc = tc[tc["source"] == "nearest"]
    for rp in FLOW_RPS:
        sub = tc[(tc["precip_rp_yr"] == rp) & (tc["flow_rp_yr"] == rp)]
        if sub.empty:
            continue
        pod, sr, csi = skill(sub["tp"].sum(), sub["fp"].sum(), sub["fn"].sum())
        pts.append(dict(group="mrms", rp=rp, pod=pod, sr=sr, csi=csi, filled=True, label=None))

    # NWM — Q10 only (A&A / Open-loop)
    nwm = _read_parquet(NWM_KEY)
    if "thresh_src" not in nwm.columns:
        log.warning("%s lacks `thresh_src` — re-run the 4-scenario 08f. Skipping NWM points.", NWM_KEY)
    else:
        for label, src, thr, filled in NWM_SCENARIOS:
            sub = nwm[(nwm["source"] == src) & (nwm["thresh_src"] == thr) & (nwm["flow_rp_yr"] == 10)]
            if sub.empty:
                log.warning("No NWM rows for %s/%s Q10 — skipping.", src, thr)
                continue
            pod, sr, csi = skill(sub["tp"].sum(), sub["fp"].sum(), sub["fn"].sum())
            pts.append(dict(group="nwm", rp=10, pod=pod, sr=sr, csi=csi, filled=filled, label=label))

    for p in pts:
        log.info("  %-9s Q%-4d  POD=%.3f SR=%.3f CSI=%.3f  %s",
                 p["group"], p["rp"], p["pod"], p["sr"], p["csi"], p["label"] or "")
    return pts


# ── Diagram ────────────────────────────────────────────────────────────────────

def performance_diagram(pts: list[dict], bucket: str, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # CSI shading + contours
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    cl = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=9, inline=True)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:                 # frequency-bias lines POD = b·SR
        x = np.linspace(0, 1, 10)
        ax.plot(x, np.minimum(b * x, 1), ls="--", color="0.5", lw=0.7)
        if b <= 1: ax.text(1.008, b, f"{b:g}", fontsize=7.5, color="0.4", va="center")
        else:      ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7.5, color="0.4", ha="center")

    # Operating points: symbol = group, colour = RP, fill splits the two NWM stars
    for p in pts:
        if not (np.isfinite(p["sr"]) and np.isfinite(p["pod"])):
            continue
        mk, ms = GROUP_MARKER[p["group"]]
        color = Q_COLOR[p["rp"]]
        if p["filled"]:
            ax.scatter([p["sr"]], [p["pod"]], marker=mk, s=ms, color=color,
                       edgecolors="black", lw=1.0, zorder=7)
        else:                                              # open marker (Open-loop)
            ax.scatter([p["sr"]], [p["pod"]], marker=mk, s=ms, facecolors="white",
                       edgecolors=color, lw=2.0, zorder=7)
        if p["label"]:                                     # tag the NWM stars
            ax.annotate(p["label"], (p["sr"], p["pod"]), textcoords="offset points",
                        xytext=(9, -2), fontsize=12, fontweight="bold", color=color)

    # ── Legends ────────────────────────────────────────────────────────────────
    group_handles = [
        Line2D([], [], marker="s", ls="", mfc="0.55", mec="black", ms=11,
               label="Stations — INDOT 24-h ≥ 2.5 in"),
        Line2D([], [], marker="o", ls="", mfc="0.55", mec="black", ms=11,
               label="MRMS — Atlas-14 (P = Q)"),
        Line2D([], [], marker="*", ls="", mfc="0.55", mec="black", ms=16,
               label="NWM streamflow (Q10)"),
    ]
    q_handles = [Line2D([], [], marker="s", ls="", mfc=c, mec="black", ms=11, label=f"Q{rp}")
                 for rp, c in Q_COLOR.items()]
    leg1 = ax.legend(handles=group_handles, loc="upper right", fontsize=11, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=q_handles, loc="lower right", fontsize=11, framealpha=0.95)
    ax.text(0.985, 0.52, "NWM: filled = A&A · open = Open-loop", transform=ax.transAxes,
            fontsize=10, color="0.25", ha="right", style="italic")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Success ratio  (1 − FAR)", fontsize=14)
    ax.set_ylabel("Probability of detection (POD)", fontsize=14)
    ax.set_title("Flood-inspection trigger performance",
                 fontsize=14, fontweight="bold", pad=12)
    cb = plt.colorbar(cf, ax=ax, label="Critical success index (CSI)", shrink=0.85)
    cb.set_label("Critical success index (CSI)", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=190, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────

_BUCKET = _PREFIX = None


def main() -> None:
    global _BUCKET, _PREFIX
    cfg = load_config()
    _BUCKET, _PREFIX = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    pts = collect_points()
    if not pts:
        log.error("No operating points collected — check the input tables.")
        return
    performance_diagram(pts, _BUCKET, _PREFIX)
    log.info("Done — %d operating points plotted.", len(pts))


if __name__ == "__main__":
    main()
