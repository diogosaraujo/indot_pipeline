"""09f_indot_trigger_schematic.py

Schematic four-panel explainer of the TP / FP / FN / TN outcomes for INDOT's
CURRENT flood-trigger procedure:

    trigger = nearest rain gauge, trailing 24-hour precipitation >= 2.5 in
    flood   = streamflow >= a flood threshold Q
    a rain trigger may lead the flood by up to 24 h

The storms and hydrographs are SYNTHETIC (idealized) — the figure shows what
each outcome means, not a specific gauge:

    TP  flood, and the 24-h rain crossed 2.5 in            (correct alarm)
    FP  24-h rain crossed 2.5 in, but no flood followed    (false alarm)
    FN  flood, but the 24-h rain never reached 2.5 in      (missed)
    TN  neither the rain nor the flow crossed              (correct silence)

Each panel is a hyetograph (hourly precip bars + trailing 24-h accumulation vs
the 2.5 in trigger) over a hydrograph (flow vs the flood threshold Q).

Writes (S3 only):
    s3://<bucket>/<prefix>analysis/figures/indot_trigger_cases.{png,svg}

Usage:
    python scripts/09f_indot_trigger_schematic.py
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import gamma as gdist

from utils import load_config, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("09f_schematic")

FIG_KEY = "analysis/figures/indot_trigger_cases"

PTHR = 2.5          # in / 24 h  — INDOT trigger
QTHR = 1000.0       # cfs        — schematic flood threshold (Q10)
BASE = 150.0        # cfs        — baseflow
H    = np.arange(0, 144)                 # 6 days, hourly
DAYS = H / 24.0

# total_in, peak_cfs per case — chosen so each outcome is unambiguous
CASES = {
    "TP": dict(total=3.2, peak=1360),    # rain crosses 2.5, flood follows
    "FP": dict(total=3.6, peak=560),     # rain crosses 2.5, no flood
    "FN": dict(total=1.5, peak=1320),    # flood, rain never reaches 2.5
    "TN": dict(total=1.2, peak=520),     # neither crosses
}
CLR = {"TP": "#2e7d32", "FP": "#ef6c00", "FN": "#c62828", "TN": "#455a64"}
DEF = {"TP": "flood + trigger fired",
       "FP": "trigger fired, no flood",
       "FN": "flood, trigger missed",
       "TN": "quiet, trigger silent"}


def storm(total_in: float, center_h: float = 30, k: float = 2.6, scale: float = 3.0) -> np.ndarray:
    """Hourly hyetograph (in) whose trailing-24-h accumulation peaks at total_in.
    Concentrated enough that the individual hourly bars stay legible."""
    x = H - (center_h - k * scale)
    p = np.where(x > 0, gdist.pdf(x / scale, k), 0.0)
    roll = np.convolve(p, np.ones(24), "full")[:len(p)]
    return p * (total_in / roll.max()) if roll.max() > 0 else p


def hydrograph(peak_cfs: float, center_h: float = 64, k: float = 4.5, scale: float = 6.0) -> np.ndarray:
    """Idealized single-peak hydrograph (cfs) rising after the rain."""
    x = H - (center_h - k * scale)
    r = np.where(x > 0, gdist.pdf(x / scale, k), 0.0)
    r = r / r.max() * (peak_cfs - BASE) if r.max() > 0 else r
    return BASE + r


def accum24(p: np.ndarray) -> np.ndarray:
    return np.convolve(p, np.ones(24), "full")[:len(p)]


def draw(fig, outer, klass: str, show_xlabel: bool = True, is_right: bool = False) -> None:
    p = storm(CASES[klass]["total"]); a = accum24(p); f = hydrograph(CASES[klass]["peak"])
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, height_ratios=[1.0, 1.55], hspace=0.07)
    axp = fig.add_subplot(inner[0]); axf = fig.add_subplot(inner[1], sharex=axp)

    # Right-column panels carry their y ticks + label on the RIGHT so the centre
    # gap stays clear and the axis reads on the figure's outer edge.
    if is_right:
        for ax in (axp, axf):
            ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")

    wet_p = a >= PTHR; wet_f = f >= QTHR
    active = wet_p | wet_f
    if active.any():
        i0, i1 = DAYS[active][0], DAYS[active][-1]
        for ax in (axp, axf):
            ax.axvspan(i0, i1, color=CLR[klass], alpha=0.12, zorder=0)

    # precip (top): hourly bars + trailing 24-h accumulation + 2.5 in trigger
    axp.bar(DAYS, p, width=0.035, color="#7a9bb3", alpha=0.95, zorder=2)
    axp.plot(DAYS, a, color="#1565c0", lw=2.0, zorder=3)
    if wet_p.any():
        axp.plot(DAYS[wet_p], a[wet_p], ".", color="#d32f2f", ms=8, zorder=4)
    axp.axhline(PTHR, color="#d32f2f", ls="--", lw=1.6)
    axp.set_ylim(0, max(PTHR * 1.3, a.max() * 1.12))
    axp.tick_params(labelbottom=False, labelsize=15)
    axp.set_title(f"{klass}  —  {DEF[klass]}", fontsize=16, color=CLR[klass],
                  fontweight="bold", loc="left", pad=6)

    # hydrograph (bottom): flow + flood threshold Q
    axf.plot(DAYS, f, color="#1b5e76", lw=2.0, zorder=3)
    if wet_f.any():
        axf.fill_between(DAYS, 0, f, where=wet_f, color="#1b5e76", alpha=0.25, zorder=2)
    axf.axhline(QTHR, color="#0d2c54", ls="--", lw=1.6)
    axf.set_ylim(0, max(QTHR, f.max()) * 1.15)
    axf.set_xlim(0, DAYS[-1])
    if show_xlabel:
        axf.set_xlabel("Days", fontsize=16)
    axf.tick_params(labelsize=15)


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(10, 6))
    outer = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.08,
                             left=0.07, right=0.93, top=0.93, bottom=0.09)
    # y ticks live on the OUTER edge of each column (left column on the left,
    # right column on the right); no axis labels or threshold captions.
    draw(fig, outer[0, 0], "TP", show_xlabel=False, is_right=False)
    draw(fig, outer[0, 1], "FP", show_xlabel=False, is_right=True)
    draw(fig, outer[1, 0], "FN", show_xlabel=True,  is_right=False)
    draw(fig, outer[1, 1], "TN", show_xlabel=True,  is_right=True)
    return fig


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]; prefix = cfg["aws"]["output_prefix"]

    fig = build_figure()
    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=200, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
