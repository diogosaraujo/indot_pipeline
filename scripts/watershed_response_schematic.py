"""watershed_response_schematic.py

5 x 5 in schematic: the SAME 2.5 in storm drives two watersheds differently —
one exceeds Q10 (flood), the other stays below Q10 — motivating a watershed-
specific trigger instead of a single fixed depth.  Synthetic hydrographs.

Writes:
    s3://<bucket>/<prefix>analysis/figures/watershed_response_2p5in.{png,svg}
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, write_bytes_to_s3

FIG_KEY = "analysis/figures/watershed_response_2p5in"

Q10 = 1.0
T = np.linspace(0, 48, 500)
P_T = np.arange(0, 6)                      # 2.5 in storm over the first 6 h
P_H = np.array([0.2, 0.4, 0.6, 0.5, 0.4, 0.4])   # sums to 2.5 in


def hydro(peak, tp, k):
    """Gamma-shaped unit hydrograph, peaking at value `peak` at time `tp`."""
    return peak * (T / tp) ** k * np.exp(k * (1 - T / tp))


def panel(ax, q, color, exceeds: bool):
    ax.axis("off")
    ax.set_xlim(-1, 48)
    ax.set_ylim(0, 1.9)

    # Q10 threshold
    ax.axhline(Q10, color="#222", ls="--", lw=1.4)
    ax.text(47.5, Q10 + 0.03, "Q10", ha="right", va="bottom", fontsize=12, color="#222")

    # hydrograph
    ax.plot(T, q, color=color, lw=2.6)
    if exceeds:
        ax.fill_between(T, Q10, q, where=q > Q10, color=color, alpha=0.30)
    tag = "Q10 exceeded" if exceeds else "Q10 not reached"
    ax.text(0.5, 1.78, tag, fontsize=12, color=color, fontweight="bold", va="top")

    # 2.5 in storm (hangs from the top, same in both panels)
    axp = ax.twinx(); axp.axis("off"); axp.set_ylim(0, 2.4); axp.invert_yaxis()
    axp.bar(P_T, P_H, width=0.9, align="edge", color="#7fb3d5")
    ax.text(6.4, 1.72, "2.5 in", fontsize=11, color="#2471a3", va="top")


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
    panel(ax1, hydro(1.5, 10, 3.5), "#c0392b", exceeds=True)    # responsive basin
    panel(ax2, hydro(0.6, 18, 3.0), "#2471a3", exceeds=False)   # damped basin
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.14)

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=200, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        print(f"Wrote s3://{bucket}/{prefix}{FIG_KEY}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
