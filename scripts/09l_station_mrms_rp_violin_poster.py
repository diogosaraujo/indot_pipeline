"""09l_station_mrms_rp_violin_poster.py

Poster version of the 09l paired observed-return-period violins (Precipitation
Station vs nearest-pixel MRMS, one pair per Atlas-14 ARI threshold). Redraws from
the per-alarm table 09l already wrote, so nothing is recomputed:
    s3://<bucket>/<prefix>analysis/station_vs_mrms_rp_per_alarm.csv

Layout: 7.3 x 4.3 in, Times New Roman, 600 dpi, transparent background outside
the plot area. Text sizes 16 pt (ticks, n labels), 18 pt (axis labels, legend).
The legend sits in one row above the n labels so it doesn't cover the violin tails.

ATLAS-14 LABEL CORRECTION (default ON), same as 09h_tc_performance_poster.py.
Both 09l triggers used Atlas-14 depths from tables that 07_extract_atlas14.py
mislabelled by one duration (MRMS: gauge-site table; station: the rain station's
own table). The violins themselves are correct; only the ARI name on each
threshold is wrong. Median true ARI over the 106 gauges is the same for both
tables to the rounding shown:
    nominal  1    2    5    10   25   50   100  200  500  1000
    true    <1   1.2  2.9  5.7  14   27   54   110  290  580
Pass --nominal for the old P1..P1000 labels. The script decides automatically
(utils.atlas14_label_state): corrected labels while the Atlas-14 tables are
buggy; plain 1..1000 once 07d has run and 09l has been re-run; refuses if the
tables are fixed but the per-alarm CSV is stale.

Writes results/station_vs_mrms_rp_violin_poster[_nominal].{png,pdf}

Usage:
    python scripts/09l_station_mrms_rp_violin_poster.py
    python scripts/09l_station_mrms_rp_violin_poster.py --nominal
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import atlas14_label_state, load_config, s3_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09l_poster")

FIG_W, FIG_H, DPI = 7.3, 4.3, 600
FS_S, FS_M = 16, 18
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
STEM = "station_vs_mrms_rp_violin_poster"
CSV_KEY = "analysis/station_vs_mrms_rp_per_alarm.csv"

ARIS       = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
DISP_CAP   = 500.0
STA_COLOR  = "#d94801"
MRMS_COLOR = "#2171b5"
SOURCES    = [("station", STA_COLOR, "Precipitation Station"),
              ("mrms", MRMS_COLOR, "Nearest-pixel MRMS")]
EFFECTIVE_ARI = {1: "<1", 2: "1.2", 5: "2.9", 10: "5.7", 25: "14", 50: "27",
                 100: "54", 200: "110", 500: "290", 1000: "580"}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": FS_S,
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", action="store_true",
                    help="label the x-axis with the stored (mislabelled) P1..P1000")
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    state = atlas14_label_state(bucket, prefix, [CSV_KEY])
    nominal = args.nominal and state == "buggy"              # old P-labels, buggy data
    labels = ([EFFECTIVE_ARI[p] for p in ARIS] if state == "buggy" and not nominal
              else [str(p) for p in ARIS])
    log.info("Atlas-14 tables: %s -> %s labels", state, "P-nominal" if nominal else "ARI (yr)")
    body = s3_client().get_object(Bucket=bucket, Key=f"{prefix}{CSV_KEY}")["Body"].read()
    df = pd.read_csv(io.BytesIO(body))
    log.info("Loaded %d alarms", len(df))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.75, bottom=0.165)
    ax.set_facecolor("white")                                # plot area stays opaque

    off, width = 0.21, 0.38
    for src, color, label in SOURCES:
        vpos, vdata, med_x, med_y = [], [], [], []
        for i, p in enumerate(ARIS):
            rps = df.loc[(df["source"] == src) & (df["ari_yr"] == p), "observed_rp"].to_numpy(float)
            x = i + (-off if src == "station" else off)
            if rps.size >= 2:
                d = np.log10(np.minimum(rps, DISP_CAP))
                vpos.append(x); vdata.append(d)
                med_x.append(x); med_y.append(float(np.median(d)))
            elif rps.size == 1:
                y = np.log10(min(rps[0], DISP_CAP))
                ax.scatter([x], [y], color=color, s=30, zorder=6)
                med_x.append(x); med_y.append(y)
            ax.text(x, 1.015, f"n={rps.size}", transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=FS_S, color=color,
                    clip_on=False)
        if vdata:
            parts = ax.violinplot(vdata, positions=vpos, widths=width,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.7)
                pc.set_edgecolor("0.3"); pc.set_linewidth(0.6)
            parts["cmedians"].set_edgecolor("black"); parts["cmedians"].set_linewidth(1.6)
        ax.plot(med_x, med_y, color=color, lw=2.2, marker="o", ms=6, mec="k", mew=0.6,
                zorder=7, label=label)

    ytick_rp = [2, 5, 10, 50, 100, 500]
    ax.set_yticks(np.log10(ytick_rp)); ax.set_yticklabels([str(v) for v in ytick_rp])
    ax.set_ylim(np.log10(0.95), np.log10(DISP_CAP * 1.05))
    ax.set_xticks(range(len(ARIS)))
    ax.set_xticklabels([f"P{p}" for p in ARIS] if nominal else labels)
    ax.set_xlim(-0.6, len(ARIS) - 0.4)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(labelsize=FS_S)
    ax.set_ylabel("Observed USGS\nreturn period (yr)", fontsize=FS_M, linespacing=1.0)
    ax.set_xlabel("Atlas 14 precipitation ARI threshold" if nominal
                  else "Precipitation ARI threshold (yr)", fontsize=FS_M, labelpad=2)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5675, 1.0), ncol=2, fontsize=FS_M,
               frameon=False, handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)

    OUT_DIR.mkdir(exist_ok=True)
    stem = STEM + ("_nominal" if nominal else "")
    for ext in ("png", "pdf"):                               # exact size, no bbox_tight
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=DPI, facecolor="none")
        log.info("Saved %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
