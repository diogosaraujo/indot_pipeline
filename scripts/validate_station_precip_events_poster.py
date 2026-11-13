"""validate_station_precip_events_poster.py

Poster version of the `station_vs_mrms_bars` figure from
validate_station_precip_events.py: precipitation events per year at or above each
Atlas-14 return period, station gauges vs MRMS storm objects. Redraws from the
summary that script already wrote, so nothing is recomputed:
    s3://<bucket>/<prefix>analysis/precip_stations/station_vs_mrms_summary.csv

Layout: 7.3 x 4.3 in, Times New Roman, 600 dpi, transparent background outside
the plot area. Text sizes 16 pt (ticks), 18 pt (axis labels, legend).

ATLAS-14 LABEL CAVEAT — the x-axis keeps the NOMINAL labels on purpose. Both
counts used the Atlas-14 table that 07_extract_atlas14.py mislabelled by one
duration (see 07d_relabel_atlas14_durations.py). Unlike 09h / 09l, there is no
single corrected label per bar: each event is classified by the highest RP
exceeded at ANY of 1/3/6/12/24 h, and the true ARI behind a stored label
differs by duration (median over sites; nominal P1000 is really ~P80 at 1 h,
~P515 at 3 h, ~P450 at 24 h; nominal P100 is ~P17 / ~P62 / ~P46). Getting
correct bars needs this script's parent and validate_storm_events.py re-run
after 07d. utils.atlas14_label_state reports which case applies, and the script
refuses if the tables are fixed but the summary CSV predates them.

Writes results/station_vs_mrms_bars_poster.{png,pdf}

Usage:
    python scripts/validate_station_precip_events_poster.py
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

from utils import atlas14_label_state, load_config, s3_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("station_bars_poster")

FIG_W, FIG_H, DPI = 7.3, 4.3, 600
FS_S, FS_M = 16, 18
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
STEM = "station_vs_mrms_bars_poster"
CSV_KEY = "analysis/precip_stations/station_vs_mrms_summary.csv"
STA_COLOR, MRMS_COLOR = "#2c7fb8", "#d95f0e"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": FS_S,
})


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    state = atlas14_label_state(bucket, prefix, [CSV_KEY])
    log.info("Atlas-14 tables: %s (%s)", state, "labels nominal, see docstring"
             if state == "buggy" else "labels true")
    body = s3_client().get_object(Bucket=bucket, Key=f"{prefix}{CSV_KEY}")["Body"].read()
    summary = pd.read_csv(io.BytesIO(body)).sort_values("return_period_yr")
    log.info("Loaded summary:\n%s", summary.to_string(index=False))

    rps = summary["return_period_yr"].astype(int).tolist()
    x = np.arange(len(rps))
    st = summary["station_events_per_yr"].to_numpy(float)
    mr = summary["mrms_storms_per_yr"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.175)
    ax.set_facecolor("white")                                # plot area stays opaque

    w = 0.38
    ax.bar(x - w / 2, st, w, color=STA_COLOR, label="Station events / yr", zorder=3)
    ax.bar(x + w / 2, mr, w, color=MRMS_COLOR, label="MRMS storms / yr", zorder=3)

    ax.set_yscale("log")
    ymax = 10 ** np.ceil(np.log10(max(st.max(), mr.max())))
    ax.set_ylim(1, ymax)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.set_yticklabels(["1", "10", "100", "1000"][: int(np.log10(ymax)) + 1])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{rp}" for rp in rps])
    ax.set_xlim(-0.6, len(rps) - 0.4)
    ax.tick_params(labelsize=FS_S)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.set_ylabel("Events per year", fontsize=FS_M)
    ax.set_xlabel("Atlas-14 return period", fontsize=FS_M, labelpad=2)
    ax.legend(loc="upper right", fontsize=FS_M, frameon=True, framealpha=0.9,
              handlelength=1.4, borderaxespad=0.4)

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):                               # exact size, no bbox_tight
        out = OUT_DIR / f"{STEM}.{ext}"
        fig.savefig(out, dpi=DPI, facecolor="none")
        log.info("Saved %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
