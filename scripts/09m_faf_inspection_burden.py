"""09m_faf_inspection_burden.py

False-alarm INSPECTION BURDEN of the Tc precipitation trigger, MRMS vs station,
reframing FAF (false alarms per station-year) as bridges inspected per year:

    left  y-axis : all bridges over water     = 16,940 x FAF   (log)
    right y-axis : scour-critical bridges only =    758 x FAF   (log)

Both axes are the same bars scaled by a constant. Three stacked panels (Q10, Q50,
Q100); at each ARI (P1..P1000) two grouped bars — MRMS nearest pixel (solid) and
precipitation station (hatched) — with the actual FAF value printed on each bar.

    FAF = Σ FP / Σ (n_common_hours / 8766)      [false alarms per station-year]
        MRMS    : analysis/event_confusion_matrix_tc.parquet        (08c, source='nearest')
        Station : analysis/event_confusion_matrix_tc_station.parquet (08h, source='station_nearest')

Figure: 6.5 x 6.5 in, fonts >= 7.

Writes:
    s3://<bucket>/<prefix>analysis/figures/faf_inspection_burden.{png,pdf}
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09m_faf")

FLOW_RPS       = [10, 50, 100]
ARIS           = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
EPS            = 1e-9
HOURS_PER_YEAR = 8766.0
ALL_BRIDGES    = 16940      # bridges over water           (left axis  = ALL_BRIDGES x FAF)
SCOUR_BRIDGES  = 758        # scour-critical bridges only   (right axis = SCOUR_BRIDGES x FAF)

NEAREST_KEY = "analysis/event_confusion_matrix_tc.parquet"
STATION_KEY = "analysis/event_confusion_matrix_tc_station.parquet"
FIG_KEY     = "analysis/figures/faf_inspection_burden"

MRMS_COLOR, MRMS_EDGE = "#2171b5", "#0b3d66"     # solid
STA_FACE,  STA_EDGE   = "#fdd0a2", "#d94801"      # hatched


def _read(bucket, key, columns=None):
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2g}")


def faf_series(df, source, flow_rp) -> np.ndarray:
    """False alarms per station-year for each ARI, pooled over stations."""
    sub = df[(df["source"] == source) & (df["flow_rp_yr"] == flow_rp)]
    if sub.empty:
        return np.zeros(len(ARIS))
    fp = sub.groupby("precip_rp_yr")["fp"].sum().reindex(ARIS).fillna(0.0)
    station_years = sub.groupby("site_no")["n_common_hours"].first().sum() / HOURS_PER_YEAR
    return (fp / (station_years + EPS)).to_numpy()


def make_figure(faf, bucket, prefix):
    x = np.arange(len(ARIS))
    w = 0.38

    # Shared log range across all panels/sources (so Q10/Q50/Q100 are comparable).
    allv = np.concatenate([faf[q][s] for q in FLOW_RPS for s in ("nearest", "station")])
    pos = allv[allv > 0]
    if pos.size == 0:
        pos = np.array([1e-3])
    floor = (pos.min() / 8) * ALL_BRIDGES
    ytop  = pos.max() * ALL_BRIDGES * 5           # headroom for the rotated FAF labels

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.5), sharex=True)
    fig.subplots_adjust(left=0.125, right=0.875, top=0.905, bottom=0.095, hspace=0.42)

    for ax, flow_rp in zip(axes, FLOW_RPS):
        fm, fs = faf[flow_rp]["nearest"], faf[flow_rp]["station"]
        ax.set_yscale("log"); ax.set_ylim(floor, ytop)

        hm = np.where(fm > 0, fm * ALL_BRIDGES, floor)
        hs = np.where(fs > 0, fs * ALL_BRIDGES, floor)
        ax.bar(x - w / 2, hm - floor, w, bottom=floor, color=MRMS_COLOR,
               edgecolor=MRMS_EDGE, linewidth=0.4, zorder=3)
        ax.bar(x + w / 2, hs - floor, w, bottom=floor, facecolor=STA_FACE,
               edgecolor=STA_EDGE, hatch="////", linewidth=0.5, zorder=3)

        for xi, f in zip(x - w / 2, fm):
            if f > 0:
                ax.annotate(_fmt(f), (xi, f * ALL_BRIDGES), textcoords="offset points",
                            xytext=(0, 2), ha="center", va="bottom", fontsize=7, rotation=90, color="0.15")
        for xi, f in zip(x + w / 2, fs):
            if f > 0:
                ax.annotate(_fmt(f), (xi, f * ALL_BRIDGES), textcoords="offset points",
                            xytext=(0, 2), ha="center", va="bottom", fontsize=7, rotation=90, color="0.15")

        ax2 = ax.twinx()                          # right axis = scour-critical (proportional)
        ax2.set_yscale("log")
        ax2.set_ylim(floor * SCOUR_BRIDGES / ALL_BRIDGES, ytop * SCOUR_BRIDGES / ALL_BRIDGES)
        ax2.tick_params(labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.text(0.985, 0.93, f"Q{flow_rp}", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"P{a}" for a in ARIS], fontsize=8)
    axes[-1].set_xlabel("Precipitation ARI threshold (return period, yr)", fontsize=8)

    fig.text(0.028, 0.5, "Inspections / yr — all bridges over water (16,940 × FAF)",
             rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.972, 0.5, "Inspections / yr — scour-critical bridges (758 × FAF)",
             rotation=270, va="center", ha="center", fontsize=8)

    handles = [Patch(facecolor=MRMS_COLOR, edgecolor=MRMS_EDGE, label="MRMS (nearest pixel)"),
               Patch(facecolor=STA_FACE, edgecolor=STA_EDGE, hatch="////", label="Precipitation station")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.99))

    for ext in ("png", "pdf"):
        buf = io.BytesIO()
        kw = {"format": ext}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(buf, **kw)
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    cols = ["site_no", "source", "precip_rp_yr", "flow_rp_yr", "fp", "n_common_hours"]
    df_near = _read(bucket, f"{prefix}{NEAREST_KEY}", cols)
    df_sta  = _read(bucket, f"{prefix}{STATION_KEY}", cols)

    faf = {q: {"nearest": faf_series(df_near, "nearest", q),
               "station": faf_series(df_sta, "station_nearest", q)} for q in FLOW_RPS}
    for q in FLOW_RPS:
        log.info("Q%d FAF — MRMS max=%.3f, station max=%.3f",
                 q, faf[q]["nearest"].max(), faf[q]["station"].max())
    make_figure(faf, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
