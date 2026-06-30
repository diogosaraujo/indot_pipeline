"""tc_vs_csi_ridge.py

Does each station's Kirpich time of concentration (Tc) land on its cluster's
CSI ridge?  This is the validation behind using Tc as the accumulation duration
(per station, RP-independent → transferable to Q50/Q100) instead of an
empirically tuned per-cluster duration.

For the MRMS (`nearest`) source at the Q10 flood target, CSI is pooled per
cluster over (duration, precip_rp) [sum TP/FP/FN, then CSI].  For each cluster:

  • top panel  — the CSI heatmap (precip_rp × duration) you already inspect,
  • bottom panel — the CSI "ridge" = max CSI over precip_rp at each duration,
  • overlaid    — the distribution of station Kirpich Tc (rug + box + median).

If the Tc cloud sits under the high-CSI plateau, Kirpich Tc is a good duration
proxy and no urbanization-aware Tc / per-cluster tuning is needed.

Output:
    s3://<bucket>/<prefix>analysis/figures/tc_vs_csi_ridge.png (+ .svg)

Usage:
    python scripts/tc_vs_csi_ridge.py
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

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tc_ridge")

SOURCE = "nearest"           # MRMS nearest pixel
FLOW_RP = 10                 # Q10 — the return period that was clustered
EPS = 1e-9
OUTPUT_KEY = "analysis/figures/tc_vs_csi_ridge"

DURATION_LABELS = {
    1: "1 h", 2: "2 h", 3: "3 h", 6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d",
    72: "3 d", 96: "4 d", 120: "5 d", 168: "7 d", 240: "10 d", 480: "20 d",
    720: "30 d", 1080: "45 d", 1440: "60 d",
}


def _read_parquet(bucket: str, key: str, columns=None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _read_csv(bucket: str, key: str, **kw) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), **kw)


def tc_to_pos(tc_hr: float, durations: list[int]) -> float:
    """Map a Tc (hours) to a fractional column position on the (log) duration
    axis: cell i is centred at i+0.5, so a Tc between grid points interpolates."""
    centres = np.arange(len(durations)) + 0.5
    return float(np.interp(np.log(tc_hr),
                           np.log(durations), centres,
                           left=centres[0], right=centres[-1]))


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    # ── CSI, pooled per cluster over (duration, precip_rp), MRMS / Q10 ─────────
    df = _read_parquet(bucket, f"{prefix}analysis/event_confusion_matrix.parquet",
                       ["site_no", "cluster", "source", "duration_hr",
                        "precip_rp_yr", "flow_rp_yr", "tp", "fp", "fn"])
    df = df[(df["source"] == SOURCE) & (df["flow_rp_yr"] == FLOW_RP)].copy()
    if df.empty or "cluster" not in df.columns:
        log.error("No %s / Q%d rows with a cluster column.", SOURCE, FLOW_RP)
        return
    df["site_no"] = df["site_no"].astype(str)

    pooled = (df.groupby(["cluster", "duration_hr", "precip_rp_yr"])[["tp", "fp", "fn"]]
                .sum().reset_index())
    pooled["CSI"] = pooled["tp"] / (pooled["tp"] + pooled["fp"] + pooled["fn"] + EPS)

    # ── Kirpich Tc per station, with its cluster ──────────────────────────────
    clusters = _read_csv(bucket, f"{prefix}clusters/clusters_k3.csv",
                         dtype={"site_no": str})[["site_no", "cluster"]]
    basin = _read_parquet(bucket, f"{prefix}watersheds/basin_characteristics.parquet",
                          ["site_no", "tc_hr"])
    basin["site_no"] = basin["site_no"].astype(str)
    tc = clusters.merge(basin, on="site_no", how="left").dropna(subset=["tc_hr"])

    durations = sorted(pooled["duration_hr"].unique())
    precip_rps = sorted(pooled["precip_rp_yr"].unique())
    clusters_ids = sorted(int(c) for c in pooled["cluster"].dropna().unique())
    n = len(clusters_ids)

    fig, axes = plt.subplots(
        2, n, figsize=(6.0 * n, 8.4), sharex="col",
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.07, "wspace": 0.22},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, c in enumerate(clusters_ids):
        ax_h, ax_r = axes[0, j], axes[1, j]
        sub = pooled[pooled["cluster"] == c]
        grid = (sub.pivot(index="precip_rp_yr", columns="duration_hr", values="CSI")
                   .reindex(index=precip_rps, columns=durations))
        M = grid.to_numpy(dtype=float)

        # ── heatmap (precip_rp rows × duration cols) ──────────────────────────
        im = ax_h.imshow(M, aspect="auto", origin="lower", cmap="YlOrRd",
                         extent=[0, len(durations), 0, len(precip_rps)])
        cb = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.02)
        cb.set_label("CSI")
        ax_h.set_yticks(np.arange(len(precip_rps)) + 0.5)
        ax_h.set_yticklabels([f"P{int(r)}" for r in precip_rps], fontsize=8)
        ax_h.set_ylabel("Precip return period")

        # ── CSI ridge = best precip_rp at each duration ───────────────────────
        ridge = np.nanmax(M, axis=0)
        centres = np.arange(len(durations)) + 0.5
        ax_r.plot(centres, ridge, "-o", color="#b30000", ms=4, lw=1.6,
                  label="max CSI over precip RP")
        best = int(np.nanargmax(ridge))
        ax_r.axvline(centres[best], color="#b30000", ls=":", lw=1.2)
        ax_h.axvline(centres[best], color="#b30000", ls=":", lw=1.2)

        # ── station Kirpich Tc: rug + box + median, on the same axis ───────────
        tc_c = tc[tc["cluster"] == c]["tc_hr"].astype(float)
        pos = np.array([tc_to_pos(v, durations) for v in tc_c])
        tc_med = float(tc_c.median())
        pos_med = tc_to_pos(tc_med, durations)
        for ax in (ax_h, ax_r):
            ax.axvline(pos_med, color="#08519c", lw=2.0)
        # rug just above the ridge axis baseline; box summarises the spread
        y0, y1 = 0.0, float(np.nanmax(ridge)) if np.isfinite(np.nanmax(ridge)) else 1.0
        ax_r.plot(pos, np.full_like(pos, y0 + 0.03 * (y1 - y0)), "|",
                  color="#08519c", ms=10, alpha=0.6)
        ax_r.boxplot(pos, vert=False, positions=[y0 + 0.12 * (y1 - y0)],
                     widths=0.06 * (y1 - y0), manage_ticks=False,
                     patch_artist=True,
                     boxprops=dict(facecolor="#c6dbef", edgecolor="#08519c"),
                     medianprops=dict(color="#08519c"),
                     flierprops=dict(marker=".", markersize=3,
                                     markeredgecolor="#08519c"))

        ax_r.set_xticks(centres)
        ax_r.set_xticklabels([DURATION_LABELS.get(d, str(d)) for d in durations],
                             rotation=45, ha="right", fontsize=8)
        ax_r.set_ylabel("CSI")
        ax_r.set_xlim(0, len(durations))
        ax_r.grid(axis="x", ls=":", alpha=0.4)

        n_st = int(tc_c.shape[0])
        ax_h.set_title(
            f"Cluster {c}  (n={n_st} stations)\n"
            f"best-CSI dur = {DURATION_LABELS.get(durations[best], durations[best])}"
            f"   |   median Tc = {tc_med:.0f} h",
            fontsize=10)

    # legend (proxy handles) on the first ridge panel
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#b30000", marker="o", lw=1.6, label="CSI ridge (best precip RP)"),
        Line2D([0], [0], color="#b30000", ls=":", lw=1.2, label="best-CSI duration"),
        Line2D([0], [0], color="#08519c", lw=2.0, label="median Kirpich Tc"),
        Line2D([0], [0], color="#08519c", marker="|", ls="none", ms=10, label="station Tc"),
    ]
    axes[1, 0].legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "Kirpich Tc vs CSI ridge per cluster — MRMS nearest pixel, Q10 target\n"
        "blue = station accumulation time (Tc); red = duration where CSI peaks",
        fontsize=12, y=0.99)

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUTPUT_KEY}.{ext}")
        log.info("Saved s3://%s/%s%s.%s", bucket, prefix, OUTPUT_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
