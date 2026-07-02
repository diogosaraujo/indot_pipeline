"""station_funnel.py

Reports the station funnel — how many USGS streamflow gauges we started with and
how many survived each exclusion to reach the fixed-Tc comparison (08c) — with
the real counts read from S3.  Also breaks down WHY stations were dropped.

Stages / exclusions:
    0. Inventory (01)        all Indiana streamflow gauges (param 00060)
    1. Has IV streamflow (02)
    2. flow_stats universe   stations evaluated for a flood-frequency fit
    3. LP3 fit (04b)         lp3_peak_series vs insufficient, with reasons:
                               - no_peak_record   (no USGS annual peaks)
                               - regulated_excluded (peak_cd 5/6 → Rule B)
                               - insufficient_record (< 10 clean annual peaks)
    4. Clustered (03c)       LP3-fitted AND complete basin characteristics
    5. Used in 08c           clustered AND valid Q AND Kirpich Tc AND Atlas14
                             AND streamflow AND a non-empty flow∩MRMS window

Usage:
    python scripts/station_funnel.py
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Polygon

from utils import load_config, write_bytes_to_s3

FUNNEL_FIG_KEY = "analysis/figures/station_funnel"

# Reuse 08's loaders so keys / logic match the real 08c run exactly.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TC_KEY = "analysis/event_confusion_matrix_tc.parquet"
LP3_SUMMARY_KEY = "analysis/lp3_frequency_curves/lp3_summary.csv"


def _pq(bucket, prefix, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=prefix + key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def _csv(bucket, prefix, key, **kw):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=prefix + key)
    return pd.read_csv(io.BytesIO(o["Body"].read()), **kw)


def draw_funnel(stages: list[tuple[str, int, str | None]],
                bucket: str, prefix: str) -> None:
    """Draw a funnel: station count per stage (width ∝ count) with the exclusion
    criteria annotated alongside each transition.

    stages: list of (label, count, criteria_that_removed_the_drop_into_this_stage).
            The first stage's criteria is None (it is the starting population)."""
    n = len(stages)
    top = stages[0][1]
    counts = [c for _, c, _ in stages]
    hw = [0.5 * c / top for c in counts]                 # half-widths (max 0.5)

    band_h, gap = 0.72, 0.55
    y_ctr = [-i * (band_h + gap) for i in range(n)]
    colors = plt.cm.Blues(np.linspace(0.42, 0.88, n))

    fig, ax = plt.subplots(figsize=(13, 1.7 * n + 1.2))

    for i in range(n):
        yt, yb = y_ctr[i] + band_h / 2, y_ctr[i] - band_h / 2
        ax.add_patch(Polygon([(-hw[i], yt), (hw[i], yt), (hw[i], yb), (-hw[i], yb)],
                             closed=True, facecolor=colors[i], edgecolor="white", lw=1.5))
        # connector trapezoid to the next (narrower) band
        if i < n - 1:
            yb2 = y_ctr[i + 1] + band_h / 2
            ax.add_patch(Polygon(
                [(-hw[i], yb), (hw[i], yb), (hw[i + 1], yb2), (-hw[i + 1], yb2)],
                closed=True, facecolor="0.85", edgecolor="none", alpha=0.6))
        # stage label + count inside the band
        txt_color = "white" if i >= n / 2 else "0.10"
        label, count, _ = stages[i]
        ax.text(0, y_ctr[i], f"{label}\n{count} stations",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=txt_color, linespacing=1.3)

        # drop + criteria annotation to the right, at the transition
        if i < n - 1:
            drop = counts[i] - counts[i + 1]
            crit = stages[i + 1][2] or ""
            y_mid = (yb + (y_ctr[i + 1] + band_h / 2)) / 2
            ax.annotate("", xy=(0.60, y_mid), xytext=(hw[i + 1] + 0.01, y_mid),
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
            ax.text(0.63, y_mid, f"−{drop}", ha="left", va="center",
                    fontsize=11, fontweight="bold", color="#b30000")
            ax.text(0.90, y_mid, crit, ha="left", va="center",
                    fontsize=8, color="0.20", linespacing=1.3)

    kept = counts[-1]
    ax.set_title("USGS streamflow station funnel — flood-trigger comparison (08c)\n"
                 f"{top} Indiana gauges  →  {kept} evaluated in 08c "
                 f"({100.0 * kept / top:.0f}% retained)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.72, 3.3)
    ax.set_ylim(y_ctr[-1] - band_h, y_ctr[0] + band_h)
    ax.axis("off")

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FUNNEL_FIG_KEY}.{ext}")
        print(f"   saved s3://{bucket}/{prefix}{FUNNEL_FIG_KEY}.{ext}")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    def hdr(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

    # 0. Inventory ------------------------------------------------------------
    inv = _pq(bucket, prefix, "stations/indiana_streamflow_sites.parquet",
              ["site_no"])
    inv_sites = set(inv["site_no"].astype(str))
    hdr("STATION FUNNEL")
    print(f"0. Inventory (01)                 : {len(inv_sites):5d}  Indiana streamflow gauges")

    # 1. Has IV streamflow ----------------------------------------------------
    try:
        sf = _pq(bucket, prefix, "streamflow/instantaneous/all_gauges_long.parquet",
                 ["site_no"])
        iv_sites = set(sf["site_no"].astype(str))
        print(f"1. Has IV streamflow (02)         : {len(iv_sites):5d}")
    except Exception as e:
        print(f"1. Has IV streamflow (02)         :   n/a  ({e})")

    # 2/3. flow_stats + LP3 outcome ------------------------------------------
    fs = _pq(bucket, prefix, "flow_stats/per_gauge_flow_stats.parquet")
    fs["site_no"] = fs["site_no"].astype(str)
    print(f"2. flow_stats universe            : {len(fs):5d}  stations evaluated (04b)")
    if "source" in fs.columns:
        hdr("3. LP3 fit (04b) — source")
        print(fs["source"].value_counts(dropna=False).to_string())
    fitted = set(fs.loc[fs.get("source") == "lp3_peak_series", "site_no"])
    print(f"\n   LP3-fitted (lp3_peak_series)   : {len(fitted):5d}")

    lp3_counts: dict = {}
    try:
        lp3 = _csv(bucket, prefix, LP3_SUMMARY_KEY, dtype={"site_no": str})
        lp3_counts = lp3["status"].value_counts(dropna=False).to_dict()
        print("\n   04b status breakdown (lp3_summary.csv):")
        print(lp3["status"].value_counts(dropna=False).to_string())
    except Exception as e:
        print(f"\n   lp3_summary.csv not read: {e}")

    # 4. Clustered ------------------------------------------------------------
    clusters = m.load_clusters(bucket, prefix)          # dict site_no → cluster
    clustered = set(clusters)
    hdr("4. Clustered (03c)")
    print(f"   Clustered stations             : {len(clustered):5d}")
    cl_series = pd.Series(list(clusters.values()))
    print("   per cluster:")
    print(cl_series.value_counts().sort_index().to_string())

    # 5. Used in 08c (reconstruct the exact intersection + window) ------------
    atlas14 = m.load_atlas14(bucket, prefix)
    tc_by_site = _pq(bucket, prefix, "watersheds/basin_characteristics.parquet",
                     ["site_no", "tc_hr"])
    tc_by_site["site_no"] = tc_by_site["site_no"].astype(str)
    tc_sites = set(tc_by_site.dropna(subset=["tc_hr"])["site_no"])
    q_cols = [f"Q{rp}" for rp in m.FLOW_RPS if f"Q{rp}" in fs.columns]
    has_q = set(fs.loc[fs[q_cols].notna().any(axis=1), "site_no"])
    a14_sites = set(atlas14["site_no"].astype(str))
    sf_sites = set(m.load_streamflow(bucket, prefix)["site_no"].astype(str))

    # 08c universe is now decoupled from clustering: physical requirements only.
    stations_all = has_q & tc_sites & a14_sites & sf_sites
    hdr("5. Used in 08c  (clustering NOT a gate)")
    print(f"   valid Q (LP3-fitted)           : {len(has_q):5d}")
    print(f"   ∩ Kirpich Tc                   : {len(has_q & tc_sites):5d}")
    print(f"   ∩ Atlas14                      : {len(has_q & tc_sites & a14_sites):5d}")
    print(f"   ∩ streamflow                   : {len(stations_all):5d}")
    print(f"   (for reference, with old cluster gate: "
          f"{len(clustered & stations_all):5d})")

    # what 08c actually wrote
    try:
        used = _pq(bucket, prefix, TC_KEY, ["site_no", "flow_rp_yr"])
        used["site_no"] = used["site_no"].astype(str)
        used_sites = set(used["site_no"])
        print(f"\n   ACTUALLY USED (08c output)     : {len(used_sites):5d}  distinct stations")
        print("   by flood target:")
        for rp in m.FLOW_RPS:
            print(f"     Q{rp:<4d}: {used[used['flow_rp_yr']==rp]['site_no'].nunique():5d} stations")
        window_dropped = stations_all - used_sites
        print(f"\n   dropped by empty flow∩MRMS win : {len(window_dropped):5d}")
    except Exception as e:
        print(f"\n   08c output not read (run 08c first): {e}")

    # LP3-fitted-but-unused reasons (universe base = valid Q / LP3-fitted)
    hdr("LP3-fitted but NOT used — reason")
    print(f"   no Kirpich Tc                  : {len(has_q - tc_sites):5d}")
    print(f"   no Atlas14                     : {len((has_q & tc_sites) - a14_sites):5d}")
    print(f"   no streamflow                  : {len((has_q & tc_sites & a14_sites) - sf_sites):5d}")
    print(f"   total fitted-but-unused        : {len(has_q - stations_all):5d}  (before MRMS-window)")

    # ── Funnel figure ────────────────────────────────────────────────────────
    # Nested subsets rooted at the inventory so counts are monotonic.
    A = inv_sites
    B = A & has_q
    C = B & tc_sites
    D = C & a14_sites
    E = D & sf_sites

    reg = lp3_counts.get("regulated_excluded", "?")
    ins = lp3_counts.get("insufficient_record", "?")
    nop = lp3_counts.get("no_peak_record", "?")
    crit_fit = ("LP3 flood-frequency fit not usable:\n"
                f"• regulated / diverted — Rule B, peak_cd 5 or 6:  {reg}\n"
                f"• < 10 clean annual peaks:  {ins}\n"
                f"• no USGS annual-peak record:  {nop}")

    stages = [
        ("Indiana streamflow gauges\n(USGS parameter 00060)", len(A), None),
        ("LP3 flood-frequency fit\n(valid Q10 / Q50 / Q100)", len(B), crit_fit),
        ("+ Kirpich Tc available", len(C),
         "No time of concentration\n(missing channel slope / length)"),
        ("+ Atlas-14 DDF available", len(D),
         "No precipitation-frequency\ngrid at the gauge location"),
        ("Evaluated in 08c\n(+ streamflow record)", len(E),
         "No instantaneous\nstreamflow record"),
    ]
    hdr("Funnel figure")
    draw_funnel(stages, bucket, prefix)
    print("   NOTE: 08c additionally requires the streamflow span to overlap MRMS\n"
          "   coverage (flow∩MRMS window); that removed 0 stations in the last run.")


if __name__ == "__main__":
    main()
