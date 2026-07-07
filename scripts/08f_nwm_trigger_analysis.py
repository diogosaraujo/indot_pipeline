"""08f_nwm_trigger_analysis.py

NWM-streamflow version of the event-overlap confusion matrix (companion to 08c/08d).

Where 08c triggers on precipitation and 08d on INDOT's rain rule, this script asks:
**does an NWM streamflow exceedance anticipate a real (USGS-observed) flood?**  This
is the ungaged-basin question — in a basin with no gauge you would derive your flood
thresholds from the NWM retrospective climatology (04c) and trigger an inspection when
the operational NWM forecast/analysis exceeds them.

    TRIGGER : NWM operational streamflow ≥ Q(flow_rp)   with Q from 04c
              (NWM-Retrospective-v3.0 LP3, nwm_per_gauge_flow_stats.parquet)
    TRUTH   : USGS observed streamflow  ≥ Q(flow_rp)    with Q from 04b
              (USGS-annual-peak LP3, per_gauge_flow_stats.parquet)

Return periods are MATCHED: the NWM-Q10 trigger is scored against USGS-Q10 floods,
NWM-Q50 vs USGS-Q50, NWM-Q100 vs USGS-Q100 (FLOW_RPS = 10/50/100).

Two NWM operational products are scored so gauged vs ungaged skill can be compared:
    nwm_analysis_assim  Standard Analysis & Assimilation (nwm/analysis_assim.parquet)
                        — assimilates USGS gauges → best case at a gauged site.
    nwm_open_loop       Open-Loop A&A, no data assimilation (nwm/open_loop.parquet)
                        — the pure model → a proxy for skill in an UNGAGED basin.

Event grouping, ±24 h linking, episodes and TP/FN/FP/TN are IDENTICAL to 08 (the
helpers are imported from it, not reimplemented), so results line up with 08c/08d.
For each (station × flow_rp × source):

    TP  a USGS flood event with an NWM-trigger-wet hour in [flood_start−24h, flood_end]
    FN  a USGS flood event with no such NWM trigger                          — missed
    FP  an NWM-trigger event with no USGS flood in [trig_start, trig_end+24h] — false alarm
    TN  each dry gap between consecutive wet events counts as one

Station universe = the 106 gauges used in 08c (event_confusion_matrix_tc.parquet),
intersected per source with: a valid 04c NWM threshold, a valid 04b USGS threshold, a
USGS streamflow record, and NWM operational data.  The common analysis window per
gauge is the USGS streamflow span ∩ the NWM operational span (so both the trigger and
the truth are defined over the same hours).

Units: NWM streamflow is m³/s → converted to cfs (×35.3147) to match the cfs Q columns
(04c is already in cfs); USGS observed is native cfs.

Writes:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_nwm.parquet  (per station × flow_rp × source)
    s3://<bucket>/<prefix>analysis/nwm_trigger_metrics.csv             (global skill, one row per source × flow_rp)
    s3://<bucket>/<prefix>analysis/figures/nwm_performance_diagram.{png,svg}

Usage:
    python scripts/08f_nwm_trigger_analysis.py [--no-figure]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config, s3_client, write_bytes_to_s3, write_parquet_to_s3

# Reuse 08's loaders + event helpers so keys / overlap logic match the real runs.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08f_nwm")

FLOW_RPS    = m.FLOW_RPS                  # Q10 / Q50 / Q100 (matched trigger & truth)
CFS_PER_CMS = 35.3146667                  # NWM m³/s → cfs

TC_KEY      = "analysis/event_confusion_matrix_tc.parquet"     # 08c → the 106 gauges
NWM_Q_KEY   = "flow_stats/nwm_per_gauge_flow_stats.parquet"    # 04c → NWM trigger thresholds
# USGS truth thresholds (04b) come via m.load_flow_stats.

# NWM operational products used as the trigger source (name, S3 key).
NWM_SOURCES = [
    ("nwm_analysis_assim", "nwm/analysis_assim.parquet"),   # with DA → gauged best case
    ("nwm_open_loop",      "nwm/open_loop.parquet"),        # no DA   → ungaged proxy
]

OUTPUT_KEY  = "analysis/event_confusion_matrix_nwm.parquet"
METRICS_KEY = "analysis/nwm_trigger_metrics.csv"
FIG_KEY     = "analysis/figures/nwm_performance_diagram"


# ---------- Skill metrics (same definitions as 08d) ----------

def skill(tp: int, fp: int, fn: int, tn: int) -> dict:
    pod  = tp / (tp + fn) if (tp + fn) else np.nan
    sr   = tp / (tp + fp) if (tp + fp) else np.nan
    far  = 1.0 - sr if (tp + fp) else np.nan
    csi  = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    acc  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan
    return {"pod": pod, "far": far, "sr": sr, "csi": csi,
            "bias": bias, "f1": f1, "accuracy": acc}


# ---------- Loaders ----------

def load_nwm_thresholds(bucket: str, prefix: str) -> pd.DataFrame:
    """04c NWM-retrospective LP3 thresholds: site_no, comid, Q10/Q50/Q100 (cfs)."""
    df = m._read_parquet_s3(bucket, f"{prefix}{NWM_Q_KEY}")
    df["site_no"] = df["site_no"].astype(str)
    keep = ["site_no", "comid"] + [f"Q{rp}" for rp in FLOW_RPS]
    return df[[c for c in keep if c in df.columns]]


def load_nwm_operational(bucket: str, prefix: str, key: str) -> pd.DataFrame:
    """Hourly NWM operational streamflow → cfs. Columns: site_no, comid, datetime_utc, value_cfs."""
    df = m._read_parquet_s3(bucket, f"{prefix}{key}",
                            columns=["site_no", "comid", "datetime_utc", "streamflow_cms"])
    df["site_no"]      = df["site_no"].astype(str)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["value_cfs"]    = pd.to_numeric(df["streamflow_cms"], errors="coerce") * CFS_PER_CMS
    return df


# ---------- Per-station analysis ----------

def analyse_station(
    site_no: str,
    comid,
    nwm_hourly: pd.Series,     # trigger: NWM operational streamflow (cfs), hourly
    usgs_hourly: pd.Series,    # truth:   USGS observed streamflow (cfs), hourly max
    nwm_row: pd.Series,        # 04c thresholds for this gauge
    usgs_row: pd.Series,       # 04b thresholds for this gauge
    source: str,
    ws: pd.Timestamp,
    we: pd.Timestamp,
) -> list[dict]:
    grid = pd.date_range(ws, we, freq="1h", tz="UTC")
    trig = nwm_hourly.reindex(grid)
    pct_nwm_missing = round(100.0 * float(trig.isna().mean()), 2)
    flow = usgs_hourly.reindex(grid)
    n_common = len(grid)

    out: list[dict] = []
    for flow_rp in FLOW_RPS:
        q_nwm  = nwm_row.get(f"Q{flow_rp}")
        q_usgs = usgs_row.get(f"Q{flow_rp}")
        if pd.isna(q_nwm) or pd.isna(q_usgs):
            continue
        trig_events = m.group_wet_events((trig >= float(q_nwm)).fillna(False))
        flow_events = m.group_wet_events((flow >= float(q_usgs)).fillna(False))
        tp, fp, fn, tn = m.classify_overlap(trig_events, flow_events, ws, we)
        out.append({
            "site_no":          site_no,
            "comid":            comid,
            "source":           source,
            "flow_rp_yr":       flow_rp,
            "q_nwm_cfs":        round(float(q_nwm), 1),
            "q_usgs_cfs":       round(float(q_usgs), 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_trigger_events": len(trig_events),
            "n_flow_events":    len(flow_events),
            "pct_nwm_missing":  pct_nwm_missing,
            "common_start":     ws,
            "common_end":       we,
            "n_common_hours":   n_common,
        })
    return out


# ---------- Performance (Roebber) diagram — A&A vs open-loop ----------

def performance_diagram(pooled: pd.DataFrame, bucket: str, prefix: str) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    cl = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=8, inline=True)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:                      # frequency-bias lines: POD = b·SR
        x = np.linspace(0, 1, 10)
        ax.plot(x, np.minimum(b * x, 1), ls="--", color="0.5", lw=0.7)
        if b <= 1: ax.text(1.008, b, f"{b:g}", fontsize=7.5, color="0.4", va="center")
        else:      ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7.5, color="0.4", ha="center")

    rp_colors  = {10: "#1a9850", 50: "#f46d43", 100: "#762a83"}
    src_marker = {                                             # marker, size, label
        "nwm_analysis_assim": ("*", 560, "A&A (gauged, w/ DA)"),
        "nwm_open_loop":      ("o", 190, "Open-loop (ungaged proxy)"),
    }
    for source_name, (mk, ms, _lbl) in src_marker.items():
        for rp in FLOW_RPS:
            pr = pooled[(pooled.source == source_name) & (pooled.flow_rp_yr == rp)]
            if pr.empty:
                continue
            r = pr.iloc[0]
            if not (np.isfinite(r.sr) and np.isfinite(r.pod)):
                continue
            ax.scatter([r.sr], [r.pod], marker=mk, s=ms, color=rp_colors.get(rp, "grey"),
                       edgecolors="black", lw=1.2, zorder=7)
            ax.annotate(f"{r.csi:.2f}", (r.sr, r.pod), textcoords="offset points",
                        xytext=(7, 6), fontsize=7.5, color="0.15")

    rp_handles = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="black", ms=10, label=f"Q{rp}")
                  for rp, c in rp_colors.items()]
    src_handles = [Line2D([0], [0], marker=mk, ls="", mfc="0.6", mec="black",
                          ms=(14 if mk == "*" else 9), label=lbl)
                   for _, (mk, _s, lbl) in src_marker.items()]
    leg1 = ax.legend(handles=rp_handles, loc="upper left", fontsize=9,
                     title="Flood return period", framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=src_handles, loc="lower right", fontsize=9,
              title="NWM product", framealpha=0.95)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Success ratio  (1 − FAR)", fontsize=12)
    ax.set_ylabel("Probability of detection (POD)", fontsize=12)
    ax.text(0.985, 0.03, "frequency-bias lines", fontsize=8, color="0.4",
            ha="right", style="italic")
    ax.set_title("NWM streamflow flood trigger vs USGS-observed floods\n"
                 "trigger: NWM ≥ Q(04c) · truth: USGS ≥ Q(04b) · matched return periods "
                 "(labels = CSI)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(cf, ax=ax, label="Critical success index (CSI)", shrink=0.85)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=190, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figure", action="store_true", help="skip the performance diagram")
    args = ap.parse_args()

    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading USGS thresholds (04b), NWM thresholds (04c), USGS streamflow...")
    usgs_q     = m.load_flow_stats(bucket, prefix)             # site_no + Q10/Q50/Q100 (truth)
    nwm_q      = load_nwm_thresholds(bucket, prefix)           # site_no, comid + Q10/Q50/Q100 (trigger)
    streamflow = m.load_streamflow(bucket, prefix)             # USGS hourly-max cfs (truth series)

    tc = m._read_parquet_s3(bucket, f"{prefix}{TC_KEY}", columns=["site_no"])
    stations_08c = set(tc["site_no"].astype(str))
    log.info("08c analysis stations: %d", len(stations_08c))

    # Precompute USGS truth series + threshold rows once (no repeated full-table scans).
    flow_by_site = {s: g.set_index("datetime_utc")["value_cfs"].sort_index()
                    for s, g in streamflow.groupby("site_no")}
    usgs_fs = usgs_q.drop_duplicates("site_no").set_index("site_no")
    nwm_fs  = nwm_q.drop_duplicates("site_no").set_index("site_no")

    all_records: list[dict] = []
    for source_name, key in NWM_SOURCES:
        log.info("── Source %s (%s) ──", source_name, key)
        try:
            nwm_df = load_nwm_operational(bucket, prefix, key)
        except Exception as e:                                 # noqa: BLE001
            log.warning("Could not load %s (%s) — skipping source.", key, e)
            continue
        nwm_df = nwm_df[nwm_df["site_no"].isin(stations_08c)]

        # Hourly-max NWM trigger series per gauge (floor to the hour so reindex aligns).
        nwm_by_site: dict[str, pd.Series] = {}
        comid_by_site: dict[str, object] = {}
        for s, gdf in nwm_df.groupby("site_no"):
            ser = gdf.set_index("datetime_utc")["value_cfs"]
            ser = ser.groupby(ser.index.floor("h")).max().sort_index()
            nwm_by_site[s] = ser
            comid_by_site[s] = int(gdf["comid"].iloc[0]) if "comid" in gdf.columns and pd.notna(gdf["comid"].iloc[0]) else None

        universe = sorted(stations_08c & set(nwm_by_site) & set(flow_by_site)
                          & set(usgs_fs.index) & set(nwm_fs.index))
        log.info("[%s] universe (08c ∩ NWM-op ∩ USGS flow ∩ USGS Q ∩ NWM Q): %d",
                 source_name, len(universe))

        n_ok = n_skip = 0
        for i, site in enumerate(universe, 1):
            usgs_ser = flow_by_site[site]
            nwm_ser  = nwm_by_site[site]
            if usgs_ser.empty or nwm_ser.empty:
                n_skip += 1
                continue
            ws = max(usgs_ser.index.min(), nwm_ser.index.min())
            we = min(usgs_ser.index.max(), nwm_ser.index.max())
            if pd.isna(ws) or pd.isna(we) or ws >= we:
                log.info("[%s][%d/%d] %s: no USGS∩NWM overlap window — skipping",
                         source_name, i, len(universe), site)
                n_skip += 1
                continue
            recs = analyse_station(site, comid_by_site.get(site), nwm_ser, usgs_ser,
                                   nwm_fs.loc[site], usgs_fs.loc[site], source_name, ws, we)
            all_records.extend(recs)
            if recs:
                n_ok += 1
                log.info("[%s][%d/%d] %s: %d flow-rp combos (win %s→%s)",
                         source_name, i, len(universe), site, len(recs), ws.date(), we.date())
        log.info("[%s] scored %d stations (%d skipped)", source_name, n_ok, n_skip)

    if not all_records:
        log.error("No results produced.")
        return

    out = pd.DataFrame(all_records)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s%s (%d rows, %d station×source pairs)",
             prefix, OUTPUT_KEY, len(out), out.groupby(["site_no", "source"]).ngroups)

    # Pooled skill per (source, flow_rp): sum the cells across stations, then score.
    pooled_rows: list[dict] = []
    for source_name, _ in NWM_SOURCES:
        for rp in FLOW_RPS:
            sub = out[(out.source == source_name) & (out.flow_rp_yr == rp)]
            if sub.empty:
                continue
            s = sub[["tp", "fp", "fn", "tn"]].sum()
            row = {"source": source_name, "flow_rp_yr": rp,
                   "n_stations": int(sub["site_no"].nunique()),
                   "tp": int(s.tp), "fp": int(s.fp), "fn": int(s.fn), "tn": int(s.tn)}
            row.update({k: (round(v, 4) if np.isfinite(v) else v)
                        for k, v in skill(s.tp, s.fp, s.fn, s.tn).items()})
            pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    log.info("Global skill (pooled across stations):\n%s",
             pooled[["source", "flow_rp_yr", "n_stations", "tp", "fp", "fn",
                     "pod", "far", "csi", "bias"]].to_string(index=False))

    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{METRICS_KEY}",
                           Body=pooled.to_csv(index=False).encode(), ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s (global skill, one row per source × flow_rp)",
             bucket, prefix, METRICS_KEY)

    if not args.no_figure:
        try:
            performance_diagram(pooled, bucket, prefix)
        except Exception as e:                                 # noqa: BLE001
            log.warning("Performance diagram failed (data already written): %s", e)
    log.info("Done.")


if __name__ == "__main__":
    main()
