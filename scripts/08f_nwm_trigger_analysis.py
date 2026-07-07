"""08f_nwm_trigger_analysis.py

NWM-streamflow version of the event-overlap confusion matrix (companion to 08c/08d).

Where 08c triggers on precipitation and 08d on INDOT's rain rule, this script asks:
**does an NWM streamflow exceedance anticipate a real (USGS-observed) flood?**  This
is the ungaged-basin question — in a basin with no gauge you would derive your flood
thresholds from the NWM retrospective climatology (04c) and trigger an inspection when
the operational NWM forecast/analysis exceeds them.

    TRIGGER : NWM operational streamflow ≥ Q(flow_rp)   with Q from a threshold source
    TRUTH   : USGS observed streamflow  ≥ Q(flow_rp)    with Q ALWAYS from 04b
              (USGS-annual-peak LP3, per_gauge_flow_stats.parquet)

Return periods are MATCHED: a Q10 trigger is scored against USGS-Q10 floods, Q50 vs
Q50, Q100 vs Q100 (FLOW_RPS = 10/50/100).

The trigger threshold is applied from TWO sources (thresh_src), so the effect of the
trigger-calibration climatology can be isolated from the operational flow itself:
    nwm_retro_04c   NWM-Retrospective-v3.0 LP3 quantiles (nwm_per_gauge_flow_stats.parquet)
                    — the ungaged calibration: alarm set from the long NWM record.
    usgs_peak_04b   USGS-annual-peak LP3 quantiles (per_gauge_flow_stats.parquet)
                    — same Q the truth uses, applied to the NWM flow (a "what if we
                    knew the real design flood" trigger).

Two NWM operational products are the trigger flow, so gauged vs ungaged skill compares:
    nwm_analysis_assim  Standard Analysis & Assimilation (nwm/analysis_assim.parquet)
                        — assimilates USGS gauges → best case at a gauged site.
    nwm_open_loop       Open-Loop A&A, no data assimilation (nwm/open_loop.parquet)
                        — the pure model → a proxy for skill in an UNGAGED basin.

→ 4 scenarios total = {A&A, open-loop} × {04c NWM Q, 04b USGS Q}, each over 3 RPs.

Event grouping, ±24 h linking, episodes and TP/FN/FP/TN are IDENTICAL to 08 (the
helpers are imported from it, not reimplemented), so results line up with 08c/08d.
For each (station × source × thresh_src × flow_rp):

    TP  a USGS flood event with an NWM-trigger-wet hour in [flood_start−24h, flood_end]
    FN  a USGS flood event with no such NWM trigger                          — missed
    FP  an NWM-trigger event with no USGS flood in [trig_start, trig_end+24h] — false alarm
    TN  each dry gap between consecutive wet events counts as one

Station universe = the 106 gauges used in 08c (event_confusion_matrix_tc.parquet),
intersected per source with: a valid 04b USGS threshold (truth), a USGS streamflow
record, and NWM operational data.  The 04c threshold is optional (its scenario is
skipped for a gauge with no 04c fit).  The common analysis window per gauge is the
USGS streamflow span ∩ the NWM operational span (so both the trigger and the truth are
defined over the same hours).

Units: NWM streamflow is m³/s → converted to cfs (×35.3147) to match the cfs Q columns
(04c is already in cfs); USGS observed is native cfs.

Writes:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_nwm.parquet  (per station × source × thresh_src × flow_rp)
    s3://<bucket>/<prefix>analysis/nwm_trigger_metrics.csv             (global skill, one row per source × thresh_src × flow_rp)
    s3://<bucket>/<prefix>analysis/figures/nwm_performance_diagram.{png,svg}  (2 panels: 04c vs 04b trigger threshold)

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

# Trigger-threshold sources (thresh_src): the Q the NWM flow is compared against.
THRESH_NWM_04C  = "nwm_retro_04c"    # 04c NWM-retrospective LP3 quantiles
THRESH_USGS_04B = "usgs_peak_04b"    # 04b USGS-annual-peak LP3 quantiles (same as truth)
THRESH_SOURCES  = [THRESH_NWM_04C, THRESH_USGS_04B]

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
    nwm_hourly: pd.Series,               # trigger flow: NWM operational (cfs), hourly
    usgs_hourly: pd.Series,              # truth flow:   USGS observed (cfs), hourly max
    trig_thresh_rows: dict[str, pd.Series | None],  # {thresh_src: trigger Q row | None}
    truth_row: pd.Series,                # 04b USGS Q — defines the truth flood events
    source: str,
    ws: pd.Timestamp,
    we: pd.Timestamp,
) -> list[dict]:
    grid = pd.date_range(ws, we, freq="1h", tz="UTC")
    trig = nwm_hourly.reindex(grid)
    pct_nwm_missing = round(100.0 * float(trig.isna().mean()), 2)
    flow = usgs_hourly.reindex(grid)
    n_common = len(grid)

    # Truth flood events depend only on the 04b threshold → compute once per RP and
    # reuse for every trigger-threshold source (they share this same truth).
    truth_by_rp: dict[int, tuple[float, list]] = {}
    for flow_rp in FLOW_RPS:
        q_truth = truth_row.get(f"Q{flow_rp}")
        if pd.isna(q_truth):
            continue
        truth_by_rp[flow_rp] = (
            float(q_truth),
            m.group_wet_events((flow >= float(q_truth)).fillna(False)),
        )

    out: list[dict] = []
    for thresh_src, qrow in trig_thresh_rows.items():
        if qrow is None:
            continue
        for flow_rp, (q_truth, flow_events) in truth_by_rp.items():
            q_trig = qrow.get(f"Q{flow_rp}")
            if pd.isna(q_trig):
                continue
            trig_events = m.group_wet_events((trig >= float(q_trig)).fillna(False))
            tp, fp, fn, tn = m.classify_overlap(trig_events, flow_events, ws, we)
            out.append({
                "site_no":          site_no,
                "comid":            comid,
                "source":           source,
                "thresh_src":       thresh_src,
                "flow_rp_yr":       flow_rp,
                "q_trigger_cfs":    round(float(q_trig), 1),
                "q_truth_cfs":      round(float(q_truth), 1),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_trigger_events": len(trig_events),
                "n_flow_events":    len(flow_events),
                "pct_nwm_missing":  pct_nwm_missing,
                "common_start":     ws,
                "common_end":       we,
                "n_common_hours":   n_common,
            })
    return out


# ---------- Performance (Roebber) diagram — 2 panels: 04c vs 04b trigger threshold ----------

_RP_COLORS  = {10: "#1a9850", 50: "#f46d43", 100: "#762a83"}
_SRC_MARKER = {"nwm_analysis_assim": ("*", 520), "nwm_open_loop": ("o", 170)}


def _perf_panel(ax, pooled_sub: pd.DataFrame, title: str):
    """Draw the CSI/bias background + one operating point per (product, RP)."""
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    cl = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=7.5, inline=True)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:                      # frequency-bias lines: POD = b·SR
        x = np.linspace(0, 1, 10)
        ax.plot(x, np.minimum(b * x, 1), ls="--", color="0.5", lw=0.7)
        if b <= 1: ax.text(1.008, b, f"{b:g}", fontsize=7, color="0.4", va="center")
        else:      ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7, color="0.4", ha="center")

    for source_name, (mk, ms) in _SRC_MARKER.items():
        for rp in FLOW_RPS:
            pr = pooled_sub[(pooled_sub.source == source_name) & (pooled_sub.flow_rp_yr == rp)]
            if pr.empty:
                continue
            r = pr.iloc[0]
            if not (np.isfinite(r.sr) and np.isfinite(r.pod)):
                continue
            ax.scatter([r.sr], [r.pod], marker=mk, s=ms, color=_RP_COLORS.get(rp, "grey"),
                       edgecolors="black", lw=1.1, zorder=7)
            ax.annotate(f"{r.csi:.2f}", (r.sr, r.pod), textcoords="offset points",
                        xytext=(6, 5), fontsize=7, color="0.15")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Success ratio  (1 − FAR)", fontsize=11)
    ax.set_ylabel("Probability of detection (POD)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return cf


def performance_diagram(pooled: pd.DataFrame, bucket: str, prefix: str) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = [(THRESH_NWM_04C, "Trigger threshold: NWM-retro Q (04c)"),
              (THRESH_USGS_04B, "Trigger threshold: USGS-peak Q (04b)")]
    cf = None
    for ax, (ts, title) in zip(axes, panels):
        cf = _perf_panel(ax, pooled[pooled.thresh_src == ts], title)

    rp_handles = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="black", ms=10, label=f"Q{rp}")
                  for rp, c in _RP_COLORS.items()]
    src_handles = [
        Line2D([0], [0], marker="*", ls="", mfc="0.6", mec="black", ms=14, label="A&A (gauged, w/ DA)"),
        Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="black", ms=9,  label="Open-loop (ungaged proxy)"),
    ]
    axes[0].legend(handles=rp_handles, loc="upper left", fontsize=8.5,
                   title="Return period", framealpha=0.95)
    axes[1].legend(handles=src_handles, loc="lower right", fontsize=8.5,
                   title="NWM product", framealpha=0.95)
    fig.colorbar(cf, ax=axes, label="Critical success index (CSI)", shrink=0.8, location="right")
    fig.suptitle("NWM streamflow flood trigger vs USGS-observed floods  (truth: USGS ≥ Q 04b)\n"
                 "left = trigger on NWM-retro Q (04c) · right = trigger on USGS-peak Q (04b) · labels = CSI",
                 fontsize=12.5, fontweight="bold")

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

        # Universe needs the 04b USGS Q (truth) + streamflow + NWM-op; the 04c NWM Q
        # is optional (its scenario is skipped per-gauge when absent).
        universe = sorted(stations_08c & set(nwm_by_site) & set(flow_by_site)
                          & set(usgs_fs.index))
        log.info("[%s] universe (08c ∩ NWM-op ∩ USGS flow ∩ USGS Q): %d",
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
            trig_rows = {
                THRESH_NWM_04C:  nwm_fs.loc[site] if site in nwm_fs.index else None,
                THRESH_USGS_04B: usgs_fs.loc[site],
            }
            recs = analyse_station(site, comid_by_site.get(site), nwm_ser, usgs_ser,
                                   trig_rows, usgs_fs.loc[site], source_name, ws, we)
            all_records.extend(recs)
            if recs:
                n_ok += 1
                log.info("[%s][%d/%d] %s: %d rows (win %s→%s)",
                         source_name, i, len(universe), site, len(recs), ws.date(), we.date())
        log.info("[%s] scored %d stations (%d skipped)", source_name, n_ok, n_skip)

    if not all_records:
        log.error("No results produced.")
        return

    out = pd.DataFrame(all_records)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s%s (%d rows, %d scenario groups: source×thresh_src)",
             prefix, OUTPUT_KEY, len(out),
             out.groupby(["source", "thresh_src"]).ngroups)

    # Pooled skill per (source, thresh_src, flow_rp): sum the cells, then score.
    pooled_rows: list[dict] = []
    for source_name, _ in NWM_SOURCES:
        for thresh_src in THRESH_SOURCES:
            for rp in FLOW_RPS:
                sub = out[(out.source == source_name) & (out.thresh_src == thresh_src)
                          & (out.flow_rp_yr == rp)]
                if sub.empty:
                    continue
                s = sub[["tp", "fp", "fn", "tn"]].sum()
                row = {"source": source_name, "thresh_src": thresh_src, "flow_rp_yr": rp,
                       "n_stations": int(sub["site_no"].nunique()),
                       "tp": int(s.tp), "fp": int(s.fp), "fn": int(s.fn), "tn": int(s.tn)}
                row.update({k: (round(v, 4) if np.isfinite(v) else v)
                            for k, v in skill(s.tp, s.fp, s.fn, s.tn).items()})
                pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    log.info("Global skill (pooled across stations):\n%s",
             pooled[["source", "thresh_src", "flow_rp_yr", "n_stations", "tp", "fp", "fn",
                     "pod", "far", "csi", "bias"]].to_string(index=False))

    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{METRICS_KEY}",
                           Body=pooled.to_csv(index=False).encode(), ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s (global skill, one row per source × thresh_src × flow_rp)",
             bucket, prefix, METRICS_KEY)

    if not args.no_figure:
        try:
            performance_diagram(pooled, bucket, prefix)
        except Exception as e:                                 # noqa: BLE001
            log.warning("Performance diagram failed (data already written): %s", e)
    log.info("Done.")


if __name__ == "__main__":
    main()
