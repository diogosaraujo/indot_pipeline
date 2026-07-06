"""09e_confusion_case_examples.py

A four-panel explainer of the TP / FP / FN / TN outcomes counted by 08c.

For ONE station, the fixed-Tc event-overlap logic of 08c is replayed (reusing
08's helpers so the classification is byte-for-byte identical) and one real
example of each outcome is drawn as a precipitation-over-hydrograph panel:

    top    hourly MRMS precip (bars) + the trailing D-hour accumulation (line),
           with the Atlas-14 depth threshold P{precip_rp} dashed in red
    bottom hourly streamflow with the flood threshold Q{flow_rp} dashed in navy
    shade  the classification window (TP green / FP orange / FN red)

    TP  flood with the rain threshold crossed <=24 h before it        (a hit)
    FP  rain threshold crossed, no flood within +24 h                 (false alarm)
    FN  flood with no rain-threshold crossing in the prior 24 h       (a miss)
    TN  a quiet stretch — neither threshold crossed

Thresholds (D, Atlas-14 depth, Q) are read straight from the 08c output, so the
picture matches the analysis exactly.  The example station is auto-selected as
the shortest-Tc gauge that has >=1 of TP, FP and FN at the chosen return-period
pair (crisp hyetograph), or pass --site to override.

Writes (S3 only):
    s3://<bucket>/<prefix>analysis/figures/confusion_case_examples.{png,svg}

Usage:
    python scripts/09e_confusion_case_examples.py
    python scripts/09e_confusion_case_examples.py --site 03357350 --precip-rp 5 --flow-rp 10
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("09e_cases")

# Reuse 08's loaders + event helpers so keys / logic match the real 08c run.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TC_KEY  = "analysis/event_confusion_matrix_tc.parquet"
FIG_KEY = "analysis/figures/confusion_case_examples"

CLR = {"TP": "#2e7d32", "FP": "#ef6c00", "FN": "#c62828", "TN": "#455a64"}
DEF = {
    "TP": "Flood occurred and the rain threshold was crossed within 24 h before it — a hit.",
    "FP": "Rain threshold was crossed but no flood followed within 24 h — a false alarm.",
    "FN": "Flood occurred but the rain threshold was never crossed in the prior 24 h — a miss.",
    "TN": "Neither threshold was crossed — a quiet stretch correctly left alone.",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def read_per_gauge_flow_hourly(bucket: str, prefix: str, site_no: str) -> pd.Series:
    """Hourly-max streamflow for one gauge (matches 08's load_streamflow binning)."""
    key = f"{prefix}streamflow/instantaneous/per_gauge/{site_no}.parquet"
    df = m._read_parquet_s3(bucket, key)
    dtc = next(c for c in df.columns if c.lower() in ("datetime", "datetime_utc", "time"))
    df[dtc] = pd.to_datetime(df[dtc], utc=True)
    df["value_cfs"] = pd.to_numeric(df["value_cfs"], errors="coerce")
    return df.set_index(dtc)["value_cfs"].resample("1h").max()


def choose_site(tc: pd.DataFrame, precip_rp: int, flow_rp: int) -> str:
    """Shortest-Tc station that has >=1 TP, FP and FN at (precip_rp, flow_rp)."""
    s = tc[(tc.source == "nearest") & (tc.precip_rp_yr == precip_rp)
           & (tc.flow_rp_yr == flow_rp)]
    good = s[(s.tp > 0) & (s.fp > 0) & (s.fn > 0)]
    if good.empty:
        raise SystemExit(f"No station has TP&FP&FN at precip_rp={precip_rp}, flow_rp={flow_rp}")
    return good.sort_values("tc_hr").iloc[0]["site_no"]


# ── Episode reconstruction (identical to 08c) ─────────────────────────────────

def reconstruct(bucket, prefix, product_key, site_no, precip_rp, flow_rp):
    tc = m._read_parquet_s3(bucket, f"{prefix}{TC_KEY}")
    tc["site_no"] = tc["site_no"].astype(str)
    row = tc[(tc.site_no == site_no) & (tc.precip_rp_yr == precip_rp)
             & (tc.flow_rp_yr == flow_rp) & (tc.source == "nearest")].iloc[0]
    D = int(row.duration_hr); p_thr = float(row.precip_depth_in)
    cs = pd.Timestamp(row.common_start); ce = pd.Timestamp(row.common_end)
    if cs.tz is None: cs = cs.tz_localize("UTC")
    if ce.tz is None: ce = ce.tz_localize("UTC")

    flow_stats = m.load_flow_stats(bucket, prefix)
    q_thr = float(flow_stats[flow_stats.site_no == site_no].iloc[0][f"Q{flow_rp}"])

    mrms = m.load_mrms_nearest(bucket, prefix, product_key)
    p_raw = mrms[mrms.site_no == site_no].set_index("datetime_utc")["precip_in"].sort_index()
    f_hr = read_per_gauge_flow_hourly(bucket, prefix, site_no)

    grid = pd.date_range(cs, ce, freq="1h", tz="UTC")
    precip = p_raw.reindex(grid).fillna(0.0)
    flow = f_hr.reindex(grid)
    roll = precip.rolling(D, min_periods=D).sum()
    p_wet = (roll >= p_thr).fillna(False)
    f_wet = (flow >= q_thr).fillna(False)
    eps = m.build_episodes(m.group_wet_events(p_wet), m.group_wet_events(f_wet))

    # Sanity: counts must equal what 08c stored.
    tn = m.count_dry_periods(m.merge_spans([(e["start"], e["end"]) for e in eps]), cs, ce)
    got = (sum(e["klass"] == "TP" for e in eps), sum(e["klass"] == "FP" for e in eps),
           sum(e["klass"] == "FN" for e in eps), tn)
    exp = (int(row.tp), int(row.fp), int(row.fn), int(row.tn))
    if got != exp:
        log.warning("Reconstructed counts %s != stored %s (proceeding)", got, exp)
    else:
        log.info("Reconstructed counts match 08c exactly: TP=%d FP=%d FN=%d TN=%d", *got)

    return dict(row=row, D=D, p_thr=p_thr, q_thr=q_thr, cs=cs, ce=ce,
                grid=grid, precip=precip, roll=roll, flow=flow, eps=eps)


def pick_examples(st: dict) -> dict[str, tuple]:
    """Choose one clear (view_start, view_end, episode|None) per class."""
    eps, flow, roll = st["eps"], st["flow"], st["roll"]
    q_thr, p_thr = st["q_thr"], st["p_thr"]

    def peakQ(e): return float(flow.loc[e["start"]:e["end"]].max())
    def peakP(e): return float(roll.loc[e["start"]:e["end"]].max())

    def best(klass, key):
        cand = [e for e in eps if e["klass"] == klass]
        return max(cand, key=key) if cand else None

    tp = best("TP", peakQ)                       # most convincing flood
    fn = best("FN", peakQ)                        # biggest missed flood
    # FP with the largest (still sub-threshold) flow response reads best.
    fp = best("FP", lambda e: peakQ(e) if peakQ(e) < q_thr else -1) or best("FP", peakP)

    def pad(e, days=3):
        return (e["start"] - pd.Timedelta(days=days), e["end"] + pd.Timedelta(days=days), e)

    out = {}
    if tp: out["TP"] = pad(tp)
    if fp: out["FP"] = pad(fp)
    if fn: out["FN"] = pad(fn)

    # TN: a calm ~12-day window far from any episode, centred on modest activity.
    spans = [(e["start"], e["end"]) for e in eps]
    def near(t0, t1, p=pd.Timedelta(days=3)):
        return any(s - p <= t1 and t0 <= e + p for s, e in spans)
    cs, ce = st["cs"], st["ce"]
    chosen = None
    for t0 in pd.date_range(cs + pd.Timedelta(days=10), ce - pd.Timedelta(days=14), freq="5D"):
        t1 = t0 + pd.Timedelta(days=12)
        if near(t0, t1):
            continue
        pa = roll.loc[t0:t1].max(); pf = flow.loc[t0:t1].max()
        if not (np.isfinite(pa) and np.isfinite(pf)):
            continue
        if 0.8 <= pa <= p_thr * 0.75 and 150 <= pf <= q_thr * 0.55:
            score = pa / p_thr + pf / q_thr
            if chosen is None or score > chosen[0]:
                chosen = (score, t0, t1)
    if chosen:
        peak_t = flow.loc[chosen[1]:chosen[2]].idxmax()
        out["TN"] = (peak_t - pd.Timedelta(days=6), peak_t + pd.Timedelta(days=6), None)
    return out


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_case(fig, outer, st, klass, view_s, view_e, episode, precip_rp, flow_rp):
    D, p_thr, q_thr = st["D"], st["p_thr"], st["q_thr"]
    grid, precip, roll, flow = st["grid"], st["precip"], st["roll"], st["flow"]
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, height_ratios=[1.0, 1.7], hspace=0.08)
    axp = fig.add_subplot(inner[0]); axf = fig.add_subplot(inner[1], sharex=axp)
    vs, ve = pd.Timestamp(view_s), pd.Timestamp(view_e)
    g = grid[(grid >= vs) & (grid <= ve)]
    pv, rv, fv = precip.reindex(g), roll.reindex(g), flow.reindex(g)

    if episode is not None:
        for ax in (axp, axf):
            ax.axvspan(episode["start"], episode["end"], color=CLR[klass], alpha=0.12, zorder=0)

    # precip (top)
    axp.bar(g, pv.values, width=0.03, color="#90a4ae", alpha=0.9, zorder=2)
    axp.plot(g, rv.values, color="#1565c0", lw=1.3, zorder=3)
    wetp = (rv >= p_thr).fillna(False)
    if wetp.any():
        axp.plot(g[wetp.values], rv.values[wetp.values], ".", color="#d32f2f", ms=4, zorder=4)
    axp.axhline(p_thr, color="#d32f2f", ls="--", lw=1.1, zorder=3)
    axp.text(0.012, 0.9, f"Atlas-14 P{precip_rp} = {p_thr:.2f} in", transform=axp.transAxes,
             fontsize=7.2, color="#d32f2f", va="top")
    axp.set_ylabel("Precip (in)", fontsize=8)
    rmax = float(np.nanmax(rv.values)) if np.isfinite(np.nanmax(rv.values)) else p_thr
    axp.set_ylim(0, max(p_thr * 1.25, rmax * 1.1))
    axp.tick_params(labelbottom=False, labelsize=7)
    axp.set_title(f"{klass}   —   {DEF[klass]}", fontsize=8.6, color=CLR[klass],
                  fontweight="bold", loc="left", pad=4)

    # hydrograph (bottom)
    axf.plot(g, fv.values, color="#1b5e76", lw=1.1, zorder=3)
    wetf = (fv >= q_thr).fillna(False)
    if wetf.any():
        axf.fill_between(g, 0, fv.values, where=wetf.values, color="#1b5e76", alpha=0.25, zorder=2)
    axf.axhline(q_thr, color="#0d2c54", ls="--", lw=1.1, zorder=3)
    axf.text(0.012, 0.93, f"Q{flow_rp} = {q_thr:,.0f} cfs", transform=axf.transAxes,
             fontsize=7.2, color="#0d2c54", va="top")
    axf.set_ylabel("Flow (cfs)", fontsize=8)
    fmax = float(np.nanmax(fv.values)) if np.isfinite(np.nanmax(fv.values)) else q_thr
    axf.set_ylim(0, max(q_thr, fmax) * 1.15)
    axf.tick_params(labelsize=7)
    axf.set_xlim(vs, ve)
    axf.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lb in axf.get_xticklabels():
        lb.set_fontsize(6.6)
    axf.text(0.985, 0.06, f"{vs.date()}  ->  {ve.date()}", transform=axf.transAxes,
             fontsize=6.6, color="0.35", ha="right")


def build_figure(st, examples, site_no, precip_rp, flow_rp):
    row = st["row"]
    fig = plt.figure(figsize=(13.5, 8.4))
    outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.16,
                             left=0.055, right=0.985, top=0.9, bottom=0.06)
    slots = {"TP": outer[0, 0], "FP": outer[0, 1], "FN": outer[1, 0], "TN": outer[1, 1]}
    for klass, cell in slots.items():
        if klass not in examples:
            continue
        vs, ve, ep = examples[klass]
        draw_case(fig, cell, st, klass, vs, ve, ep, precip_rp, flow_rp)

    fig.suptitle(
        f"The four event-overlap outcomes in 08c  —  USGS {site_no}, nearest MRMS pixel"
        f"   (T$_c$≈{row.tc_hr:.0f} h → D={st['D']} h,  precip RP={precip_rp} yr,  flow RP={flow_rp} yr)",
        fontsize=12, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             "Precip wet = trailing D-hour accumulation ≥ Atlas-14 depth   •   "
             "Flow wet = hourly streamflow ≥ Q$_{RP}$   •   "
             "rain may lead the flood by up to 24 h; wet runs <24 h apart merge into one event",
             ha="center", fontsize=8.2, color="0.3")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None, help="station to illustrate (default: auto)")
    ap.add_argument("--precip-rp", type=int, default=5)
    ap.add_argument("--flow-rp", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    tc = m._read_parquet_s3(bucket, f"{prefix}{TC_KEY}")
    tc["site_no"] = tc["site_no"].astype(str)
    site_no = str(args.site) if args.site else choose_site(tc, args.precip_rp, args.flow_rp)
    log.info("Illustrating station %s (precip_rp=%d, flow_rp=%d)",
             site_no, args.precip_rp, args.flow_rp)

    st = reconstruct(bucket, prefix, product_key, site_no, args.precip_rp, args.flow_rp)
    examples = pick_examples(st)
    missing = [k for k in ("TP", "FP", "FN", "TN") if k not in examples]
    if missing:
        log.warning("No example found for: %s", ", ".join(missing))
    fig = build_figure(st, examples, site_no, args.precip_rp, args.flow_rp)

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=170, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
