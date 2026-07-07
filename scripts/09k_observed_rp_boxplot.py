"""09k_observed_rp_boxplot.py

"When a trigger fires, how big was the flood that actually happened?"

For every alarm event of each trigger, take the PEAK observed USGS streamflow in the
event window and convert it to its return period via the at-site USGS LP3 (04b).  Pool
those observed RPs across all alarms and all stations → one box per trigger.  This is
the magnitude-aware companion to the categorical skill diagrams: a trigger whose box
sits high fires on genuine floods; one whose box sits near RP≈1 mostly fires on nothing.
It also reframes "false alarm" — an alarm that coincides with a Q8 flood is a justified
inspection even though the categorical score (thresholded at Q10) calls it an FP.

Six triggers (boxes):
    INDOT       nearest ISD/GHCNh gauge, trailing 24-h >= 2.5 in           (08d rule)
    MRMS-P10    nearest MRMS pixel, Tc-hour accumulation >= Atlas14(Tc, P10)  (08c)
    MRMS-P50    "                                       >= Atlas14(Tc, P50)
    MRMS-P100   "                                       >= Atlas14(Tc, P100)
    NWM A&A     analysis_assim streamflow >= Q10 (USGS-peak 04b)            (08f)
    NWM OpenL.  open-loop     streamflow >= Q10 (NWM-retro 04c)

Observed RP: peak USGS flow in [alarm_start, alarm_end + 24 h] mapped to a return period
by log-log interpolation of the station's 04b Q2..Q500 quantiles (extrapolated, clipped
to [1, 1000]).  Event grouping (±24 h merge) is 08's group_wet_events.

Writes:
    s3://<bucket>/<prefix>analysis/figures/observed_rp_boxplot.{png,svg}
    s3://<bucket>/<prefix>analysis/observed_rp_per_alarm.csv   (trigger, site_no, observed_rp)

Usage:
    python scripts/09k_observed_rp_boxplot.py
"""
from __future__ import annotations

import importlib.util
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config, s3_client, write_bytes_to_s3


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    import sys; sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

m     = _load("trigger_analysis_08", "08_trigger_analysis.py")
m08c  = _load("tc_trigger_08c",      "08c_tc_trigger_analysis.py")
m08d  = _load("indot_08d",           "08d_indot_trigger_analysis.py")
m08f  = _load("nwm_08f",             "08f_nwm_trigger_analysis.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09k_obs_rp")

RP_PTS      = [2, 5, 10, 25, 50, 100, 200, 500]
OBS_FWD_H   = 24                       # look for the flood in [alarm_start, alarm_end + 24 h]
MRMS_PRPS   = [10, 50, 100]            # MRMS ARI boxes (matched to Q10/Q50/Q100 flood targets)
INV_KEY     = "stations/indiana_streamflow_sites.parquet"
FS_04B_KEY  = "flow_stats/per_gauge_flow_stats.parquet"       # USGS-peak LP3 (RP conversion + A&A alarm)
FS_04C_KEY  = "flow_stats/nwm_per_gauge_flow_stats.parquet"   # NWM-retro LP3 (open-loop alarm)
TC_KEY      = "analysis/event_confusion_matrix_tc.parquet"
CSV_KEY     = "analysis/observed_rp_per_alarm.csv"
FIG_KEY     = "analysis/figures/observed_rp_boxplot"

# (trigger label, x-tick, colour)
BOXES = [
    ("INDOT",      "INDOT\n2.5 in / 24 h", "#8c564b"),
    ("MRMS-P10",   "MRMS P10\n@ Tc",       "#9ecae1"),
    ("MRMS-P50",   "MRMS P50\n@ Tc",       "#4292c6"),
    ("MRMS-P100",  "MRMS P100\n@ Tc",      "#08519c"),
    ("NWM-AA",     "NWM A&A\n≥ Q10 (04b)", "#e6550d"),
    ("NWM-OL",     "NWM OL\n≥ Q10 (04c)",  "#fdae6b"),
]


# ── Flow → return period (at-site 04b LP3, log-log) ───────────────────────────

def rp_of_flow(qrow: pd.Series, flow: float) -> float:
    pts = sorted((float(qrow[f"Q{rp}"]), rp) for rp in RP_PTS
                 if pd.notna(qrow.get(f"Q{rp}")) and qrow[f"Q{rp}"] > 0)
    if len(pts) < 2 or flow <= 0:
        return np.nan
    lq = np.log([q for q, _ in pts]); lr = np.log([r for _, r in pts]); x = np.log(flow)
    if x <= lq[0]:
        sl = (lr[1] - lr[0]) / (lq[1] - lq[0]);   y = lr[0] + sl * (x - lq[0])
    elif x >= lq[-1]:
        sl = (lr[-1] - lr[-2]) / (lq[-1] - lq[-2]); y = lr[-1] + sl * (x - lq[-1])
    else:
        y = float(np.interp(x, lq, lr))
    return float(np.clip(np.exp(y), 1.0, 1000.0))


def events_to_rps(alarm_wet: pd.Series, flow: pd.Series, b04_row: pd.Series) -> list[float]:
    """Observed peak RP for each alarm event: peak USGS flow in [start, end+24h] → 04b RP."""
    out: list[float] = []
    fwd = pd.Timedelta(hours=OBS_FWD_H)
    for s, e in m.group_wet_events(alarm_wet):
        win = flow[(flow.index >= s) & (flow.index <= e + fwd)]
        if win.empty:
            continue
        pk = float(win.max())
        if not np.isfinite(pk):
            continue
        rp = rp_of_flow(b04_row, pk)
        if np.isfinite(rp):
            out.append(rp)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    log.info("Loading streamflow, thresholds, station coords, 08c universe...")
    streamflow = m.load_streamflow(bucket, prefix)
    flow_by_site = {s: g.set_index("datetime_utc")["value_cfs"].sort_index()
                    for s, g in streamflow.groupby("site_no")}
    b04 = m._read_parquet_s3(bucket, f"{prefix}{FS_04B_KEY}"); b04["site_no"] = b04["site_no"].astype(str)
    b04 = b04.drop_duplicates("site_no").set_index("site_no")
    c04 = m._read_parquet_s3(bucket, f"{prefix}{FS_04C_KEY}"); c04["site_no"] = c04["site_no"].astype(str)
    c04 = c04.drop_duplicates("site_no").set_index("site_no")
    inv = m._read_parquet_s3(bucket, f"{prefix}{INV_KEY}", columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    coords = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).set_index("site_no")
    tc = m._read_parquet_s3(bucket, f"{prefix}{TC_KEY}", columns=["site_no"])
    stations = sorted(set(tc["site_no"].astype(str)))
    log.info("08c analysis stations: %d", len(stations))

    rp_by_trigger: dict[str, list[float]] = {b[0]: [] for b in BOXES}

    # ── INDOT — nearest ISD/GHCNh gauge, 24-h >= 2.5 in ──────────────────────
    log.info("INDOT trigger (loading ISD/GHCNh, assigning nearest stations)...")
    qual, hourly_by_sid = m08d.load_and_qualify(bucket, prefix)
    for s in stations:
        flow = flow_by_site.get(s)
        if flow is None or flow.empty or s not in coords.index or s not in b04.index:
            continue
        c = coords.loc[s]
        assigned = m08d.assign_station(float(c["dec_lat_va"]), float(c["dec_long_va"]),
                                       flow.index.min(), flow.index.max(), qual, hourly_by_sid)
        if assigned is None:
            continue
        _, _, ws, we, precip_grid, _ = assigned
        wet = (precip_grid.rolling(m08d.DURATION_HR, min_periods=m08d.DURATION_HR).sum()
               >= m08d.PRECIP_THRESH_IN).fillna(False)
        rp_by_trigger["INDOT"].extend(events_to_rps(wet, flow, b04.loc[s]))
    log.info("  INDOT alarms: %d", len(rp_by_trigger["INDOT"]))

    # ── MRMS — nearest pixel, Tc accumulation vs Atlas-14 (P10/P50/P100) ─────
    log.info("MRMS trigger (loading nearest-pixel MRMS, Atlas-14, Tc)...")
    atlas14 = m.load_atlas14(bucket, prefix)
    tc_by_site = m08c.load_tc(bucket, prefix)
    mrms = m.load_mrms_nearest(bucket, prefix, product_key)
    mrms_start = mrms.groupby("site_no")["datetime_utc"].min()
    mrms_end   = mrms.groupby("site_no")["datetime_utc"].max()
    for s in stations:
        flow = flow_by_site.get(s)
        if (flow is None or flow.empty or s not in b04.index or s not in tc_by_site
                or s not in mrms_start.index):
            continue
        ws = max(flow.index.min(), mrms_start[s]); we = min(flow.index.max(), mrms_end[s])
        if pd.isna(ws) or pd.isna(we) or ws >= we:
            continue
        grid = pd.date_range(ws, we, freq="1h", tz="UTC")
        precip = mrms[mrms["site_no"] == s].set_index("datetime_utc")["precip_in"].sort_index()
        precip = precip.reindex(grid).fillna(0.0)
        a14_site = atlas14[atlas14["site_no"] == s]
        d_tc = max(1, int(round(float(tc_by_site[s]))))
        rolling = precip.rolling(d_tc, min_periods=d_tc).sum()
        for prp in MRMS_PRPS:
            depth = m08c.depth_at_duration(a14_site, prp, d_tc)
            if not np.isfinite(depth) or depth <= 0:
                continue
            wet = (rolling >= depth).fillna(False)
            rp_by_trigger[f"MRMS-P{prp}"].extend(events_to_rps(wet, flow, b04.loc[s]))
    for prp in MRMS_PRPS:
        log.info("  MRMS-P%d alarms: %d", prp, len(rp_by_trigger[f"MRMS-P{prp}"]))

    # ── NWM — A&A (>= 04b Q10) and open-loop (>= 04c Q10) ────────────────────
    log.info("NWM triggers (A&A, open-loop)...")
    for trig, key, qtab in [("NWM-AA", "nwm/analysis_assim.parquet", b04),
                            ("NWM-OL", "nwm/open_loop.parquet",       c04)]:
        try:
            nwm_df = m08f.load_nwm_operational(bucket, prefix, key)
        except Exception as e:                                   # noqa: BLE001
            log.warning("  %s load failed: %s", trig, e)
            continue
        nwm_df = nwm_df[nwm_df["site_no"].isin(stations)]
        for s, gdf in nwm_df.groupby("site_no"):
            flow = flow_by_site.get(s)
            if flow is None or flow.empty or s not in b04.index or s not in qtab.index:
                continue
            thr = qtab.loc[s, "Q10"]
            if pd.isna(thr):
                continue
            ser = gdf.set_index("datetime_utc")["value_cfs"]
            ser = ser.groupby(ser.index.floor("h")).max().sort_index()
            ws = max(flow.index.min(), ser.index.min()); we = min(flow.index.max(), ser.index.max())
            if pd.isna(ws) or pd.isna(we) or ws >= we:
                continue
            grid = pd.date_range(ws, we, freq="1h", tz="UTC")
            wet = (ser.reindex(grid) >= float(thr)).fillna(False)
            rp_by_trigger[trig].extend(events_to_rps(wet, flow, b04.loc[s]))
        log.info("  %s alarms: %d", trig, len(rp_by_trigger[trig]))

    # ── Outputs ───────────────────────────────────────────────────────────────
    rows = [{"trigger": t, "observed_rp": round(rp, 2)} for t, rps in rp_by_trigger.items() for rp in rps]
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{CSV_KEY}",
                           Body=pd.DataFrame(rows).to_csv(index=False).encode(), ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s", bucket, prefix, CSV_KEY)

    make_figure(rp_by_trigger, bucket, prefix)
    log.info("Done.")


def make_figure(rp_by_trigger: dict[str, list[float]], bucket: str, prefix: str) -> None:
    keys   = [b[0] for b in BOXES]
    ticks  = [b[1] for b in BOXES]
    colors = [b[2] for b in BOXES]
    data   = [rp_by_trigger[k] if rp_by_trigger[k] else [np.nan] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    widths=0.62, medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
    ax.set_xticks(range(1, len(ticks) + 1))
    ax.set_xticklabels(ticks)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("0.3")

    ax.set_yscale("log")
    ax.set_ylabel("Observed USGS return period when the trigger fires", fontsize=13)
    for rp, lab in [(2, "Q2"), (10, "Q10"), (50, "Q50"), (100, "Q100")]:
        ax.axhline(rp, color="0.6", ls=":", lw=0.9)
        ax.text(0.35, rp, lab, fontsize=9, color="0.4", va="center", ha="right")
    # n_alarms above each box
    for i, k in enumerate(keys, 1):
        n = len(rp_by_trigger[k])
        ax.text(i, ax.get_ylim()[1] * 0.9, f"n={n}", ha="center", fontsize=9, color="0.25")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("Observed flood severity when each trigger fires\n"
                 "(box = distribution of observed USGS return period per alarm; higher = fires on real floods)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=170, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
