"""09l_station_mrms_rp_violin.py

Paired observed-return-period violins comparing the STATION-gauge trigger against
the nearest-pixel MRMS trigger, across the 10 Atlas 14 ARI thresholds (P1..P1000).

Both triggers use the SAME Atlas 14 POINT depth at the station's Kirpich Tc; only
the precipitation field differs — MRMS nearest pixel (08c) vs. the nearest
qualifying ISD/GHCNh gauge (08d/08h pairing). For every alarm event of each
(source, ARI) the PEAK observed USGS flow in [alarm_start, alarm_end+24h] is
converted to a return period via the at-site 04b LP3 (as in 09k). Pooled over all
106 stations → one violin per (source, ARI).

Figure (6.5 x 4 in): x-axis = P1..P1000; at each ARI two violins side by side
(station, MRMS) with the median highlighted; a line connects the station medians
and another the MRMS medians; the alarm count (n) is printed above each violin.

Writes:
    s3://<bucket>/<prefix>analysis/figures/station_vs_mrms_rp_violin.{png,pdf}
    s3://<bucket>/<prefix>analysis/station_vs_mrms_rp_per_alarm.csv
"""
from __future__ import annotations

import importlib.util
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import load_config, s3_client, write_bytes_to_s3


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

m    = _load("trigger_analysis_08", "08_trigger_analysis.py")
m08c = _load("tc_trigger_08c",      "08c_tc_trigger_analysis.py")
m08d = _load("indot_08d",           "08d_indot_trigger_analysis.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("09l_pair_rp")

ARIS       = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
RP_PTS     = [2, 5, 10, 25, 50, 100, 200, 500]
OBS_FWD_H  = 24
DISP_CAP   = 500.0
STA_COLOR  = "#d94801"     # station gauge
MRMS_COLOR = "#2171b5"     # nearest-pixel MRMS
INV_KEY    = "stations/indiana_streamflow_sites.parquet"
FS_04B_KEY = "flow_stats/per_gauge_flow_stats.parquet"
TC_KEY     = "analysis/event_confusion_matrix_tc.parquet"
CSV_KEY    = "analysis/station_vs_mrms_rp_per_alarm.csv"
FIG_KEY    = "analysis/figures/station_vs_mrms_rp_violin"


# ── Flow → return period (at-site 04b LP3, log-log) — from 09k ─────────────────

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

    log.info("Loading streamflow, thresholds, coords, universe...")
    streamflow = m.load_streamflow(bucket, prefix)
    flow_by_site = {s: g.set_index("datetime_utc")["value_cfs"].sort_index()
                    for s, g in streamflow.groupby("site_no")}
    b04 = m._read_parquet_s3(bucket, f"{prefix}{FS_04B_KEY}"); b04["site_no"] = b04["site_no"].astype(str)
    b04 = b04.drop_duplicates("site_no").set_index("site_no")
    inv = m._read_parquet_s3(bucket, f"{prefix}{INV_KEY}", columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    coords = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).set_index("site_no")
    stations = sorted(set(m._read_parquet_s3(bucket, f"{prefix}{TC_KEY}",
                                             columns=["site_no"])["site_no"].astype(str)))
    log.info("Universe: %d stations", len(stations))

    atlas14 = m.load_atlas14(bucket, prefix)                          # streamgage climatology (MRMS)
    atlas14_st = m._read_parquet_s3(bucket, f"{prefix}atlas14/precipitation_frequency_stations.parquet")
    atlas14_st["station_id"] = atlas14_st["station_id"].astype(str)   # station climatology (gauge trigger)
    tc_by_site = m08c.load_tc(bucket, prefix)
    mrms = m.load_mrms_nearest(bucket, prefix, product_key)
    mrms_start = mrms.groupby("site_no")["datetime_utc"].min()
    mrms_end   = mrms.groupby("site_no")["datetime_utc"].max()
    log.info("Loading + qualifying precip stations (ISD + GHCNh)...")
    qual, hourly_by_sid = m08d.load_and_qualify(bucket, prefix)

    rp = {"station": {p: [] for p in ARIS}, "mrms": {p: [] for p in ARIS}}

    for i, s in enumerate(stations, 1):
        flow = flow_by_site.get(s)
        if flow is None or flow.empty or s not in b04.index or s not in tc_by_site:
            continue
        b04_row = b04.loc[s]
        d_tc = max(1, int(round(float(tc_by_site[s]))))
        a14_site = atlas14[atlas14["site_no"] == s]
        if a14_site.empty:
            continue
        depths_mrms = {p: m08c.depth_at_duration(a14_site, p, d_tc) for p in ARIS}
        depths_mrms = {p: (d if np.isfinite(d) and d > 0 else None) for p, d in depths_mrms.items()}

        # MRMS nearest pixel (flow ∩ MRMS window) — streamgage climatology
        if s in mrms_start.index:
            ws = max(flow.index.min(), mrms_start[s]); we = min(flow.index.max(), mrms_end[s])
            if not (pd.isna(ws) or pd.isna(we) or ws >= we):
                grid = pd.date_range(ws, we, freq="1h", tz="UTC")
                precip = (mrms[mrms["site_no"] == s].set_index("datetime_utc")["precip_in"]
                          .sort_index().reindex(grid).fillna(0.0))
                roll = precip.rolling(d_tc, min_periods=d_tc).sum()
                for p in ARIS:
                    if depths_mrms[p] is None:
                        continue
                    wet = (roll >= depths_mrms[p]).fillna(False)
                    rp["mrms"][p].extend(events_to_rps(wet, flow, b04_row))

        # Nearest qualifying station gauge (08d pairing) — the station's OWN climatology
        if s in coords.index:
            cc = coords.loc[s]
            assigned = m08d.assign_station(float(cc["dec_lat_va"]), float(cc["dec_long_va"]),
                                           flow.index.min(), flow.index.max(), qual, hourly_by_sid)
            if assigned is not None:
                sid = assigned[0]
                a14_sta = atlas14_st[atlas14_st["station_id"] == sid]
                if not a14_sta.empty:
                    depths_sta = {p: m08c.depth_at_duration(a14_sta, p, d_tc) for p in ARIS}
                    depths_sta = {p: (d if np.isfinite(d) and d > 0 else None) for p, d in depths_sta.items()}
                    roll = assigned[4].rolling(d_tc, min_periods=d_tc).sum()
                    for p in ARIS:
                        if depths_sta[p] is None:
                            continue
                        wet = (roll >= depths_sta[p]).fillna(False)
                        rp["station"][p].extend(events_to_rps(wet, flow, b04_row))
        if i % 20 == 0:
            log.info("  %d/%d stations", i, len(stations))

    rows = [{"source": src, "ari_yr": p, "observed_rp": round(v, 2)}
            for src in rp for p, vals in rp[src].items() for v in vals]
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{CSV_KEY}",
                           Body=pd.DataFrame(rows).to_csv(index=False).encode(),
                           ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s", bucket, prefix, CSV_KEY)

    make_figure(rp, bucket, prefix)
    log.info("Done.")


def _violins(ax, data, positions, color, width):
    parts = ax.violinplot(data, positions=positions, widths=width,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.7)
        pc.set_edgecolor("0.3"); pc.set_linewidth(0.4)
    parts["cmedians"].set_edgecolor("black"); parts["cmedians"].set_linewidth(1.2)


def make_figure(rp, bucket, prefix) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.85, bottom=0.135)

    off, width = 0.21, 0.36
    for src, color, dx in [("station", STA_COLOR, -off), ("mrms", MRMS_COLOR, +off)]:
        vpos, vdata, med_x, med_y = [], [], [], []
        for i, p in enumerate(ARIS):
            rps = np.asarray(rp[src][p], dtype=float)
            if rps.size >= 2:
                d = np.log10(np.minimum(rps, DISP_CAP))
                vpos.append(i + dx); vdata.append(d)
                med_x.append(i + dx); med_y.append(float(np.median(d)))
            elif rps.size == 1:
                y = np.log10(min(rps[0], DISP_CAP))
                ax.scatter([i + dx], [y], color=color, s=10, zorder=6)
                med_x.append(i + dx); med_y.append(y)
            # n above each violin (rotated so 20 labels don't collide)
            ax.text(i + dx, 1.01, f"n={rps.size}", transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=6, color=color, clip_on=False)
        if vdata:
            _violins(ax, vdata, vpos, color, width)
        ax.plot(med_x, med_y, color=color, lw=1.5, marker="o", ms=3, zorder=7,
                label="Precipitation Station" if src == "station" else "Nearest-pixel MRMS")

    ytick_rp = [2, 5, 10, 50, 100, 500]
    ax.set_yticks(np.log10(ytick_rp)); ax.set_yticklabels([str(v) for v in ytick_rp])
    ax.set_ylim(np.log10(0.95), np.log10(DISP_CAP * 1.05))
    ax.set_xticks(range(len(ARIS)))
    ax.set_xticklabels([f"P{p}" for p in ARIS], fontsize=7)
    ax.set_xlim(-0.6, len(ARIS) - 0.4)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Observed USGS Return Period", fontsize=9)
    ax.set_xlabel("Atlas 14 precipitation ARI threshold", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    for ext in ("png", "pdf"):
        buf = io.BytesIO()
        kw = {"format": ext}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(buf, **kw)
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        log.info("Wrote s3://%s/%s%s.%s", bucket, prefix, FIG_KEY, ext)
    plt.close(fig)


if __name__ == "__main__":
    main()
