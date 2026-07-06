"""08d_indot_trigger_analysis.py

INDOT current-procedure version of the event-overlap confusion matrix.

Where 08c tests the PROPOSED trigger (nearest MRMS pixel, Kirpich-Tc duration,
Atlas-14 depth), this script tests INDOT's CURRENT operational rule:

    precip source   : nearest hourly weather station (NOAA ISD / ASOS-METAR)
    accumulation    : trailing 24 hours          (FIXED, not Tc)
    trigger depth   : 2.5 in                       (FIXED, not Atlas-14 / return period)
    flood           : hourly streamflow >= Q(flow_rp)   (Q10 / Q50 / Q100)

Everything else — event grouping, +/-24 h linking, episodes, TP/FN/FP/TN — is
IDENTICAL to 08 (helpers are imported from it, not reimplemented), so the flood
side is directly comparable to 08c.

For each gauge the nearest ISD station that (a) has a usable hourly precip record
and (b) overlaps the streamflow record by >= MIN_OVERLAP_YEARS is assigned; the
analysis window is that streamflow-station overlap.  GHCNh is deliberately NOT
used here — many of its stations are daily/accumulation COOP gauges whose values
break a sub-daily accumulation; ISD is true hourly depth in inches.

Per station we keep only the TP/FP/FN/TN counts (+ event counts and window QC).
Skill metrics are computed GLOBALLY ONLY — the cells are pooled across all
stations and scored once per flow return period, so INDOT gets three metric sets
(Q10, Q50, Q100):
    POD  = TP / (TP+FN)                 probability of detection (hit rate)
    FAR  = FP / (TP+FP)                 false-alarm ratio
    SR   = 1 - FAR = TP / (TP+FP)       success ratio
    CSI  = TP / (TP+FP+FN)              critical success index (threat score)
    BIAS = (TP+FP) / (TP+FN)            frequency bias
    F1, accuracy                        associated scores

Writes (S3 only):
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_indot.parquet   (per station x flow_rp: counts)
    s3://<bucket>/<prefix>analysis/indot_trigger_metrics.csv              (global skill, one row per flow_rp)
    s3://<bucket>/<prefix>analysis/figures/indot_performance_diagram.{png,svg}

Usage:
    python scripts/08d_indot_trigger_analysis.py
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

from utils import load_config, s3_client, write_bytes_to_s3, write_parquet_to_s3

# Reuse 08's loaders + event helpers so keys / logic match the real runs.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("08d_indot")

FLOW_RPS   = m.FLOW_RPS                 # Q10 / Q50 / Q100
DURATION_HR = 24                        # INDOT: trailing 24-hour accumulation
PRECIP_THRESH_IN = 2.5                  # INDOT: fixed 2.5 in trigger
ISD_KEY = "precip/noaa/isd_hourly.parquet"
INV_KEY = "stations/indiana_streamflow_sites.parquet"
OUTPUT_KEY  = "analysis/event_confusion_matrix_indot.parquet"
METRICS_KEY = "analysis/indot_trigger_metrics.csv"
FIG_KEY     = "analysis/figures/indot_performance_diagram"

# Station-eligibility knobs
MIN_NONNULL_HOURS = 4000        # station must carry at least this many hourly precip records
MIN_OVERLAP_YEARS = 2.0         # station record must overlap the streamflow record this long
MAX_MISSING_PCT   = 60.0        # drop a gauge if its assigned station is emptier than this
SANITY_MAX_IN     = 12.0        # reject stations with implausibly large hourly values
EARTH_MI = 3958.7613


# ---------- Skill metrics ----------

def skill(tp: int, fp: int, fn: int, tn: int) -> dict:
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    sr  = tp / (tp + fp) if (tp + fp) else np.nan
    far = 1.0 - sr if (tp + fp) else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    f1  = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan
    return {"pod": pod, "far": far, "sr": sr, "csi": csi,
            "bias": bias, "f1": f1, "accuracy": acc}


# ---------- ISD precip: load, catalogue, assign ----------

def load_isd(bucket: str, prefix: str) -> pd.DataFrame:
    df = m._read_parquet_s3(bucket, f"{prefix}{ISD_KEY}",
                            columns=["station_id", "datetime_utc", "latitude", "longitude", "precip_in"])
    df["station_id"]   = df["station_id"].astype(str)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["precip_in"]    = pd.to_numeric(df["precip_in"], errors="coerce")
    return df


def isd_catalogue(isd: pd.DataFrame) -> pd.DataFrame:
    """One row per ISD station: coords, record span, non-null count, max value."""
    g = isd.groupby("station_id")
    cat = pd.DataFrame({
        "latitude":  g["latitude"].first(),
        "longitude": g["longitude"].first(),
        "start":     g["datetime_utc"].min(),
        "end":       g["datetime_utc"].max(),
        "n_nonnull": g["precip_in"].count(),
        "max_in":    g["precip_in"].max(),
    }).reset_index()
    usable = cat[(cat["n_nonnull"] >= MIN_NONNULL_HOURS)
                 & (cat["max_in"] <= SANITY_MAX_IN)
                 & cat["latitude"].notna() & cat["longitude"].notna()].copy()
    log.info("ISD stations: %d total, %d usable (>=%d hrs, <=%.0f in)",
             len(cat), len(usable), MIN_NONNULL_HOURS, SANITY_MAX_IN)
    return usable.reset_index(drop=True)


def _haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d1, d2 = lat2 - lat1, lon2 - lon1
    a = np.sin(d1 / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d2 / 2) ** 2
    return 2 * EARTH_MI * np.arcsin(np.sqrt(a))


def assign_station(lat: float, lon: float,
                   flow_start: pd.Timestamp, flow_end: pd.Timestamp,
                   usable: pd.DataFrame):
    """Nearest usable ISD station whose record overlaps the streamflow record by
    >= MIN_OVERLAP_YEARS.  Returns (row, dist_mi, win_start, win_end) or None."""
    d = _haversine_mi(lat, lon, usable["latitude"].to_numpy(float), usable["longitude"].to_numpy(float))
    for idx in np.argsort(d):
        s = usable.iloc[idx]
        ws, we = max(flow_start, s["start"]), min(flow_end, s["end"])
        if ws < we and (we - ws) >= pd.Timedelta(days=365.25 * MIN_OVERLAP_YEARS):
            return s, float(d[idx]), ws, we
    return None


# ---------- Per-station analysis (fixed 24 h / 2.5 in) ----------

def analyse(site_no, precip_hourly, flow_hourly, flow_stats_row,
            ws, we, station_id, dist_mi) -> list[dict]:
    grid = pd.date_range(ws, we, freq="1h", tz="UTC")
    precip = precip_hourly.reindex(grid)
    pct_missing = round(100.0 * float(precip.isna().mean()), 2)
    if pct_missing > MAX_MISSING_PCT:
        log.info("[%s] assigned station %s is %.0f%% empty — skipping",
                 site_no, station_id, pct_missing)
        return []
    precip = precip.fillna(0.0)
    flow = flow_hourly.reindex(grid)
    n_common = len(grid)

    precip_wet = (precip.rolling(DURATION_HR, min_periods=DURATION_HR).sum() >= PRECIP_THRESH_IN).fillna(False)
    precip_events = m.group_wet_events(precip_wet)

    out: list[dict] = []
    for flow_rp in FLOW_RPS:
        q = flow_stats_row.get(f"Q{flow_rp}")
        if pd.isna(q):
            continue
        flow_events = m.group_wet_events((flow >= float(q)).fillna(False))
        tp, fp, fn, tn = m.classify_overlap(precip_events, flow_events, ws, we)
        out.append({"site_no": site_no, "precip_station_id": station_id,
                    "dist_mi": round(dist_mi, 2), "flow_rp_yr": flow_rp,
                    "duration_hr": DURATION_HR, "precip_thresh_in": PRECIP_THRESH_IN,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "n_precip_events": len(precip_events), "n_flow_events": len(flow_events),
                    "pct_precip_missing": pct_missing,
                    "common_start": ws, "common_end": we, "n_common_hours": n_common})
    return out


# ---------- Performance (Roebber) diagram ----------

def performance_diagram(pooled: pd.DataFrame, bucket: str, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7.5))
    g = np.linspace(0.001, 1, 400)
    SR, POD = np.meshgrid(g, g)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cf = ax.contourf(SR, POD, CSI, levels=np.arange(0, 1.01, 0.1), cmap="Blues", alpha=0.75)
    cl = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.45", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=8, inline=True)
    for b in [0.3, 0.5, 1, 1.5, 2, 3, 5]:                  # frequency-bias lines: POD = b*SR
        x = np.linspace(0, 1, 10)
        ax.plot(x, np.minimum(b * x, 1), ls="--", color="0.5", lw=0.7)
        if b <= 1: ax.text(1.008, b, f"{b:g}", fontsize=7.5, color="0.4", va="center")
        else:      ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7.5, color="0.4", ha="center")

    # One global operating point per flow return period (Q10 / Q50 / Q100).
    colors = {10: "#1a9850", 50: "#f46d43", 100: "#762a83"}
    for rp in FLOW_RPS:
        pr = pooled[pooled.flow_rp_yr == rp]
        if pr.empty or not (np.isfinite(pr.iloc[0].sr) and np.isfinite(pr.iloc[0].pod)):
            continue
        r = pr.iloc[0]
        ax.scatter([r.sr], [r.pod], marker="*", s=520, color=colors.get(rp, "grey"),
                   edgecolors="black", lw=1.3, zorder=7,
                   label=f"Q{rp}:  POD={r.pod:.2f}  FAR={r.far:.2f}  CSI={r.csi:.2f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Success ratio  (1 − FAR)", fontsize=12)
    ax.set_ylabel("Probability of detection (POD)", fontsize=12)
    ax.text(0.985, 0.03, "frequency-bias lines", fontsize=8, color="0.4", ha="right", style="italic")
    ax.set_title("INDOT current flood trigger — nearest ISD gauge, 24-h ≥ 2.5 in\n"
                 "performance diagram (dashed = bias, shaded = CSI)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
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
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading flow stats, streamflow, station coords...")
    flow_stats = m.load_flow_stats(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)
    inv = m._read_parquet_s3(bucket, f"{prefix}{INV_KEY}", columns=["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    coords = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).set_index("site_no")

    q_cols = [f"Q{rp}" for rp in FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])
    universe = sorted(has_q & set(streamflow["site_no"]) & set(coords.index))
    log.info("Universe (valid Q ∩ streamflow ∩ coords): %d", len(universe))

    log.info("Loading NOAA ISD hourly precip (this is the big one)...")
    isd = load_isd(bucket, prefix)
    usable = isd_catalogue(isd)

    flow_start = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end   = streamflow.groupby("site_no")["datetime_utc"].max()

    precip_cache: dict[str, pd.Series] = {}
    def station_hourly(sid: str) -> pd.Series:
        if sid not in precip_cache:
            sub = isd[isd["station_id"] == sid].set_index("datetime_utc")["precip_in"].sort_index()
            precip_cache[sid], _ = m.resample_station_precip(sub)
        return precip_cache[sid]

    all_records: list[dict] = []
    n_assigned = 0
    for i, site_no in enumerate(universe, 1):
        fs_row = flow_stats[flow_stats["site_no"] == site_no]
        if fs_row.empty:
            continue
        fs0, fe0 = flow_start.get(site_no, pd.NaT), flow_end.get(site_no, pd.NaT)
        if pd.isna(fs0) or pd.isna(fe0):
            continue
        c = coords.loc[site_no]
        assigned = assign_station(float(c["dec_lat_va"]), float(c["dec_long_va"]), fs0, fe0, usable)
        if assigned is None:
            log.info("[%d/%d] %s: no usable ISD station within record — skipping", i, len(universe), site_no)
            continue
        srow, dist_mi, ws, we = assigned
        n_assigned += 1
        precip_site = station_hourly(srow["station_id"])
        flow_site = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
        recs = analyse(site_no, precip_site, flow_site, fs_row.iloc[0], ws, we,
                       srow["station_id"], dist_mi)
        all_records.extend(recs)
        if recs:
            log.info("[%d/%d] %s ← %s (%.1f mi): %d combos", i, len(universe), site_no,
                     srow["station_id"], dist_mi, len(recs))

    if not all_records:
        log.error("No results produced.")
        return

    out = pd.DataFrame(all_records)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s%s (%d rows, %d stations assigned)", prefix, OUTPUT_KEY, len(out), n_assigned)

    # Pooled skill per flow return period (sum the cells, then score)
    pooled_rows = []
    for rp in FLOW_RPS:
        s = out[out.flow_rp_yr == rp][["tp", "fp", "fn", "tn"]].sum()
        row = {"flow_rp_yr": rp, "n_stations": int((out.flow_rp_yr == rp).sum()),
               "tp": int(s.tp), "fp": int(s.fp), "fn": int(s.fn), "tn": int(s.tn)}
        row.update({k: (round(v, 4) if np.isfinite(v) else v)
                    for k, v in skill(s.tp, s.fp, s.fn, s.tn).items()})
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    log.info("Global skill:\n%s", pooled[["flow_rp_yr", "n_stations", "pod", "far", "csi", "bias"]].to_string(index=False))

    # Metrics CSV: GLOBAL skill only — one row per flow return period (Q10/Q50/Q100).
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}{METRICS_KEY}",
                           Body=pooled.to_csv(index=False).encode(), ContentType="text/csv")
    log.info("Wrote s3://%s/%s%s (global skill, one row per flow_rp)", bucket, prefix, METRICS_KEY)

    performance_diagram(pooled, bucket, prefix)
    log.info("Done.")


if __name__ == "__main__":
    main()
