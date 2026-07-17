"""validate_station_precip_events.py

Count precipitation events per Atlas-14 return period from the VALID, ACTIVE Indiana
precipitation STATIONS (NOAA ISD/LCD + GHCNh rain gauges), and compare the ARI
frequency distribution against the MRMS storm-object counts from validate_storm_events.py.

This is the rain-gauge counterpart to the MRMS-based counters:
    validate_storm_events.py     — MRMS storm OBJECTS (domain-wide, per-pixel Atlas-14)
    validate_precip_frequency.py — MRMS per-LOCATION events (temporal event-ID)
    THIS script                  — STATION per-gauge events (temporal event-ID)

Method (station events)
───────────────────────
Reuses validate_precip_frequency.py verbatim:
  • identify_events()      independent events separated by >= 24 dry hours
                           (Restrepo-Posada & Eagleson 1982)
  • assign_return_period() each event's max rolling accumulation (1/3/6/12/24 h) vs
                           Atlas-14 at that station → highest RP exceeded (cumulative:
                           a P100 event also counts as P50, P25, ... P2)
Station hourly precip comes from 08d's load_and_qualify (ISD/LCD inches + GHCNh mm→in,
kept where >= 50 % coverage of the 2002-present era and plausible max).  Atlas-14 is
taken at each precip-station's OWN location (07c extracts it directly from the PFDS at
the station coordinates), so each gauge's rain is judged against its own climatology.

"Active" = the station's hourly record extends to within ACTIVE_MONTHS (12) of the
dataset's most-recent hour — i.e. still reporting — on top of the 08d qualification.

Comparison
──────────
MRMS storm objects are domain-wide (one storm spans many gauges) while station events
are per-gauge, so magnitudes differ by ~the number of stations.  Both are normalised to
per-YEAR, and a second panel normalises each source to its own P2 count so the ARI
distribution SHAPE (how fast frequency falls with return period) is directly comparable.

Reads:
    precip/noaa/isd_hourly.parquet, ghcnh_hourly.parquet   (via 08d load_and_qualify)
    atlas14/precipitation_frequency_stations.parquet       (run 07c first)
    analysis/storm_events/storm_events.parquet             (run validate_storm_events.py first)

Writes (analysis/precip_stations/):
    station_precip_event_counts.csv      per-station event counts per RP (+ station-years)
    station_vs_mrms_summary.csv          per-RP totals + per-year rates, both sources
    station_vs_mrms_comparison.{png,svg} the comparison figure

Usage:
    python scripts/validate_station_precip_events.py
    python scripts/validate_station_precip_events.py --start 2020-01-01 --end 2026-12-31
"""
from __future__ import annotations

import argparse
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

from utils import load_config, s3_client, write_bytes_to_s3

# Reuse 08d (precip station loader) and validate_precip_frequency (event functions).
def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register BEFORE exec so module-level @dataclass resolves
    spec.loader.exec_module(mod)
    return mod

m08d = _load("indot_08d", "08d_indot_trigger_analysis.py")
vpf  = _load("validate_precip_frequency", "validate_precip_frequency.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("station_precip")

RETURN_PERIODS      = vpf.RETURN_PERIODS            # [2,10,25,50,100,500,1000]
PRECIP_DURATIONS_HR = vpf.PRECIP_DURATIONS_HR       # [1,3,6,12,24]
HOURS_PER_YEAR      = 8766.0
ACTIVE_MONTHS       = 12                            # "active" = reporting within this many months

STATION_A14_KEY = "atlas14/precipitation_frequency_stations.parquet"   # 07c: Atlas 14 at station coords
STORMS_KEY      = "analysis/storm_events/storm_events.parquet"
OUT_PREFIX      = "analysis/precip_stations/"


def _read_parquet(bucket: str, key: str, columns=None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), columns=columns)


# ── Atlas-14 interpolated to precip-station locations ─────────────────────────

def atlas14_at_stations(stations: pd.DataFrame, bucket: str, prefix: str) -> dict[str, pd.DataFrame]:
    """{station_id -> DataFrame[return_period_yr, duration_hr, depth_in]} from Atlas 14
    extracted DIRECTLY at each precip-station location (07c) — replaces the earlier
    spatial interpolation of the gauge-location Atlas 14 (each gauge's rain is now judged
    against its own climatology)."""
    a14 = _read_parquet(bucket, f"{prefix}{STATION_A14_KEY}")
    a14["station_id"] = a14["station_id"].astype(str)
    a14 = a14[a14["duration_hr"].isin(PRECIP_DURATIONS_HR)
              & a14["return_period_yr"].isin(RETURN_PERIODS)]
    ids = set(stations["station_id"].astype(str))
    a14 = a14[a14["station_id"].isin(ids)]
    return {sid: g[["return_period_yr", "duration_hr", "depth_in"]].reset_index(drop=True)
            for sid, g in a14.groupby("station_id")}


# ── Active-station selection ──────────────────────────────────────────────────

def select_active(qual: pd.DataFrame, hourly_by_sid: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Keep qualified stations still reporting within ACTIVE_MONTHS of the dataset's
    most-recent hour, with valid coordinates."""
    last_hour = {sid: s.index.max() for sid, s in hourly_by_sid.items() if len(s)}
    ref_end = max(last_hour.values())
    cutoff = ref_end - pd.DateOffset(months=ACTIVE_MONTHS)
    active_ids = {sid for sid, t in last_hour.items() if t >= cutoff}
    keep = qual[qual["station_id"].isin(active_ids)
                & qual["latitude"].notna() & qual["longitude"].notna()].reset_index(drop=True)
    return keep, ref_end


# ── Station event counting ────────────────────────────────────────────────────

def count_station_events(active: pd.DataFrame, hourly_by_sid: dict, a14_map: dict,
                         start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(active)
    for i, r in enumerate(active.itertuples(index=False), 1):
        sid = r.station_id
        ser = hourly_by_sid.get(sid)
        a14_site = a14_map.get(sid)
        if ser is None or ser.empty or a14_site is None or a14_site.empty:
            continue
        ts = ser[(ser.index >= start) & (ser.index <= end)].sort_index()
        if ts.empty:
            continue
        events = vpf.identify_events(ts, vpf.DRY_THRESHOLD_IN, vpf.DRY_HOURS_MIN)
        counts = {rp: 0 for rp in RETURN_PERIODS}
        for ev in events:
            rp = vpf.assign_return_period(ev, ts, a14_site)
            if rp is not None:
                for r_ in RETURN_PERIODS:
                    if r_ <= rp:
                        counts[r_] += 1
        row = {"station_id": sid, "source": getattr(r, "source", ""),
               "latitude": float(r.latitude), "longitude": float(r.longitude),
               "n_events_total": len(events),
               "station_years": round(len(ts) / HOURS_PER_YEAR, 3)}
        row.update({f"P{rp}": counts[rp] for rp in RETURN_PERIODS})
        rows.append(row)
        if i % 25 == 0 or i == n:
            log.info("  [%d/%d] stations processed", i, n)
    return pd.DataFrame(rows)


# ── MRMS storm-object counts ──────────────────────────────────────────────────

def mrms_storm_counts(bucket: str, prefix: str, start: pd.Timestamp,
                      end: pd.Timestamp) -> tuple[dict[int, int], float] | tuple[None, None]:
    try:
        storms = _read_parquet(bucket, f"{prefix}{STORMS_KEY}")
    except Exception as e:                                     # noqa: BLE001
        log.warning("Could not read MRMS storm_events.parquet (%s) — run validate_storm_events.py "
                    "first; comparison panel will be skipped.", e)
        return None, None
    storms["start_dt"] = pd.to_datetime(storms["start_dt"], utc=True)
    storms = storms[(storms["start_dt"] >= start) & (storms["start_dt"] <= end)]
    n_years = (end - start).days / 365.25
    counts = {rp: int((storms["assigned_rp"].dropna() >= rp).sum()) for rp in RETURN_PERIODS}
    log.info("MRMS storms in window: %d (%.1f yr)", len(storms), n_years)
    return counts, n_years


# ── Comparison figure ─────────────────────────────────────────────────────────

def _save_fig(fig, bucket: str, prefix: str, stem: str) -> None:
    for ext in ("png", "pdf"):                               # exact 6.5x4 (no bbox_tight)
        buf = io.BytesIO()
        kw = {"format": ext}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(buf, **kw)
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{OUT_PREFIX}{stem}.{ext}")
        log.info("Wrote s3://%s/%s%s%s.%s", bucket, prefix, OUT_PREFIX, stem, ext)
    plt.close(fig)


def make_figure(summary: pd.DataFrame, has_mrms: bool, n_stations: int,
                bucket: str, prefix: str) -> None:
    rp_labels = [f"P{rp}" for rp in summary["return_period_yr"]]
    x = np.arange(len(rp_labels))

    # ── Figure 1: per-year event frequency (grouped bars) ─────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    w = 0.38
    st = summary["station_events_per_yr"].to_numpy()
    ax.bar(x - (w / 2 if has_mrms else 0), st, w, color="#2c7fb8", label="Station events / yr")
    if has_mrms:
        mr = summary["mrms_storms_per_yr"].to_numpy()
        ax.bar(x + w / 2, mr, w, color="#d95f0e", label="MRMS storms / yr")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(rp_labels)
    ax.set_ylabel("Events per year (log)", fontsize=10)
    ax.set_xlabel("Atlas-14 return period", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, bucket, prefix, "station_vs_mrms_bars")

    # ── Figure 2: ARI distribution shape (each normalised to its own P2) ──────
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    def norm(col):
        v = summary[col].to_numpy(float); base = v[0] if v[0] > 0 else np.nan
        return v / base

    ax.plot(x, norm("station_events"), "o-", color="#2c7fb8", lw=2, label="Stations")
    if has_mrms:
        ax.plot(x, norm("mrms_storms"), "s-", color="#d95f0e", lw=2, label="MRMS storms")
    ax.plot(x, [2.0 / rp for rp in summary["return_period_yr"]], "k--", lw=1,
            label="Atlas-14 (1/RP, ×P2)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(rp_labels)
    ax.set_ylabel("Frequency relative to P2", fontsize=10)
    ax.set_xlabel("Atlas-14 return period", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, bucket, prefix, "station_vs_mrms_shape")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2020-01-01", help="analysis window start (matches validate_storm_events)")
    ap.add_argument("--end",   default="2026-12-31", help="analysis window end")
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC"); end = pd.Timestamp(args.end, tz="UTC")

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    log.info("Loading + qualifying ISD/GHCNh precip stations (08d)...")
    qual, hourly_by_sid = m08d.load_and_qualify(bucket, prefix)

    active, ref_end = select_active(qual, hourly_by_sid)
    log.info("Active precip stations (reporting within %d mo of %s): %d / %d qualified",
             ACTIVE_MONTHS, ref_end.date(), len(active), len(qual))

    log.info("Loading station-location Atlas 14 (07c)...")
    a14_map = atlas14_at_stations(active, bucket, prefix)

    log.info("Counting station precip events (%s → %s)...", start.date(), end.date())
    per_station = count_station_events(active, hourly_by_sid, a14_map, start, end)
    if per_station.empty:
        log.error("No station events counted.")
        return

    total_station_years = float(per_station["station_years"].sum())
    n_years_window = (end - start).days / 365.25
    n_stations = len(per_station)

    mrms_counts, mrms_years = mrms_storm_counts(bucket, prefix, start, end)
    has_mrms = mrms_counts is not None

    rows = []
    for rp in RETURN_PERIODS:
        st_total = int(per_station[f"P{rp}"].sum())
        row = {
            "return_period_yr":            rp,
            "station_events":              st_total,
            "station_events_per_yr":       round(st_total / n_years_window, 3),
            "station_events_per_station_yr": round(st_total / total_station_years, 4) if total_station_years else None,
        }
        if has_mrms:
            row["mrms_storms"]        = mrms_counts[rp]
            row["mrms_storms_per_yr"] = round(mrms_counts[rp] / mrms_years, 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    log.info("Per-RP summary:\n%s", summary.to_string(index=False))

    # ── Write outputs ─────────────────────────────────────────────────────────
    def _put_csv(df, key):
        s3_client().put_object(Bucket=bucket, Key=f"{prefix}{OUT_PREFIX}{key}",
                               Body=df.to_csv(index=False).encode(), ContentType="text/csv")
        log.info("Wrote s3://%s/%s%s%s", bucket, prefix, OUT_PREFIX, key)

    _put_csv(per_station, "station_precip_event_counts.csv")
    _put_csv(summary, "station_vs_mrms_summary.csv")
    make_figure(summary, has_mrms, n_stations, bucket, prefix)
    log.info("Done. %d active stations, %.1f total station-years.", n_stations, total_station_years)


if __name__ == "__main__":
    main()
