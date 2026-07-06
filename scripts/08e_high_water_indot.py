"""08e_high_water_indot.py

Apply INDOT's CURRENT trigger rule (trailing 24 h >= 2.5 in) to observed
"high water over roadway" events and score how many the rule would have caught.

Inputs (S3, under <prefix>high_water/):
    high_water_2025-06-25_clean.csv   roadway_event_id, lat, lon  (location)
    high_water_2025-06-25.xlsx        roadway_event_id, record_created_date
                                      (original, offset-aware times e.g. ...-04)
The two are JOINED on roadway_event_id: location from the clean CSV, authoritative
timestamps from the xlsx (its UTC offset makes the 24-h window unambiguous).

Per roadway_event_id:
  1. earliest / latest = min / max record_created_date (from the xlsx, -> UTC).
  2. lat / lon          = the event location (from the clean CSV).
  3. nearest precip station whose hourly record OVERLAPS [earliest-48 h, latest]
     (walk outward by distance; 48 h back so the 24-h accumulation is defined at
     the detection start).
  4. detection: max trailing-24 h accumulation over [earliest-24 h, latest].
        >= 2.5 in  -> TRUE POSITIVE  (high water anticipated)
        <  2.5 in  -> FALSE NEGATIVE (missed)
  Every record is a real event, so only TP and FN exist (no FP / TN).

Outputs (S3, under <prefix>high_water/):
  1. hw_classification_bars.{png,svg}   FN vs TP counts, labelled FNR / POD (% of events)
  2. hw_example_tp_fn.{png,svg}         one TP + one FN explainer (24-h precip over a
                                        bar chart of the high-water record times)
  3. precip_stations_used.geojson       precip stations actually used
  4. high_water_events_classified.geojson  event points tagged TP / FN
  (+ hw_classification.csv)

Usage:
    python scripts/08e_high_water_indot.py
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
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

# Reuse 08d's precip loaders (ISD/GHCNh, memory-lean) + its haversine.
_spec = importlib.util.spec_from_file_location(
    "indot_08d", Path(__file__).with_name("08d_indot_trigger_analysis.py"))
m08d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08d)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("08e_hw")

DURATION_HR   = 24        # INDOT trailing accumulation
THRESH_IN     = 2.5       # INDOT trigger depth
BACK_COVER_H  = 48        # station must overlap [earliest-48h, latest]
BACK_DETECT_H = 24        # detection window starts 24 h before the earliest record

HW_DIR    = "high_water/"
CLEAN_KEY = HW_DIR + "high_water_2025-06-25_clean.csv"
XLSX_KEY  = HW_DIR + "high_water_2025-06-25.xlsx"
# Fallback tz if the xlsx times ever come through tz-naive (they shouldn't — they
# carry an explicit offset).  Indiana is Eastern.
FALLBACK_TZ = "America/Indiana/Indianapolis"
CLR = {"TP": "#2e7d32", "FN": "#c62828"}


# ---------- I/O ----------

def _get(bucket: str, key: str) -> bytes:
    return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _put(bucket: str, key: str, body: bytes, ctype: str) -> None:
    s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType=ctype)
    log.info("Wrote s3://%s/%s", bucket, key)


def _save_fig(fig, bucket: str, key_stem: str, dpi: int) -> None:
    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=dpi, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{key_stem}.{ext}")
        log.info("Wrote s3://%s/%s.%s", bucket, key_stem, ext)
    plt.close(fig)


def _to_utc(s: pd.Series) -> pd.Series:
    """Parse offset-aware (e.g. ...-04) timestamps to UTC; localise naive as Eastern."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if getattr(t.dt, "tz", None) is None:          # no offset present -> assume Eastern
        t = t.dt.tz_localize(FALLBACK_TZ, ambiguous="NaT",
                             nonexistent="NaT").dt.tz_convert("UTC")
    return t


def load_events(bucket: str, prefix: str) -> pd.DataFrame:
    """One row per roadway_event_id: lat, lon (clean CSV) + earliest/latest/record
    times (xlsx).  Inner-joined on the id."""
    clean = pd.read_csv(io.BytesIO(_get(bucket, f"{prefix}{CLEAN_KEY}")))
    clean["roadway_event_id"] = clean["roadway_event_id"].astype(str)
    loc = (clean.dropna(subset=["lat", "lon"])
                .groupby("roadway_event_id")[["lat", "lon"]].first())

    xls = pd.read_excel(io.BytesIO(_get(bucket, f"{prefix}{XLSX_KEY}")), engine="openpyxl")
    xls["roadway_event_id"] = xls["roadway_event_id"].astype(str)
    xls["record_created_date"] = _to_utc(xls["record_created_date"])
    xls = xls.dropna(subset=["record_created_date"])
    log.info("Sample parsed time (UTC): %s", xls["record_created_date"].iloc[0])
    tg = xls.groupby("roadway_event_id")["record_created_date"]
    times = pd.DataFrame({
        "earliest": tg.min(), "latest": tg.max(),
        "record_times": tg.agg(lambda s: sorted(s.tolist())),
        "n_records": tg.size(),
    })

    df = loc.join(times, how="inner")
    log.info("Events: %d located (clean CSV), %d timed (xlsx), %d with BOTH (used)",
             len(loc), len(times), len(df))
    return df.reset_index()


# ---------- analysis ----------

def assign_station(lat, lon, cov_start, cov_end, qual, hourly_by_sid):
    """Nearest qualifying station whose hourly record overlaps [cov_start, cov_end]."""
    ids  = qual["station_id"].to_numpy()
    lats = qual["latitude"].to_numpy(float)
    lons = qual["longitude"].to_numpy(float)
    srcs = qual["source"].to_numpy() if "source" in qual.columns else np.array([""] * len(ids))
    d = m08d._haversine_mi(lat, lon, lats, lons)
    for idx in np.argsort(d):
        sid = str(ids[idx])
        ser = hourly_by_sid.get(sid)
        if ser is None or ser.empty:
            continue
        win = ser[(ser.index >= cov_start) & (ser.index <= cov_end)]
        if len(win) > 0 and bool(win.notna().any()):
            return sid, float(d[idx]), float(lats[idx]), float(lons[idx]), str(srcs[idx])
    return None


def max_accum(ser: pd.Series, earliest, latest) -> float:
    # Floor to the hour: event times carry sub-hour precision, but the station
    # series is on the hour — an unfloored grid would align to nothing.
    e0, l0 = earliest.floor("h"), latest.floor("h")
    grid = pd.date_range(e0 - pd.Timedelta(hours=BACK_COVER_H), l0, freq="1h", tz="UTC")
    p = ser.reindex(grid).fillna(0.0)
    roll = p.rolling(DURATION_HR, min_periods=DURATION_HR).sum()
    det = roll[(roll.index >= e0 - pd.Timedelta(hours=BACK_DETECT_H)) & (roll.index <= l0)]
    return float(det.max()) if len(det) and np.isfinite(det.max()) else 0.0


# ---------- figures ----------

def bars_figure(n_tp, n_fn, bucket, key_stem):
    total = n_tp + n_fn
    fig, ax = plt.subplots(figsize=(8, 6))
    vals, acr, colors, x = [n_fn, n_tp], ["FNR", "POD"], [CLR["FN"], CLR["TP"]], [0, 1]
    ax.bar(x, vals, width=0.6, color=colors)
    for xi, v, a in zip(x, vals, acr):
        pct = 100.0 * v / total if total else 0.0
        ax.text(xi, v + (max(vals) * 0.01 if max(vals) else 0.01),
                f"{pct:.0f}%\n{a}", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["False negative", "True positive"], fontsize=15)
    ax.set_ylabel("Number of events", fontsize=15)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) else 1)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(f"INDOT 24-h ≥ 2.5 in trigger on {total} high-water events", fontsize=15)
    fig.tight_layout()
    _save_fig(fig, bucket, key_stem, dpi=150)


def draw_example(fig, cell, ev, ser):
    klass, earliest, latest = ev["klass"], ev["earliest"], ev["latest"]
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[2.2, 1.0], hspace=0.12)
    axp = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=axp)

    e0, l0 = earliest.floor("h"), latest.floor("h")
    vs, ve = e0 - pd.Timedelta(hours=72), l0 + pd.Timedelta(hours=24)
    grid = pd.date_range(e0 - pd.Timedelta(hours=BACK_COVER_H), l0, freq="1h", tz="UTC")
    roll = ser.reindex(grid).fillna(0.0).rolling(DURATION_HR, min_periods=DURATION_HR).sum()
    gv = pd.date_range(vs, ve, freq="1h", tz="UTC")
    pv, rv = ser.reindex(gv), roll.reindex(gv)

    det_s, det_e = earliest - pd.Timedelta(hours=BACK_DETECT_H), latest
    for ax in (axp, axr):
        ax.axvspan(det_s, det_e, color=CLR[klass], alpha=0.10, zorder=0)

    axp.bar(gv, pv.fillna(0.0).values, width=0.035, color="#90a4ae", alpha=0.9, zorder=2)
    axp.plot(gv, rv.values, color="#1565c0", lw=1.6, zorder=3)
    axp.axhline(THRESH_IN, color="#d32f2f", ls="--", lw=1.3, zorder=3)
    axp.text(0.012, 0.9, f"threshold = {THRESH_IN} in", transform=axp.transAxes,
             fontsize=9, color="#d32f2f", va="top")
    axp.set_ylabel("Precip (in)", fontsize=10)
    rmax = float(np.nanmax(rv.values)) if np.isfinite(np.nanmax(rv.values)) else THRESH_IN
    axp.set_ylim(0, max(THRESH_IN * 1.3, rmax * 1.12))
    axp.tick_params(labelbottom=False, labelsize=9)
    name = "True Positive — high water anticipated" if klass == "TP" \
        else "False Negative — high water missed"
    axp.set_title(f"{name}\nmax 24-h accum = {ev['max_accum']:.2f} in   "
                  f"({ev['dist_mi']:.0f} mi to station {ev['sid']})",
                  fontsize=10.5, color=CLR[klass], fontweight="bold", loc="left", pad=5)

    axr.bar(ev["record_times"], [1] * len(ev["record_times"]), width=0.12, color="#37474f", zorder=2)
    axr.set_ylabel("HW records", fontsize=10)
    axr.set_ylim(0, 1.4); axr.set_yticks([]); axr.set_xlim(vs, ve)
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axr.tick_params(labelsize=8)


def example_figure(ev_tp, ev_fn, hourly_by_sid, bucket, key_stem):
    fig = plt.figure(figsize=(13.5, 5.6))
    outer = fig.add_gridspec(1, 2, wspace=0.16, left=0.05, right=0.985, top=0.85, bottom=0.1)
    if ev_tp is not None:
        draw_example(fig, outer[0, 0], ev_tp, hourly_by_sid[ev_tp["sid"]])
    if ev_fn is not None:
        draw_example(fig, outer[0, 1], ev_fn, hourly_by_sid[ev_fn["sid"]])
    fig.suptitle("INDOT trigger on high-water events — worked examples "
                 "(bars below = high-water record times)",
                 fontsize=12.5, fontweight="bold", y=0.965)
    _save_fig(fig, bucket, key_stem, dpi=170)


# ---------- main ----------

def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    ev_df = load_events(bucket, prefix)
    log.info("Loading NOAA hourly precip (ISD + GHCNh)...")
    qual, hourly_by_sid = m08d.load_and_qualify(bucket, prefix)

    events: list[dict] = []
    n_no_station = 0
    for r in ev_df.itertuples(index=False):
        earliest, latest = r.earliest, r.latest
        assigned = assign_station(float(r.lat), float(r.lon),
                                  earliest - pd.Timedelta(hours=BACK_COVER_H), latest,
                                  qual, hourly_by_sid)
        if assigned is None:
            n_no_station += 1
            continue
        sid, dist_mi, st_lat, st_lon, source = assigned
        acc = max_accum(hourly_by_sid[sid], earliest, latest)
        events.append({
            "roadway_event_id": r.roadway_event_id, "lat": float(r.lat), "lon": float(r.lon),
            "earliest": earliest, "latest": latest, "n_records": int(r.n_records),
            "record_times": r.record_times, "sid": sid, "station_lat": st_lat,
            "station_lon": st_lon, "source": source, "dist_mi": dist_mi,
            "max_accum": acc, "klass": "TP" if acc >= THRESH_IN else "FN",
        })

    if not events:
        log.error("No events could be classified (no covering precip station).")
        return
    n_tp = sum(e["klass"] == "TP" for e in events)
    n_fn = sum(e["klass"] == "FN" for e in events)
    pod = 100.0 * n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    log.info("Classified %d events: TP=%d  FN=%d  POD=%.0f%%  (no station: %d)",
             len(events), n_tp, n_fn, pod, n_no_station)

    # ── Diagnostics: is 0 TP a real result or an assignment/coverage artefact? ──
    accs = np.array([e["max_accum"] for e in events])
    dists = np.array([e["dist_mi"] for e in events])
    log.info("max 24-h accum (in):  min=%.2f  median=%.2f  p90=%.2f  max=%.2f  |  >=%.1f in: %d",
             accs.min(), np.median(accs), np.percentile(accs, 90), accs.max(),
             THRESH_IN, int((accs >= THRESH_IN).sum()))
    log.info("assigned-station distance (mi):  median=%.1f  p90=%.1f  max=%.1f",
             np.median(dists), np.percentile(dists, 90), dists.max())
    log.info("events with ZERO accumulation (no precip in window): %d / %d",
             int((accs == 0).sum()), len(accs))

    # ── Data outputs FIRST (so they persist even if a figure errors) ──────────
    # 3) precip stations used
    used: dict = {}
    for e in events:
        u = used.setdefault(e["sid"], {"sid": e["sid"], "lat": e["station_lat"],
                                       "lon": e["station_lon"], "source": e["source"], "n": 0})
        u["n"] += 1
    st_feats = [{"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [u["lon"], u["lat"]]},
                 "properties": {"station_id": u["sid"], "source": u["source"],
                                "n_events_used": u["n"]}} for u in used.values()]
    _put(bucket, f"{prefix}{HW_DIR}precip_stations_used.geojson",
         json.dumps({"type": "FeatureCollection", "features": st_feats}, indent=1).encode(),
         "application/geo+json")

    # 4) events with classification
    ev_feats = [{"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
                 "properties": {"roadway_event_id": e["roadway_event_id"],
                                "classification": e["klass"],
                                "max_accum_in": round(e["max_accum"], 3),
                                "n_records": e["n_records"],
                                "earliest": e["earliest"].isoformat(),
                                "latest": e["latest"].isoformat(),
                                "precip_station_id": e["sid"],
                                "dist_mi": round(e["dist_mi"], 2)}} for e in events]
    _put(bucket, f"{prefix}{HW_DIR}high_water_events_classified.geojson",
         json.dumps({"type": "FeatureCollection", "features": ev_feats}, indent=1).encode(),
         "application/geo+json")

    # bonus CSV
    csv = pd.DataFrame([{k: e[k] for k in
                         ("roadway_event_id", "lat", "lon", "earliest", "latest", "n_records",
                          "sid", "dist_mi", "source", "max_accum", "klass")} for e in events])
    _put(bucket, f"{prefix}{HW_DIR}hw_classification.csv", csv.to_csv(index=False).encode(),
         "text/csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    bars_figure(n_tp, n_fn, bucket, f"{prefix}{HW_DIR}hw_classification_bars")
    tps = [e for e in events if e["klass"] == "TP"]
    fns = [e for e in events if e["klass"] == "FN"]
    ev_tp = max(tps, key=lambda e: e["max_accum"]) if tps else None
    ev_fn = max(fns, key=lambda e: e["max_accum"]) if fns else None
    try:
        example_figure(ev_tp, ev_fn, hourly_by_sid, bucket, f"{prefix}{HW_DIR}hw_example_tp_fn")
    except Exception as e:                                       # noqa: BLE001
        log.warning("example figure failed (data outputs already written): %s", e)
    log.info("Done. Outputs under s3://%s/%s%s", bucket, prefix, HW_DIR)


if __name__ == "__main__":
    main()
