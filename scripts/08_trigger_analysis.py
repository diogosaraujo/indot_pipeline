"""08_trigger_analysis.py

Event-overlap confusion matrix: does an exceedance of an Atlas 14 precipitation
threshold anticipate an exceedance of the matching streamflow flood threshold?

For every (station, source, accumulation duration D, precip return period, flow
return period) the hourly record is classified into wet/dry events and scored:

    Precip wet hour : trailing D-hour accumulation ≥ Atlas14(D, precip_rp)
    Flow   wet hour : hourly streamflow            ≥ Q(flow_rp)
    Events          : contiguous wet hours; runs separated by < 24 dry hours
                      are merged into one event.

    TP  a flood (flow-wet) event with a precip-wet hour in [flood_start−24h,
        flood_end]      — precipitation correctly preceded (≤24 h) the flood
    FN  a flood event with no such preceding precip-wet               — missed
    FP  a precip-wet event with no flood-wet in [precip_start, precip_end+24h]
                                                                 — false alarm
    TN  each dry gap between consecutive wet events (precip or flow) counts as 1

Sources (nearest only — watershed-mean sources removed):
    nearest          nearest MRMS pixel  (mrms/<product>/nearest_pixel.parquet)
    station_nearest  nearest ISD/GHCNh gauge that COVERS the analysis window

Common analysis window (per gauge, IDENTICAL for both sources):
    window = streamflow span  ∩  MRMS coverage   (clip-to-MRMS-era)
    Both sources are scored over this same window, so a gauge's flood-event count
    is identical regardless of precipitation source.  For station_nearest the
    nearest ISD/GHCNh gauge whose record SPANS this window is used (walk outward
    by distance, skipping any that don't cover it); a gauge with no covering
    precip station is omitted from the station source only.

Scope: only the LP3-fitted, clustered stations (clusters/clusters_k3.csv); each
output row carries the station's basin-type cluster so results can be pooled by
cluster.  Stations without valid Q (regulated / insufficient record) have null
thresholds and are skipped automatically — no separate QC.

Output schema (one row per combination):
    site_no, cluster, source, duration_hr, precip_rp_yr, flow_rp_yr,
    tp, fp, fn, tn, n_precip_events, n_flow_events,
    pct_precip_missing,                 # % of window hours with no precip record
    precip_agg,                         # hourly-binning method (mrms / max / sum / none)
    common_start, common_end, n_common_hours

Writes:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix.parquet
"""
from __future__ import annotations

import argparse
import io
import logging
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("08_events")

LEAD_HOURS       = 24    # precip may precede the flood by up to this many hours
MERGE_GAP_HOURS  = 24    # wet runs separated by fewer dry hours are one event

DURATIONS_HR = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 168, 240, 480, 720, 1080, 1440]
PRECIP_RPS   = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
FLOW_RPS     = [10, 50, 100]

OUTPUT_KEY = "analysis/event_confusion_matrix.parquet"
COMPLETE_COMBINATIONS = len(DURATIONS_HR) * len(PRECIP_RPS) * len(FLOW_RPS)  # 480

Source = Literal["nearest", "station_nearest"]


# ---------- Data loaders ----------

_s3_fs: pafs.S3FileSystem | None = None


def _read_parquet_s3(bucket: str, key: str, columns: list[str] | None = None) -> pd.DataFrame:
    global _s3_fs
    if _s3_fs is None:
        _s3_fs = pafs.S3FileSystem()
    return pq.read_table(f"{bucket}/{key}", filesystem=_s3_fs, columns=columns).to_pandas()


def load_mrms_nearest(bucket: str, prefix: str, product_key: str) -> pd.DataFrame:
    key = f"{prefix}mrms/{product_key}/nearest_pixel.parquet"
    df = _read_parquet_s3(bucket, key)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.rename(columns={"value": "precip_in"})
    df["site_no"] = df["site_no"].astype(str)
    return df[["datetime_utc", "site_no", "precip_in"]]


def load_streamflow(bucket: str, prefix: str) -> pd.DataFrame:
    """Per-gauge streamflow, resampled to hourly max to match the hourly precip grid."""
    key = f"{prefix}streamflow/instantaneous/all_gauges_long.parquet"
    df = _read_parquet_s3(bucket, key, columns=["site_no", "datetime", "value_cfs"])
    df["datetime"]  = pd.to_datetime(df["datetime"], utc=True)
    df["value_cfs"] = pd.to_numeric(df["value_cfs"], errors="coerce")
    df["site_no"]   = df["site_no"].astype(str)
    df = (
        df.set_index("datetime")
        .groupby("site_no")["value_cfs"]
        .resample("1h").max()
        .reset_index()
        .rename(columns={"datetime": "datetime_utc"})
    )
    return df


def load_atlas14(bucket: str, prefix: str) -> pd.DataFrame:
    df = _read_parquet_s3(bucket, f"{prefix}atlas14/precipitation_frequency.parquet")
    df["site_no"] = df["site_no"].astype(str)
    return df


def load_flow_stats(bucket: str, prefix: str) -> pd.DataFrame:
    df = _read_parquet_s3(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    df["site_no"] = df["site_no"].astype(str)
    cols = ["site_no"] + [f"Q{rp}" for rp in FLOW_RPS if f"Q{rp}" in df.columns]
    return df[cols]


def load_clusters(bucket: str, prefix: str) -> dict[str, int]:
    """Map site_no → basin-type cluster from clusters/clusters_k3.csv."""
    obj = s3_client().get_object(Bucket=bucket, Key=f"{prefix}clusters/clusters_k3.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype={"site_no": str})
    return dict(zip(df["site_no"], df["cluster"].astype(int)))


def load_gauges(bucket: str, prefix: str) -> pd.DataFrame:
    df = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites_active.parquet")
    df["site_no"] = df["site_no"].astype(str)
    return df[["site_no", "dec_lat_va", "dec_long_va"]].dropna().reset_index(drop=True)


_STATION_SOURCES_CFG = [
    ("isd",   "precip/noaa/isd_hourly.parquet",   "station_id"),
    ("ghcnh", "precip/noaa/ghcnh_hourly.parquet", "station_id"),
]


def load_station_meta(bucket: str, prefix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tag, fname, id_col in _STATION_SOURCES_CFG:
        try:
            df = _read_parquet_s3(bucket, f"{prefix}{fname}",
                                  columns=[id_col, "latitude", "longitude"])
            df = df.rename(columns={id_col: "_sid"})
            df["station_id"] = tag + "_" + df["_sid"].astype(str)
            frames.append(
                df[["station_id", "latitude", "longitude"]]
                .dropna(subset=["latitude", "longitude"])
                .drop_duplicates("station_id")
            )
        except Exception as e:
            log.warning("Could not load %s station metadata: %s", tag, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("station_id").reset_index(drop=True)


def load_precip_for_stations(bucket: str, prefix: str, station_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tag, fname, id_col in _STATION_SOURCES_CFG:
        try:
            df = _read_parquet_s3(bucket, f"{prefix}{fname}")
            df = df.rename(columns={id_col: "_sid"})
            df["station_id"] = tag + "_" + df["_sid"].astype(str)
            df = df[df["station_id"].isin(station_ids)]
            df["precip_in"] = pd.to_numeric(df["precip_in"], errors="coerce")
            frames.append(df[["station_id", "datetime_utc", "precip_in"]].copy())
            log.info("Loaded %s precip: %d rows, %d stations",
                     tag, len(df), df["station_id"].nunique())
        except Exception as e:
            log.warning("Could not load %s precip: %s", tag, e)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    return combined


def assign_nearest_station(gauges: pd.DataFrame, stations_meta: pd.DataFrame) -> dict[str, str]:
    from scipy.spatial import KDTree
    if stations_meta.empty:
        return {}
    kd = KDTree(stations_meta[["latitude", "longitude"]].values)
    result: dict[str, str] = {}
    for _, row in gauges.iterrows():
        _, idx = kd.query([float(row["dec_lat_va"]), float(row["dec_long_va"])])
        result[str(row["site_no"])] = stations_meta.iloc[idx]["station_id"]
    return result


def load_station_coverage(bucket: str, prefix: str) -> pd.DataFrame:
    """station_id, latitude, longitude, and record [start, end] for each precip gauge.

    Reads only id/datetime/lat/lon so the period of record can be computed without
    pulling the full precip values.
    """
    frames: list[pd.DataFrame] = []
    for tag, fname, id_col in _STATION_SOURCES_CFG:
        try:
            df = _read_parquet_s3(bucket, f"{prefix}{fname}",
                                  columns=[id_col, "datetime_utc", "latitude", "longitude"])
            df = df.rename(columns={id_col: "_sid"})
            df["station_id"]   = tag + "_" + df["_sid"].astype(str)
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            g = df.groupby("station_id")
            cov = pd.DataFrame({
                "latitude":  g["latitude"].first(),
                "longitude": g["longitude"].first(),
                "start":     g["datetime_utc"].min(),
                "end":       g["datetime_utc"].max(),
            }).reset_index()
            frames.append(cov.dropna(subset=["latitude", "longitude"]))
        except Exception as e:
            log.warning("Could not load %s coverage: %s", tag, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("station_id").reset_index(drop=True)


def assign_covering_station(
    gauges: pd.DataFrame,
    coverage: pd.DataFrame,
    window_by_site: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, str]:
    """Map each gauge to the NEAREST precip station whose record covers the gauge's
    analysis window [window_start, window_end].  Walks outward by distance, skipping
    stations that don't span the window; returns {} entry omitted if none qualifies.
    """
    from scipy.spatial import KDTree
    if coverage.empty:
        return {}
    kd = KDTree(coverage[["latitude", "longitude"]].values)
    starts = coverage["start"].to_numpy()
    ends   = coverage["end"].to_numpy()
    ids    = coverage["station_id"].to_numpy()
    k_all  = len(coverage)

    result: dict[str, str] = {}
    for _, row in gauges.iterrows():
        site = str(row["site_no"])
        win = window_by_site.get(site)
        if win is None:
            continue
        ws, we = win
        _, idxs = kd.query([float(row["dec_lat_va"]), float(row["dec_long_va"])], k=k_all)
        for idx in np.atleast_1d(idxs):
            if starts[idx] <= ws and ends[idx] >= we:   # covers the whole window
                result[site] = str(ids[idx])
                break
    return result


def resample_station_precip(precip: pd.Series) -> tuple[pd.Series, str]:
    """Bin a (sub-hourly, NaN-padded) station precip series to a clean hourly series.

    ISD/GHCNh precipitation is ACCUMULATION-based, never independent sub-hourly
    increments: a station reports the depth over the preceding hour, either once
    per hour or repeated across sub-hourly METAR rows as a running 1-hour total.
    So the correct hourly value is the per-hour MAX (skip-NaN) — it recovers the
    single hourly report, and for running-accumulation stations takes the largest
    trailing-hour total.  (Summing would multiply overlapping accumulations.)
    An hour with no report stays NaN (truly missing).
    """
    if precip.empty:
        return precip, "none"
    hourly = precip.groupby(precip.index.floor("h")).max()   # skip-NaN; NaN if all-NaN
    return hourly, "max"


# ---------- Event detection & classification ----------

def group_wet_events(is_wet: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge contiguous wet hours into (start, end) events.

    Two wet runs are merged into one event when the dry gap between them is
    shorter than MERGE_GAP_HOURS.  ``is_wet`` must be a boolean Series on a
    regular hourly DatetimeIndex.
    """
    wet_times = is_wet.index[is_wet.values.astype(bool)]
    if len(wet_times) == 0:
        return []
    merge_gap = pd.Timedelta(hours=MERGE_GAP_HOURS)
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = wet_times[0]
    for t in wet_times[1:]:
        if t - prev <= merge_gap:
            prev = t
        else:
            events.append((start, prev))
            start = prev = t
    events.append((start, prev))
    return events


def count_dry_periods(
    combined: list[tuple[pd.Timestamp, pd.Timestamp]],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
) -> int:
    """Edge-inclusive TN: every maximal dry stretch counts as one.

    Counts the interior gaps BETWEEN combined wet events, PLUS the leading dry
    period (record start → first event) and the trailing dry period (last event
    → record end).  A record with no wet event at all is one quiet stretch → 1.
    """
    if not combined:
        return 1
    tn = len(combined) - 1                       # interior gaps
    if combined[0][0] > grid_start:
        tn += 1                                  # leading dry
    if combined[-1][1] < grid_end:
        tn += 1                                  # trailing dry
    return tn


def merge_spans(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Union of (start, end) intervals, merging any that touch or overlap."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def build_episodes(
    precip_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    flow_events: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[dict]:
    """Merge overlapping precip & flow events into classified episodes.

    A precip event [ps,pe] and a flow event [qs,qe] are LINKED when they overlap
    with precipitation allowed to lead the flood by up to LEAD_HOURS:

        ps <= qe   AND   pe + LEAD_HOURS >= qs

    The two boundary cases that still link (→ one TP):
      • last P exactly LEAD_HOURS before first Q   (pe + lead == qs)
      • first P exactly at last Q                  (ps == qe)

    Linked events are unioned into episodes (a shared precip can pull several
    floods into one episode).  Each episode spans first-wet → last-wet and is:
        TP  contains ≥1 precip AND ≥1 flow event
        FN  contains only flow events (flood with no qualifying precip)
        FP  contains only precip events (rain with no following flood)
    """
    lead = pd.Timedelta(hours=LEAD_HOURS)
    nP, nF = len(precip_events), len(flow_events)
    parent = list(range(nP + nF))           # 0..nP-1 precip, nP..nP+nF-1 flow

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, (ps, pe) in enumerate(precip_events):
        for j, (qs, qe) in enumerate(flow_events):
            if ps <= qe and pe + lead >= qs:
                union(i, nP + j)

    comp: dict[int, dict] = {}

    def add(root: int, kind: str, s: pd.Timestamp, e: pd.Timestamp) -> None:
        c = comp.get(root)
        if c is None:
            comp[root] = {"P": 0, "F": 0, "start": s, "end": e}
            c = comp[root]
        c[kind] += 1
        c["start"] = min(c["start"], s)
        c["end"]   = max(c["end"], e)

    for i, (ps, pe) in enumerate(precip_events):
        add(find(i), "P", ps, pe)
    for j, (qs, qe) in enumerate(flow_events):
        add(find(nP + j), "F", qs, qe)

    episodes = []
    for c in comp.values():
        if c["P"] and c["F"]:
            # TP spans first-wet → last-wet (precip→flood already merged)
            klass, start, end = "TP", c["start"], c["end"]
        elif c["F"]:
            # FN owns the 24h look-back window (where qualifying precip was absent):
            # from 24h before the first Q>threshold to the last Q>threshold.
            klass, start, end = "FN", c["start"] - lead, c["end"]
        else:
            # FP owns the 24h look-forward window (where no flood followed):
            # from the first P>threshold to 24h after the last P>threshold.
            klass, start, end = "FP", c["start"], c["end"] + lead
        episodes.append({"start": start, "end": end, "klass": klass,
                         "n_precip": c["P"], "n_flow": c["F"]})
    episodes.sort(key=lambda d: d["start"])
    return episodes


def classify_overlap(
    precip_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    flow_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn): episodes classified, TN = edge-inclusive dry gaps."""
    eps = build_episodes(precip_events, flow_events)
    tp = sum(e["klass"] == "TP" for e in eps)
    fn = sum(e["klass"] == "FN" for e in eps)
    fp = sum(e["klass"] == "FP" for e in eps)
    spans = merge_spans([(e["start"], e["end"]) for e in eps])
    tn = count_dry_periods(spans, grid_start, grid_end)
    return tp, fp, fn, tn


# ---------- Per-station analysis ----------

def analyse_station(
    site_no: str,
    cluster: int,
    precip_hourly: pd.Series,    # index=datetime_utc, precip_in
    flow_hourly: pd.Series,      # index=datetime_utc, value_cfs (hourly max)
    atlas14_site: pd.DataFrame,
    flow_stats_row: pd.Series,
    source: Source,
    window_start: pd.Timestamp,  # analysis window = streamflow span ∩ MRMS coverage
    window_end: pd.Timestamp,    # IDENTICAL for both sources → identical flood counts
    precip_agg: str = "mrms",    # hourly-binning method for the precip series
) -> list[dict]:
    if pd.isna(window_start) or pd.isna(window_end) or window_start >= window_end:
        log.warning("[%s] %s: empty window — skipping", source, site_no)
        return []
    common_start, common_end = window_start, window_end

    grid = pd.date_range(common_start, common_end, freq="1h", tz="UTC")
    precip = precip_hourly.reindex(grid)                          # NaN where no record
    pct_precip_missing = round(100.0 * float(precip.isna().mean()), 2)
    precip = precip.fillna(0.0)                                   # missing precip = 0
    flow   = flow_hourly.reindex(grid)                            # keep NaN gaps
    n_common = len(grid)

    # Pre-compute flow wet masks + events per flow return period (depend only on Q_rp)
    flow_wet_by_rp: dict[int, pd.Series] = {}
    flow_events_by_rp: dict[int, list] = {}
    for flow_rp in FLOW_RPS:
        q_col = f"Q{flow_rp}"
        if q_col not in flow_stats_row or pd.isna(flow_stats_row[q_col]):
            continue
        q_thr = float(flow_stats_row[q_col])
        wet = (flow >= q_thr).fillna(False)
        flow_wet_by_rp[flow_rp] = wet
        flow_events_by_rp[flow_rp] = group_wet_events(wet)

    if not flow_wet_by_rp:
        return []

    records: list[dict] = []
    for duration_hr in DURATIONS_HR:
        rolling = precip.rolling(window=duration_hr, min_periods=duration_hr).sum()
        for precip_rp in PRECIP_RPS:
            a14 = atlas14_site.loc[
                (atlas14_site["duration_hr"] == duration_hr)
                & (atlas14_site["return_period_yr"] == precip_rp)
            ]
            if a14.empty:
                continue
            precip_thr = float(a14["depth_in"].iloc[0])
            precip_wet = (rolling >= precip_thr).fillna(False)
            precip_events = group_wet_events(precip_wet)

            for flow_rp in flow_wet_by_rp:
                tp, fp, fn, tn = classify_overlap(
                    precip_events, flow_events_by_rp[flow_rp],
                    common_start, common_end,
                )
                records.append({
                    "site_no":         site_no,
                    "cluster":         cluster,
                    "source":          source,
                    "duration_hr":     duration_hr,
                    "precip_rp_yr":    precip_rp,
                    "flow_rp_yr":      flow_rp,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "n_precip_events": len(precip_events),
                    "n_flow_events":   len(flow_events_by_rp[flow_rp]),
                    "pct_precip_missing": pct_precip_missing,
                    "precip_agg":      precip_agg,
                    "common_start":    common_start,
                    "common_end":      common_end,
                    "n_common_hours":  n_common,
                })
    return records


# ---------- Resume helpers ----------

def load_existing(bucket: str, prefix: str):
    """Return (existing_df_or_None, complete_keys, stored_end) for resume."""
    try:
        existing = _read_parquet_s3(bucket, f"{prefix}{OUTPUT_KEY}")
    except Exception:
        log.info("No existing results — running fresh.")
        return None, set(), {}
    existing["site_no"] = existing["site_no"].astype(str)
    counts = existing.groupby(["site_no", "source"]).size()
    complete = {key for key, n in counts.items() if n == COMPLETE_COMBINATIONS}
    stored_end = {
        key: grp["common_end"].max()
        for key, grp in existing.groupby(["site_no", "source"]) if key in complete
    }
    log.info("Existing: %d rows | %d complete (site,source) pairs", len(existing), len(complete))
    return existing, complete, stored_end


# ---------- Diagnostic plot ----------

def get_precip_series(bucket: str, prefix: str, product_key: str,
                      site_no: str, source: Source) -> pd.Series:
    """Hourly precip series for one gauge from the chosen source."""
    if source == "nearest":
        mrms = load_mrms_nearest(bucket, prefix, product_key)
        return mrms[mrms["site_no"] == site_no].set_index("datetime_utc")["precip_in"].sort_index()
    meta   = load_station_meta(bucket, prefix)
    gauges = load_gauges(bucket, prefix)
    nearest = assign_nearest_station(gauges[gauges["site_no"] == site_no], meta)
    sid = nearest.get(site_no)
    if not sid:
        return pd.Series(dtype=float)
    precip = load_precip_for_stations(bucket, prefix, {sid})
    if precip.empty:
        return pd.Series(dtype=float)
    return precip[precip["station_id"] == sid].set_index("datetime_utc")["precip_in"].sort_index()


def plot_station_diagnostic(
    bucket: str, prefix: str, product_key: str, site_no: str, source: Source,
    duration_hr: int, precip_rp: int, flow_rp: int,
    start: str | None = None, end: str | None = None,
) -> None:
    """Hyetograph-over-hydrograph with classification shading, to eyeball one combo.

    Streamflow: bottom x-axis, left y-axis.  Precip (the D-hour accumulation):
    top x-axis, right y-axis inverted (bars hang from the top).  Q and P
    thresholds are dashed horizontal lines; wet events are shaded by class.
    """
    site_no = str(site_no)
    atlas14    = load_atlas14(bucket, prefix)
    flow_stats = load_flow_stats(bucket, prefix)
    sf = load_streamflow(bucket, prefix)
    flow_raw = sf[sf["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
    precip_raw = get_precip_series(bucket, prefix, product_key, site_no, source)
    if precip_raw.empty or flow_raw.empty:
        log.error("No data for site %s (%s)", site_no, source)
        return

    a14 = atlas14[(atlas14["site_no"] == site_no)
                  & (atlas14["duration_hr"] == duration_hr)
                  & (atlas14["return_period_yr"] == precip_rp)]
    fs = flow_stats[flow_stats["site_no"] == site_no]
    if a14.empty or fs.empty or pd.isna(fs.iloc[0].get(f"Q{flow_rp}")):
        log.error("Missing Atlas14 / Q%d for site %s", flow_rp, site_no)
        return
    precip_thr = float(a14["depth_in"].iloc[0])
    q_thr      = float(fs.iloc[0][f"Q{flow_rp}"])

    cs = max(precip_raw.index.min(), flow_raw.index.min())
    ce = min(precip_raw.index.max(), flow_raw.index.max())
    grid   = pd.date_range(cs, ce, freq="1h", tz="UTC")
    precip = precip_raw.reindex(grid)
    pct_precip_missing = round(100.0 * float(precip.isna().mean()), 1)
    precip = precip.fillna(0.0)
    flow   = flow_raw.reindex(grid)
    rolling = precip.rolling(duration_hr, min_periods=duration_hr).sum()
    precip_wet = (rolling >= precip_thr).fillna(False)
    flow_wet   = (flow >= q_thr).fillna(False)
    precip_events = group_wet_events(precip_wet)
    flow_events   = group_wet_events(flow_wet)

    eps = build_episodes(precip_events, flow_events)

    # Auto-zoom to the calendar year with the most flood-wet hours (legibility)
    if start is None:
        if flow_wet.any():
            yr = pd.Series(flow_wet[flow_wet].index.year).value_counts().idxmax()
            view_s = pd.Timestamp(f"{yr}-01-01", tz="UTC")
            view_e = pd.Timestamp(f"{yr}-12-31 23:00", tz="UTC")
        else:
            view_s, view_e = cs, ce
    else:
        view_s = pd.Timestamp(start, tz="UTC")
        view_e = pd.Timestamp(end, tz="UTC") if end else ce

    fig, ax1 = plt.subplots(figsize=(16, 6))
    # Hydrograph — bottom x, left y
    ax1.plot(grid, flow, color="steelblue", lw=0.8, zorder=4, label="Streamflow")
    ax1.axhline(q_thr, color="navy", ls="--", lw=1.3, zorder=5,
                label=f"Q{flow_rp} = {q_thr:,.0f} cfs")
    ax1.set_ylabel("Streamflow (cfs)", color="steelblue")
    ax1.set_xlabel("Date (UTC)")
    ax1.set_xlim(view_s, view_e)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.secondary_xaxis("top")   # mirror time on the upper x-axis

    # Hyetograph — top x (mirrored), right y inverted (increasing downward)
    ax2 = ax1.twinx()
    accum = precip if duration_hr == 1 else rolling
    ax2.bar(grid, accum, width=0.045, color="teal", alpha=0.55, align="center", zorder=2)
    ax2.axhline(precip_thr, color="darkred", ls="--", lw=1.3, zorder=5)
    label = "Hourly precip" if duration_hr == 1 else f"{duration_hr}-h accum precip"
    ax2.set_ylabel(f"{label} (in)\nP{precip_rp} = {precip_thr:.2f} in", color="teal")
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelcolor="teal")

    # Classification shading — one span per merged episode
    span_colors = {"TP": "green", "FN": "red", "FP": "orange"}
    for e in eps:
        ax1.axvspan(e["start"], e["end"], color=span_colors[e["klass"]], alpha=0.28, zorder=1)

    tp = sum(e["klass"] == "TP" for e in eps)
    fn = sum(e["klass"] == "FN" for e in eps)
    fp = sum(e["klass"] == "FP" for e in eps)
    tn = count_dry_periods(merge_spans([(e["start"], e["end"]) for e in eps]), cs, ce)

    legend = [
        Patch(facecolor="green",  alpha=.45, label="TP — flood with precip ≤24h prior"),
        Patch(facecolor="red",    alpha=.45, label="FN — flood, no precip (+24h look-back)"),
        Patch(facecolor="orange", alpha=.45, label="FP — precip, no flood (+24h look-ahead)"),
    ]
    h1, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + legend, loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_title(
        f"Station {site_no} | {source} | D={duration_hr}h | P{precip_rp}→Q{flow_rp}\n"
        f"whole record: TP={tp}  FP={fp}  FN={fn}  TN={tn}   "
        f"({pct_precip_missing}% precip hrs missing)   "
        f"(view {view_s.date()} → {view_e.date()})"
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()

    base = f"{site_no}_{source}_d{duration_hr}_P{precip_rp}_Q{flow_rp}"
    out_prefix = f"{prefix}analysis/event_diagnostics/"

    # PNG (raster) + SVG (vector, infinite zoom in any viewer)
    for ext, ctype in (("png", "image/png"), ("svg", "image/svg+xml")):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=140)
        buf.seek(0)
        s3_client().put_object(Bucket=bucket, Key=f"{out_prefix}{base}.{ext}",
                               Body=buf, ContentType=ctype)
    plt.close(fig)

    # MATLAB data export (.mat): rebuild / zoom natively in MATLAB
    import numpy as np
    from scipy.io import savemat

    def _epoch(idx) -> "np.ndarray":
        return (pd.DatetimeIndex(idx).view("int64") // 10**9).astype("float64")

    def _ev_epoch(evs):
        if not evs:
            return np.array([]), np.array([])
        return _epoch([e[0] for e in evs]), _epoch([e[1] for e in evs])

    ep_s = _epoch([e["start"] for e in eps]) if eps else np.array([])
    ep_e = _epoch([e["end"]   for e in eps]) if eps else np.array([])
    ep_c = np.array([e["klass"] for e in eps], dtype=object) if eps else np.array([], dtype=object)
    pf_s, pf_e = _ev_epoch(precip_events)
    qf_s, qf_e = _ev_epoch(flow_events)
    mat = {
        "site_no": site_no, "source": source,
        "duration_hr": duration_hr, "precip_rp": precip_rp, "flow_rp": flow_rp,
        "time_unix_s":        _epoch(grid),
        "streamflow_cfs":     flow.to_numpy(dtype="float64"),
        "precip_accum_in":    accum.to_numpy(dtype="float64"),
        "q_threshold_cfs":    q_thr,
        "p_threshold_in":     precip_thr,
        "episode_start_s": ep_s, "episode_end_s": ep_e, "episode_class": ep_c,
        "precip_event_start_s": pf_s, "precip_event_end_s": pf_e,
        "flow_event_start_s":   qf_s, "flow_event_end_s":   qf_e,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "readme": ("time_unix_s is UTC seconds. In MATLAB: "
                   "t = datetime(time_unix_s,'ConvertFrom','posixtime','TimeZone','UTC'). "
                   "episode_class is TP/FN/FP; shade each [episode_start_s, episode_end_s]."),
    }
    mbuf = io.BytesIO()
    savemat(mbuf, mat)
    mbuf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=f"{out_prefix}{base}.mat",
                           Body=mbuf, ContentType="application/octet-stream")

    log.info("Diagnostic → s3://%s/%s{%s.png,.svg,.mat}", bucket, out_prefix, base)


# ---------- Main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    log.info("Loading inputs...")
    atlas14    = load_atlas14(bucket, prefix)
    flow_stats = load_flow_stats(bucket, prefix)
    streamflow = load_streamflow(bucket, prefix)
    clusters   = load_clusters(bucket, prefix)

    q_cols = [f"Q{rp}" for rp in FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q  = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])

    # Only clustered, LP3-fitted stations with at least one valid Q threshold.
    stations_all = sorted(
        set(clusters)
        & has_q
        & set(atlas14["site_no"])
        & set(streamflow["site_no"])
    )
    log.info("Clustered stations with all inputs: %d", len(stations_all))

    flow_start_by_site = streamflow.groupby("site_no")["datetime_utc"].min()
    flow_end_by_site   = streamflow.groupby("site_no")["datetime_utc"].max()

    # ── Per-gauge analysis window = streamflow span ∩ MRMS coverage ───────────
    # Computed ONCE and used for BOTH sources, so the flood-event count for a
    # gauge is identical regardless of precipitation source.
    log.info("Loading MRMS nearest-pixel precipitation...")
    try:
        mrms = load_mrms_nearest(bucket, prefix, product_key)
    except Exception as e:
        log.error("Could not load MRMS nearest: %s", e)
        mrms = pd.DataFrame()

    mrms_start_by_site = mrms.groupby("site_no")["datetime_utc"].min() if not mrms.empty else {}
    mrms_end_by_site   = mrms.groupby("site_no")["datetime_utc"].max() if not mrms.empty else {}

    window_by_site: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in stations_all:
        fs_, fe_ = flow_start_by_site.get(s, pd.NaT), flow_end_by_site.get(s, pd.NaT)
        ms_ = mrms_start_by_site.get(s, pd.NaT) if len(mrms_start_by_site) else pd.NaT
        me_ = mrms_end_by_site.get(s, pd.NaT) if len(mrms_end_by_site) else pd.NaT
        if any(pd.isna(x) for x in (fs_, fe_, ms_, me_)):
            continue
        ws, we = max(fs_, ms_), min(fe_, me_)
        if ws < we:
            window_by_site[s] = (ws, we)
    sites_win = [s for s in stations_all if s in window_by_site]
    log.info("Gauges with a valid flow∩MRMS window: %d / %d", len(sites_win), len(stations_all))

    existing, complete_keys, stored_end = load_existing(bucket, prefix)
    all_records: list[dict] = []

    def _resume_skip(site_no: str, source: str, we: pd.Timestamp) -> bool:
        if (site_no, source) in complete_keys:
            if stored_end.get((site_no, source), pd.NaT) >= we:
                return True
            complete_keys.discard((site_no, source))
        return False

    # ── Source 1: nearest MRMS pixel ──────────────────────────────────────────
    if not mrms.empty:
        n = len(sites_win)
        for i, site_no in enumerate(sites_win, 1):
            ws, we = window_by_site[site_no]
            if _resume_skip(site_no, "nearest", we):
                log.info("[nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                continue
            precip_site = mrms[mrms["site_no"] == site_no].set_index("datetime_utc")["precip_in"].sort_index()
            flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
            a14_site    = atlas14[atlas14["site_no"] == site_no]
            fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
            if precip_site.empty or flow_site.empty or a14_site.empty or fs_rows.empty:
                log.warning("[nearest][%d/%d] %s: missing data, skipping", i, n, site_no)
                continue
            recs = analyse_station(site_no, clusters[site_no], precip_site, flow_site,
                                   a14_site, fs_rows.iloc[0], "nearest", ws, we)
            all_records.extend(recs)
            log.info("[nearest][%d/%d] %s: %d combos", i, n, site_no, len(recs))

    # ── Source 2: nearest station COVERING the window (ISD / GHCNh) ────────────
    log.info("Loading station coverage (ISD + GHCNh)...")
    coverage = load_station_coverage(bucket, prefix)
    if coverage.empty:
        log.warning("No station precip — skipping station_nearest source.")
    else:
        gauges = load_gauges(bucket, prefix)
        assign = assign_covering_station(gauges, coverage, window_by_site)
        n_missing = len(sites_win) - len(set(sites_win) & set(assign))
        log.info("Covering precip station found for %d / %d gauges (%d have none)",
                 len(assign), len(sites_win), n_missing)
        needed = set(assign.values())
        log.info("Loading precip for %d covering stations...", len(needed))
        precip_hourly = load_precip_for_stations(bucket, prefix, needed)
        if precip_hourly.empty:
            log.warning("No station precip rows — skipping station_nearest.")
        else:
            sites_st = [s for s in sites_win if s in assign]
            n = len(sites_st)
            resamp_cache: dict[str, tuple[pd.Series, str]] = {}   # sid → (hourly, agg)
            for i, site_no in enumerate(sites_st, 1):
                ws, we = window_by_site[site_no]
                if _resume_skip(site_no, "station_nearest", we):
                    log.info("[station_nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                    continue
                sid = assign[site_no]
                if sid not in resamp_cache:
                    sub = precip_hourly[precip_hourly["station_id"] == sid]
                    if sub.empty:
                        resamp_cache[sid] = (pd.Series(dtype=float), "none")
                    else:
                        resamp_cache[sid] = resample_station_precip(
                            sub.set_index("datetime_utc")["precip_in"].sort_index())
                precip_site, agg_method = resamp_cache[sid]
                if precip_site.empty:
                    log.warning("[station_nearest][%d/%d] %s: no precip, skipping", i, n, site_no)
                    continue
                flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
                a14_site    = atlas14[atlas14["site_no"] == site_no]
                fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
                if flow_site.empty or a14_site.empty or fs_rows.empty:
                    continue
                recs = analyse_station(site_no, clusters[site_no], precip_site, flow_site,
                                       a14_site, fs_rows.iloc[0], "station_nearest", ws, we,
                                       precip_agg=agg_method)
                all_records.extend(recs)
                log.info("[station_nearest][%d/%d] %s: %d combos (agg=%s)",
                         i, n, site_no, len(recs), agg_method)

    # ── Combine with retained complete pairs and write ────────────────────────
    parts: list[pd.DataFrame] = []
    if existing is not None and complete_keys:
        kept = existing[
            existing[["site_no", "source"]].apply(
                lambda r: (r["site_no"], r["source"]) in complete_keys, axis=1
            )
        ]
        parts.append(kept)
        log.info("Retaining %d rows from previous run", len(kept))
    if all_records:
        parts.append(pd.DataFrame(all_records))

    if not parts:
        log.error("No results produced.")
        return

    out = pd.concat(parts, ignore_index=True)
    write_parquet_to_s3(out, bucket, f"{prefix}{OUTPUT_KEY}")
    log.info("Wrote %s%s (%d rows, %d stations)",
             prefix, OUTPUT_KEY, len(out), out["site_no"].nunique())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", metavar="SITE_NO",
                   help="produce a diagnostic figure for ONE station instead of the full run")
    p.add_argument("--source", default="nearest",
                   choices=["nearest", "station_nearest"])
    p.add_argument("--duration", type=int, default=1, help="accumulation hours (default 1)")
    p.add_argument("--precip-rp", type=int, default=10, help="precip return period (default 10)")
    p.add_argument("--flow-rp", type=int, default=10, help="flow return period (default 10)")
    p.add_argument("--start", default=None, help="view window start (YYYY-MM-DD); default auto-zoom")
    p.add_argument("--end", default=None, help="view window end (YYYY-MM-DD)")
    args = p.parse_args()

    if args.plot:
        cfg = load_config()
        plot_station_diagnostic(
            cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"],
            cfg["mrms"]["products"][0]["key"],
            args.plot, args.source, args.duration, args.precip_rp, args.flow_rp,
            args.start, args.end,
        )
    else:
        main()
