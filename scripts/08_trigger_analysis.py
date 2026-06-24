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
    station_nearest  nearest ISD/GHCNh hourly gauge

Scope: only the LP3-fitted, clustered stations (clusters/clusters_k3.csv); each
output row carries the station's basin-type cluster so results can be pooled by
cluster.  Stations without valid Q (regulated / insufficient record) have null
thresholds and are skipped automatically — no separate QC.

Output schema (one row per combination):
    site_no, cluster, source, duration_hr, precip_rp_yr, flow_rp_yr,
    tp, fp, fn, tn, n_precip_events, n_flow_events,
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


def classify_overlap(
    precip_wet: pd.Series,
    flow_wet: pd.Series,
    precip_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    flow_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for one combination.

    A precip event leads a flood by ≤ LEAD_HOURS (or is simultaneous):
      TP/FN  per flood event — does any precip event overlap [flood_start−lead,
             flood_end]?
      FP     per precip event — is there NO flood event in [precip_start,
             precip_end+lead]?
      TN     every dry stretch (leading, interior, trailing) between combined
             (precip|flow) wet events — see count_dry_periods.
    """
    lead = pd.Timedelta(hours=LEAD_HOURS)

    tp = fn = 0
    for qs, qe in flow_events:
        lo = qs - lead
        if any(ps <= qe and pe >= lo for ps, pe in precip_events):
            tp += 1
        else:
            fn += 1

    fp = 0
    for ps, pe in precip_events:
        hi = pe + lead
        if not any(qs <= hi and qe >= ps for qs, qe in flow_events):
            fp += 1

    combined = group_wet_events(precip_wet | flow_wet)
    tn = count_dry_periods(combined, grid_start, grid_end)
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
) -> list[dict]:
    common_start = max(precip_hourly.index.min(), flow_hourly.index.min())
    common_end   = min(precip_hourly.index.max(), flow_hourly.index.max())
    if pd.isna(common_start) or pd.isna(common_end) or common_start >= common_end:
        log.warning("[%s] %s: no common period — skipping", source, site_no)
        return []

    grid = pd.date_range(common_start, common_end, freq="1h", tz="UTC")
    precip = precip_hourly.reindex(grid, fill_value=0.0)          # missing precip = 0
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

            for flow_rp, flow_wet in flow_wet_by_rp.items():
                tp, fp, fn, tn = classify_overlap(
                    precip_wet, flow_wet, precip_events, flow_events_by_rp[flow_rp],
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
    precip = precip_raw.reindex(grid, fill_value=0.0)
    flow   = flow_raw.reindex(grid)
    rolling = precip.rolling(duration_hr, min_periods=duration_hr).sum()
    precip_wet = (rolling >= precip_thr).fillna(False)
    flow_wet   = (flow >= q_thr).fillna(False)
    precip_events = group_wet_events(precip_wet)
    flow_events   = group_wet_events(flow_wet)

    lead = pd.Timedelta(hours=LEAD_HOURS)
    flow_cls = [(qs, qe, "TP" if any(ps <= qe and pe >= qs - lead for ps, pe in precip_events)
                 else "FN") for qs, qe in flow_events]
    precip_cls = [(ps, pe, "matched" if any(qs <= pe + lead and qe >= ps for qs, qe in flow_events)
                   else "FP") for ps, pe in precip_events]

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

    # Classification shading
    span_colors = {"TP": "green", "FN": "red", "FP": "orange"}
    for qs, qe, c in flow_cls:
        ax1.axvspan(qs, qe, color=span_colors[c], alpha=0.30, zorder=1)
    for ps, pe, c in precip_cls:
        if c == "FP":
            ax1.axvspan(ps, pe, color=span_colors["FP"], alpha=0.20, zorder=1)

    tp = sum(c == "TP" for *_, c in flow_cls)
    fn = sum(c == "FN" for *_, c in flow_cls)
    fp = sum(c == "FP" for *_, c in precip_cls)
    tn = count_dry_periods(group_wet_events(precip_wet | flow_wet), cs, ce)

    legend = [
        Patch(facecolor="green",  alpha=.45, label="TP — flood with precip ≤24h prior"),
        Patch(facecolor="red",    alpha=.45, label="FN — flood, no precip"),
        Patch(facecolor="orange", alpha=.45, label="FP — precip, no flood ≤24h after"),
    ]
    h1, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + legend, loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_title(
        f"Station {site_no} | {source} | D={duration_hr}h | P{precip_rp}→Q{flow_rp}\n"
        f"whole record: TP={tp}  FP={fp}  FN={fn}  TN={tn}   "
        f"(view {view_s.date()} → {view_e.date()})"
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()

    fname = f"{site_no}_{source}_d{duration_hr}_P{precip_rp}_Q{flow_rp}.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket,
                           Key=f"{prefix}analysis/event_diagnostics/{fname}",
                           Body=buf, ContentType="image/png")
    plt.close(fig)
    log.info("Diagnostic → s3://%s/%sanalysis/event_diagnostics/%s", bucket, prefix, fname)


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

    flow_end_by_site = streamflow.groupby("site_no")["datetime_utc"].max()
    existing, complete_keys, stored_end = load_existing(bucket, prefix)

    all_records: list[dict] = []

    # ── Source 1: nearest MRMS pixel ──────────────────────────────────────────
    log.info("Loading MRMS nearest-pixel precipitation...")
    try:
        mrms = load_mrms_nearest(bucket, prefix, product_key)
        mrms_end_by_site = mrms.groupby("site_no")["datetime_utc"].max()
    except Exception as e:
        log.error("Could not load MRMS nearest: %s", e)
        mrms = pd.DataFrame()

    if not mrms.empty:
        n = len(stations_all)
        for i, site_no in enumerate(stations_all, 1):
            if (site_no, "nearest") in complete_keys:
                exp_end = min(mrms_end_by_site.get(site_no, pd.NaT),
                              flow_end_by_site.get(site_no, pd.NaT))
                if pd.notna(exp_end) and exp_end <= stored_end.get((site_no, "nearest"), pd.NaT):
                    log.info("[nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                    continue
                complete_keys.discard((site_no, "nearest"))

            precip_site = mrms[mrms["site_no"] == site_no].set_index("datetime_utc")["precip_in"].sort_index()
            flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
            a14_site    = atlas14[atlas14["site_no"] == site_no]
            fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
            if precip_site.empty or flow_site.empty or a14_site.empty or fs_rows.empty:
                log.warning("[nearest][%d/%d] %s: missing data, skipping", i, n, site_no)
                continue
            recs = analyse_station(site_no, clusters[site_no], precip_site, flow_site,
                                   a14_site, fs_rows.iloc[0], "nearest")
            all_records.extend(recs)
            log.info("[nearest][%d/%d] %s: %d combos", i, n, site_no, len(recs))

    # ── Source 2: nearest station (ISD / GHCNh) ───────────────────────────────
    log.info("Loading station metadata for nearest-station assignment...")
    stations_meta = load_station_meta(bucket, prefix)
    if stations_meta.empty:
        log.warning("No station precip — skipping station_nearest source.")
    else:
        gauges  = load_gauges(bucket, prefix)
        nearest = assign_nearest_station(gauges, stations_meta)
        needed  = {nearest[s] for s in stations_all if s in nearest}
        log.info("Loading precip for %d nearest stations...", len(needed))
        precip_hourly = load_precip_for_stations(bucket, prefix, needed)
        if precip_hourly.empty:
            log.warning("No station precip rows — skipping station_nearest.")
        else:
            precip_end_by_sid = precip_hourly.groupby("station_id")["datetime_utc"].max()
            n = len(stations_all)
            for i, site_no in enumerate(stations_all, 1):
                sid = nearest.get(site_no)
                if sid is None:
                    continue
                if (site_no, "station_nearest") in complete_keys:
                    exp_end = min(precip_end_by_sid.get(sid, pd.NaT),
                                  flow_end_by_site.get(site_no, pd.NaT))
                    if pd.notna(exp_end) and exp_end <= stored_end.get((site_no, "station_nearest"), pd.NaT):
                        log.info("[station_nearest][%d/%d] %s: complete, skipping", i, n, site_no)
                        continue
                    complete_keys.discard((site_no, "station_nearest"))

                sub = precip_hourly[precip_hourly["station_id"] == sid]
                if sub.empty:
                    log.warning("[station_nearest][%d/%d] %s: no precip, skipping", i, n, site_no)
                    continue
                precip_site = sub.set_index("datetime_utc")["precip_in"].sort_index()
                flow_site   = streamflow[streamflow["site_no"] == site_no].set_index("datetime_utc")["value_cfs"].sort_index()
                a14_site    = atlas14[atlas14["site_no"] == site_no]
                fs_rows     = flow_stats[flow_stats["site_no"] == site_no]
                if flow_site.empty or a14_site.empty or fs_rows.empty:
                    continue
                recs = analyse_station(site_no, clusters[site_no], precip_site, flow_site,
                                       a14_site, fs_rows.iloc[0], "station_nearest")
                all_records.extend(recs)
                log.info("[station_nearest][%d/%d] %s: %d combos", i, n, site_no, len(recs))

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
