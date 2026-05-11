"""08_trigger_analysis.py

Compare precipitation triggers derived from MRMS against observed USGS
streamflow to evaluate their ability to indicate potentially damaging
bridge events.

Trigger logic:
    1. Compute rolling accumulated MRMS precipitation for each Atlas 14
       duration (1 h to 60 days) using both nearest-pixel and watershed-mean
       MRMS sources.
    2. A trigger fires when the rolling sum exceeds an Atlas 14 threshold
       for a given return period (1 to 1000 years).
    3. Events are de-duplicated: two threshold exceedances are considered
       the same event unless separated by at least 24 consecutive dry hours
       (raw hourly MRMS < 0.1 in). Within each merged event the peak hour
       is the trigger time.
    4. A trigger is evaluated over a 24-hour response window: check whether
       max hourly streamflow in [trigger_time, trigger_time + 24 h] meets or
       exceeds a flow threshold (Q10 or Q50 from script 04).

Classification per trigger event:
    TP  trigger fired  AND  streamflow >= flow threshold
    FP  trigger fired  AND  streamflow <  flow threshold

Classification for missed events:
    FN  streamflow peak >= flow threshold  AND  no trigger in preceding 24 h

True negatives (hourly):
    TN  hours in common period with no active trigger AND streamflow < threshold

All metrics are computed only over the common period between MRMS and USGS
observations for each station.

USGS 15-min streamflow is resampled to hourly max before comparison to keep
the time axis coherent with the hourly MRMS data.

Output schema (one row per combination):
    site_no, mrms_source, duration_hr, precip_rp_yr, flow_rp_yr,
    n_trigger_events, tp, fp, fn, tn,
    common_start, common_end, n_common_hours

Writes:
    s3://<bucket>/<prefix>analysis/trigger_analysis.parquet
"""
from __future__ import annotations

import io
import logging
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("08_trigger")

DRY_THRESHOLD_IN = 0.1   # in/h below which an hour is considered dry
DRY_SPELL_HOURS  = 24    # min dry hours between independent events
RESPONSE_HOURS   = 24    # window after trigger to look for streamflow response

DURATIONS_HR = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 168, 240, 480, 720, 1080, 1440]
PRECIP_RPS   = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
FLOW_RPS     = [10, 50]

MrmsSource = Literal["nearest", "watershed"]


# ---------- Data loaders ----------

def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def load_mrms(bucket: str, prefix: str, product_key: str, source: MrmsSource) -> pd.DataFrame:
    fname = "nearest_pixel.parquet" if source == "nearest" else "watershed_mean.parquet"
    key = f"{prefix}mrms/{product_key}/{fname}"
    df = _read_parquet_s3(bucket, key)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.rename(columns={"value": "precip_in"})
    return df[["datetime_utc", "site_no", "precip_in"]]


def load_streamflow(bucket: str, prefix: str) -> pd.DataFrame:
    """Load all per-gauge streamflow and resample 15-min data to hourly max."""
    key = f"{prefix}streamflow/instantaneous/all_gauges_long.parquet"
    df = _read_parquet_s3(bucket, key)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["value_cfs"] = pd.to_numeric(df["value_cfs"], errors="coerce")

    # Resample to hourly max to match MRMS temporal resolution
    df = (
        df.set_index("datetime")
        .groupby("site_no")["value_cfs"]
        .resample("1h")
        .max()
        .reset_index()
        .rename(columns={"datetime": "datetime_utc"})
    )
    return df


def load_atlas14(bucket: str, prefix: str) -> pd.DataFrame:
    return _read_parquet_s3(bucket, f"{prefix}atlas14/precipitation_frequency.parquet")


def load_flow_stats(bucket: str, prefix: str) -> pd.DataFrame:
    """Load Q10 and Q50 per station from script 04 output."""
    df = _read_parquet_s3(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    df["site_no"] = df["site_no"].astype(str)
    return df[["site_no", "Q10", "Q50"]]


# ---------- Event detection ----------

def find_trigger_events(
    hourly_raw: pd.Series,
    rolling_sum: pd.Series,
    threshold: float,
) -> list[pd.Timestamp]:
    """Return a list of peak trigger timestamps, de-duplicated by 24-h dry spell.

    Parameters
    ----------
    hourly_raw:   raw hourly MRMS values (index = DatetimeIndex, UTC)
    rolling_sum:  trailing rolling sum for the target duration (same index)
    threshold:    Atlas 14 depth threshold in inches
    """
    exceeds = rolling_sum >= threshold
    if not exceeds.any():
        return []

    # Label each hour as dry (1) or wet (0) based on raw hourly value
    is_dry = (hourly_raw < DRY_THRESHOLD_IN).astype(int)

    # Build dry-spell run lengths ending at each hour
    dry_run = is_dry * (is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumcount() + 1)

    trigger_times: list[pd.Timestamp] = []
    in_event = False
    event_start: pd.Timestamp | None = None
    peak_val = -np.inf
    peak_ts: pd.Timestamp | None = None

    for ts, exc in exceeds.items():
        if exc:
            if not in_event:
                in_event = True
                event_start = ts
                peak_val = rolling_sum[ts]
                peak_ts = ts
            else:
                if rolling_sum[ts] > peak_val:
                    peak_val = rolling_sum[ts]
                    peak_ts = ts
        else:
            if in_event:
                # Check if we've had enough dry hours to close the event
                if dry_run[ts] >= DRY_SPELL_HOURS:
                    trigger_times.append(peak_ts)
                    in_event = False
                    peak_val = -np.inf
                    peak_ts = None

    # Close any open event at end of series
    if in_event and peak_ts is not None:
        trigger_times.append(peak_ts)

    return trigger_times


def find_flow_events(hourly_flow: pd.Series, threshold_cfs: float) -> list[pd.Timestamp]:
    """Return peak timestamps for independent streamflow exceedance events.

    Two exceedances are the same event unless separated by >= 24 h below
    the threshold (analogous to the precipitation dry-spell rule).
    """
    exceeds = hourly_flow >= threshold_cfs
    if not exceeds.any():
        return []

    events: list[pd.Timestamp] = []
    in_event = False
    peak_val = -np.inf
    peak_ts: pd.Timestamp | None = None
    dry_count = 0

    for ts, exc in exceeds.items():
        if exc:
            dry_count = 0
            if not in_event:
                in_event = True
            if hourly_flow[ts] > peak_val:
                peak_val = hourly_flow[ts]
                peak_ts = ts
        else:
            if in_event:
                dry_count += 1
                if dry_count >= DRY_SPELL_HOURS:
                    events.append(peak_ts)
                    in_event = False
                    peak_val = -np.inf
                    peak_ts = None
                    dry_count = 0

    if in_event and peak_ts is not None:
        events.append(peak_ts)

    return events


# ---------- Classification ----------

def classify(
    trigger_times: list[pd.Timestamp],
    flow_events: list[pd.Timestamp],
    hourly_flow: pd.Series,
    flow_threshold_cfs: float,
    n_common_hours: int,
) -> dict:
    """Compute TP, FP, FN, TN for one (station, duration, precip_rp, flow_rp) combo."""
    flow_event_set = set(flow_events)
    tp = fp = fn = 0
    matched_flow_events: set[pd.Timestamp] = set()

    for t_trigger in trigger_times:
        window = hourly_flow[t_trigger: t_trigger + pd.Timedelta(hours=RESPONSE_HOURS)]
        if window.empty:
            fp += 1
            continue

        responded = window.max() >= flow_threshold_cfs
        if responded:
            tp += 1
            # Mark whichever flow event falls within this window as matched
            for fe in flow_event_set:
                if t_trigger <= fe <= t_trigger + pd.Timedelta(hours=RESPONSE_HOURS):
                    matched_flow_events.add(fe)
        else:
            fp += 1

    # Unmatched flow events are false negatives
    fn = len(flow_event_set - matched_flow_events)

    # TN: hours with no active trigger and streamflow below threshold
    trigger_hours: set[pd.Timestamp] = set()
    for t in trigger_times:
        for h in range(RESPONSE_HOURS + 1):
            trigger_hours.add(t + pd.Timedelta(hours=h))

    exceedance_hours = set(hourly_flow[hourly_flow >= flow_threshold_cfs].index.tolist())
    neither = n_common_hours - len(trigger_hours | exceedance_hours)
    tn = max(0, neither)

    return {
        "n_trigger_events": len(trigger_times),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------- Per-station analysis ----------

def analyse_station(
    site_no: str,
    mrms_wide: pd.DataFrame,     # index=datetime_utc, columns=precip_in (one source)
    flow_hourly: pd.Series,      # index=datetime_utc
    atlas14_site: pd.DataFrame,  # duration_hr, return_period_yr, depth_in
    flow_stats_row: pd.Series,   # Q10, Q50
    mrms_source: MrmsSource,
) -> list[dict]:
    # Common period
    common_start = max(mrms_wide.index.min(), flow_hourly.index.min())
    common_end   = min(mrms_wide.index.max(), flow_hourly.index.max())

    log.debug(
        "Site %s (%s): MRMS %s → %s | flow %s → %s | common %s → %s",
        site_no, mrms_source,
        mrms_wide.index.min(), mrms_wide.index.max(),
        flow_hourly.index.min(), flow_hourly.index.max(),
        common_start, common_end,
    )

    if common_start >= common_end:
        log.warning(
            "Site %s (%s): no common period — MRMS %s → %s | flow %s → %s",
            site_no, mrms_source,
            mrms_wide.index.min(), mrms_wide.index.max(),
            flow_hourly.index.min(), flow_hourly.index.max(),
        )
        return []

    mrms_c  = mrms_wide.loc[common_start:common_end, "precip_in"]
    flow_c  = flow_hourly.loc[common_start:common_end]

    # Fill gaps with 0 for MRMS (missing hours = no rain reported)
    mrms_c = mrms_c.reindex(
        pd.date_range(common_start, common_end, freq="1h", tz="UTC"), fill_value=0.0
    )
    flow_c = flow_c.reindex(mrms_c.index)  # keep NaN gaps in flow
    n_common = len(mrms_c)

    records = []
    for duration_hr in DURATIONS_HR:
        rolling = mrms_c.rolling(window=duration_hr, min_periods=duration_hr).sum()

        for precip_rp in PRECIP_RPS:
            row_a14 = atlas14_site.loc[
                (atlas14_site["duration_hr"] == duration_hr) &
                (atlas14_site["return_period_yr"] == precip_rp)
            ]
            if row_a14.empty:
                continue
            threshold_in = float(row_a14["depth_in"].iloc[0])

            triggers = find_trigger_events(mrms_c, rolling, threshold_in)

            for flow_rp in FLOW_RPS:
                q_col = f"Q{flow_rp}"
                if q_col not in flow_stats_row or pd.isna(flow_stats_row[q_col]):
                    continue
                flow_threshold = float(flow_stats_row[q_col])

                flow_events = find_flow_events(flow_c.dropna(), flow_threshold)

                metrics = classify(triggers, flow_events, flow_c, flow_threshold, n_common)

                records.append({
                    "site_no": site_no,
                    "mrms_source": mrms_source,
                    "duration_hr": duration_hr,
                    "precip_rp_yr": precip_rp,
                    "flow_rp_yr": flow_rp,
                    "common_start": common_start,
                    "common_end": common_end,
                    "n_common_hours": n_common,
                    **metrics,
                })

    return records


# ---------- Main ----------

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]  # default: QPE_01H_Pass2

    log.info("Loading input data...")
    atlas14   = load_atlas14(bucket, prefix)
    atlas14["site_no"] = atlas14["site_no"].astype(str)

    flow_stats = load_flow_stats(bucket, prefix)

    streamflow = load_streamflow(bucket, prefix)
    streamflow["site_no"] = streamflow["site_no"].astype(str)

    stations = sorted(set(atlas14["site_no"]) & set(flow_stats["site_no"]) & set(streamflow["site_no"]))
    log.info("Stations with all required inputs: %d", len(stations))

    # Drop stations whose streamflow record ends before the MRMS period starts.
    # These would produce 0 combinations regardless and are not re-run in script 01.
    mrms_start = pd.Timestamp("2020-10-14", tz="UTC")
    flow_end_by_site = (
        streamflow.groupby("site_no")["datetime_utc"].max()
    )
    pre_mrms = [s for s in stations if flow_end_by_site.get(s, pd.NaT) < mrms_start]
    if pre_mrms:
        log.info(
            "Skipping %d station(s) with streamflow ending before MRMS start (2020-10-14): %s",
            len(pre_mrms), sorted(pre_mrms),
        )
    stations = [s for s in stations if s not in set(pre_mrms)]
    log.info("Stations after MRMS-era filter: %d", len(stations))

    all_records: list[dict] = []

    for source in ("nearest", "watershed"):
        log.info("Loading MRMS source: %s", source)
        try:
            mrms = load_mrms(bucket, prefix, product_key, source)
        except Exception as e:
            log.error("Could not load MRMS %s: %s", source, e)
            continue
        mrms["site_no"] = mrms["site_no"].astype(str)

        for i, site_no in enumerate(stations, 1):
            mrms_site = (
                mrms[mrms["site_no"] == site_no]
                .set_index("datetime_utc")
                .sort_index()
            )
            flow_site = (
                streamflow[streamflow["site_no"] == site_no]
                .set_index("datetime_utc")["value_cfs"]
                .sort_index()
            )
            atlas14_site = atlas14[atlas14["site_no"] == site_no].copy()
            flow_row = flow_stats[flow_stats["site_no"] == site_no].iloc[0]

            if mrms_site.empty or flow_site.empty or atlas14_site.empty:
                log.warning("[%s][%d/%d] %s: missing data, skipping", source, i, len(stations), site_no)
                continue

            records = analyse_station(
                site_no, mrms_site, flow_site, atlas14_site, flow_row, source
            )
            all_records.extend(records)
            log.info("[%s][%d/%d] %s: %d combinations", source, i, len(stations), site_no, len(records))

    if not all_records:
        log.error("No results produced.")
        return

    out = pd.DataFrame(all_records)
    write_parquet_to_s3(out, bucket, f"{prefix}analysis/trigger_analysis.parquet")
    log.info(
        "Wrote analysis/trigger_analysis.parquet (%d rows, %d stations)",
        len(out), out["site_no"].nunique(),
    )


if __name__ == "__main__":
    main()
