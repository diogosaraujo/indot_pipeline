"""09b_flow_threshold_diagnostic.py

For each station produce a multi-page PDF:
  Page 1      : Full streamflow timeseries with Q10 / Q50 threshold lines.
                Exceedance periods are shaded (red = Q10, orange = Q50).
  Pages 2 +   : One close-up per exceedance event (event window ± 24 h),
                showing streamflow, both threshold lines, the peak annotation,
                and whether Q50 was also breached.

Source labels on threshold lines indicate whether the value came from
USGS StreamStats (gage_stats) or the Rao 2005 regional regression.

Outputs (one PDF per station):
    s3://<bucket>/<prefix>figures/flow_diagnostic/<site_no>.pdf

Download locally:
    aws s3 sync s3://<bucket>/<prefix>figures/flow_diagnostic/ ./flow_diagnostic/
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages

from utils import load_config, s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("09b_flow_diagnostic")

DRY_SPELL_HOURS = 24
BUFFER_HOURS    = 24   # padding on each side of an event for close-up plots


# ---------- helpers -----------------------------------------------------------

def _read_parquet_s3(bucket: str, key: str, columns=None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _source_label(source: Optional[str], region: Optional[str]) -> str:
    if source == "gage_stats":
        return "USGS StreamStats"
    if source == "regression" and region:
        r = str(region).replace("Rao2005_R", "")
        return f"Rao 2005 Region {r}"
    return source or "unknown"


# ---------- event detection ---------------------------------------------------

def find_flow_events(
    hourly_flow: pd.Series,
    threshold_cfs: float,
) -> list[dict]:
    """Return [{start, peak, end, peak_val}] for each independent exceedance event.

    Two exceedances belong to the same event unless separated by >= DRY_SPELL_HOURS
    consecutive hours below the threshold (mirrors script 08 logic).
    """
    exceeds = hourly_flow >= threshold_cfs
    if not exceeds.any():
        return []

    events: list[dict] = []
    in_event = False
    peak_val  = -float("inf")
    peak_ts   = None
    event_start = None
    dry_count = 0
    last_exc_ts = None

    for ts, exc in exceeds.items():
        if exc:
            dry_count = 0
            last_exc_ts = ts
            if not in_event:
                in_event    = True
                event_start = ts
                peak_val    = float(hourly_flow[ts])
                peak_ts     = ts
            elif float(hourly_flow[ts]) > peak_val:
                peak_val = float(hourly_flow[ts])
                peak_ts  = ts
        else:
            if in_event:
                dry_count += 1
                if dry_count >= DRY_SPELL_HOURS:
                    events.append({
                        "start":    event_start,
                        "peak":     peak_ts,
                        "end":      last_exc_ts,
                        "peak_val": peak_val,
                    })
                    in_event    = False
                    peak_val    = -float("inf")
                    peak_ts     = None
                    event_start = None
                    dry_count   = 0

    if in_event and peak_ts is not None:
        events.append({
            "start":    event_start,
            "peak":     peak_ts,
            "end":      last_exc_ts,
            "peak_val": peak_val,
        })

    return events


# ---------- plotting ----------------------------------------------------------

def _fmt_axes(ax: plt.Axes, x0: pd.Timestamp, x1: pd.Timestamp) -> None:
    span_days = (x1 - x0).total_seconds() / 86400
    if span_days <= 14:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    elif span_days <= 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int(span_days // 180))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_xlim(x0, x1)
    ax.grid(True, alpha=0.3)


def plot_overview(
    ax: plt.Axes,
    site_no: str,
    site_name: str,
    flow: pd.Series,
    q10: Optional[float],
    q50: Optional[float],
    src_label: str,
    events_q10: list[dict],
    events_q50: list[dict],
) -> None:
    ax.plot(flow.index, flow.values, color="#2c7bb6", linewidth=0.5, label="Streamflow")

    # Shade exceedance periods (Q10 first so Q50 overlays on top)
    for ev in events_q10:
        ax.axvspan(ev["start"], ev["end"], alpha=0.12, color="red",    zorder=0)
    for ev in events_q50:
        ax.axvspan(ev["start"], ev["end"], alpha=0.18, color="darkorange", zorder=0)

    if q10 is not None:
        ax.axhline(
            q10, color="red", linestyle="--", linewidth=1.1,
            label=f"Q10 = {q10:,.0f} cfs  [{src_label}]",
        )
    if q50 is not None:
        ax.axhline(
            q50, color="darkorange", linestyle="--", linewidth=1.1,
            label=f"Q50 = {q50:,.0f} cfs  [{src_label}]",
        )

    n10 = len(events_q10)
    n50 = len(events_q50)
    ax.set_title(
        f"{site_no}  |  {site_name}  |  "
        f"Q10 events: {n10}   Q50 events: {n50}",
        fontsize=9,
    )
    ax.set_ylabel("Streamflow (cfs)")
    ax.legend(fontsize=8, loc="upper right")
    _fmt_axes(ax, flow.index.min(), flow.index.max())


def plot_event_closeup(
    ax: plt.Axes,
    site_no: str,
    site_name: str,
    flow: pd.Series,
    q10: Optional[float],
    q50: Optional[float],
    src_label: str,
    event: dict,
    event_n: int,
) -> None:
    t0 = max(event["start"] - pd.Timedelta(hours=BUFFER_HOURS), flow.index.min())
    t1 = min(event["end"]   + pd.Timedelta(hours=BUFFER_HOURS), flow.index.max())
    sliced = flow.loc[t0:t1]

    ax.plot(sliced.index, sliced.values, color="#2c7bb6", linewidth=1.2)

    # Shade exceedance window
    q10_shade = q10 if q10 is not None else q50
    ax.axvspan(event["start"], event["end"], alpha=0.15, color="red",
               label="Exceedance period", zorder=0)

    if q10 is not None:
        ax.axhline(q10, color="red", linestyle="--", linewidth=1.1,
                   label=f"Q10 = {q10:,.0f} cfs  [{src_label}]")
    if q50 is not None:
        ax.axhline(q50, color="darkorange", linestyle="--", linewidth=1.1,
                   label=f"Q50 = {q50:,.0f} cfs  [{src_label}]")

    # Peak marker
    ax.axvline(event["peak"], color="darkred", linestyle=":", linewidth=1.0,
               label="Peak flow")
    ax.annotate(
        f"Peak: {event['peak_val']:,.0f} cfs",
        xy=(event["peak"], event["peak_val"]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )

    # Label whether Q50 was also exceeded
    q50_tag = ""
    if q50 is not None and event["peak_val"] >= q50:
        q50_tag = "  ⚑ Q50 exceeded"

    ax.set_title(
        f"{site_no}  |  Event {event_n}  |  "
        f"{event['start'].strftime('%Y-%m-%d')} – {event['end'].strftime('%Y-%m-%d')}"
        f"{q50_tag}",
        fontsize=9,
    )
    ax.set_ylabel("Streamflow (cfs)")
    ax.legend(fontsize=8, loc="upper right")
    _fmt_axes(ax, t0, t1)


# ---------- per-station driver ------------------------------------------------

def process_station(
    site_no: str,
    site_name: str,
    flow_hourly: pd.Series,
    row: pd.Series,
    bucket: str,
    prefix: str,
) -> None:
    q10 = float(row["Q10"]) if "Q10" in row and not pd.isna(row["Q10"]) else None
    q50 = float(row["Q50"]) if "Q50" in row and not pd.isna(row["Q50"]) else None
    src = str(row.get("source", "")) if not pd.isna(row.get("source", pd.NA)) else ""
    region = (
        str(row["regression_region"])
        if "regression_region" in row.index and not pd.isna(row.get("regression_region"))
        else None
    )
    src_label = _source_label(src, region)

    clean_flow = flow_hourly.dropna()
    events_q10 = find_flow_events(clean_flow, q10) if q10 is not None else []
    events_q50 = find_flow_events(clean_flow, q50) if q50 is not None else []

    # Build unified event list for close-ups: Q10 events, then any Q50 events
    # whose peak wasn't already captured by a Q10 event.
    q10_peak_set = {ev["peak"] for ev in events_q10}
    extra_q50 = [ev for ev in events_q50 if ev["peak"] not in q10_peak_set]
    all_events = events_q10 + extra_q50

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- page 1: overview ---
        fig, ax = plt.subplots(figsize=(15, 4))
        plot_overview(ax, site_no, site_name, flow_hourly,
                      q10, q50, src_label, events_q10, events_q50)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- pages 2+: one close-up per event ---
        for n, ev in enumerate(all_events, 1):
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_event_closeup(ax, site_no, site_name, flow_hourly,
                               q10, q50, src_label, ev, n)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    key = f"{prefix}figures/flow_diagnostic/{site_no}.pdf"
    s3_client().put_object(
        Bucket=bucket, Key=key,
        Body=buf.getvalue(), ContentType="application/pdf",
    )
    log.info(
        "%s: %d pages  (Q10 events: %d, Q50 events: %d)",
        site_no, 1 + len(all_events), len(events_q10), len(events_q50),
    )


# ---------- main --------------------------------------------------------------

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Loading streamflow...")
    flow_raw = _read_parquet_s3(
        bucket,
        f"{prefix}streamflow/instantaneous/all_gauges_long.parquet",
        columns=["site_no", "datetime", "value_cfs"],
    )
    flow_raw["value_cfs"] = pd.to_numeric(flow_raw["value_cfs"], errors="coerce")
    flow_raw["datetime"]  = pd.to_datetime(flow_raw["datetime"], utc=True)
    flow_raw["site_no"]   = flow_raw["site_no"].astype(str)

    # Resample to hourly max (consistent with script 08)
    log.info("Resampling to hourly max...")
    flow_hourly = (
        flow_raw.set_index("datetime")
        .groupby("site_no")["value_cfs"]
        .resample("1h")
        .max()
        .reset_index()
        .rename(columns={"datetime": "datetime_utc"})
    )
    flow_hourly["site_no"] = flow_hourly["site_no"].astype(str)

    log.info("Loading flow stats and station inventory...")
    stats = _read_parquet_s3(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    stats["site_no"] = stats["site_no"].astype(str)

    inv = _read_parquet_s3(
        bucket,
        f"{prefix}stations/indiana_streamflow_sites.parquet",
        columns=["site_no", "station_nm"],
    )
    inv["site_no"] = inv["site_no"].astype(str)
    name_map = inv.set_index("site_no")["station_nm"].to_dict()

    stations = sorted(set(stats["site_no"]) & set(flow_hourly["site_no"]))
    log.info("Stations to process: %d", len(stations))

    for i, site_no in enumerate(stations, 1):
        row = stats[stats["site_no"] == site_no].iloc[0]
        if pd.isna(row.get("Q10")) and pd.isna(row.get("Q50")):
            log.warning("[%d/%d] %s: no Q10 or Q50 — skipping", i, len(stations), site_no)
            continue

        flow_site = (
            flow_hourly[flow_hourly["site_no"] == site_no]
            .set_index("datetime_utc")["value_cfs"]
            .sort_index()
        )
        site_name = name_map.get(site_no, "")

        log.info("[%d/%d] %s  %s", i, len(stations), site_no, site_name)
        try:
            process_station(site_no, site_name, flow_site, row, bucket, prefix)
        except Exception as e:
            log.error("[%d/%d] %s: failed — %s", i, len(stations), site_no, e)


if __name__ == "__main__":
    main()
