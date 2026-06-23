"""diagnose_regression.py

Sanity-check the regression-derived flood thresholds by ranking every station
by number of Q10 exceedance events and plotting the worst offenders.

For each of the TOP_N stations the script produces a timeseries PNG showing:
  - Observed instantaneous discharge (cfs)
  - Q10 / Q50 / Q100 horizontal lines (from flow_stats, no QC filter)
  - Median observed flow (dashed green)

A summary CSV ranking all stations is also saved.

All outputs go to s3://indot-bridge-pipeline/v1/analysis/test_flow_regression/.

Usage:
    python scripts/diagnose_regression.py
"""
from __future__ import annotations

import io

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

# ── Configuration ──────────────────────────────────────────────────────────────
BUCKET      = "indot-bridge-pipeline"
PREFIX      = "v1/"
S3_OUT      = "v1/analysis/test_flow_regression/"
START_DATE  = "2002-01-01"
END_DATE    = "2022-12-31"
TOP_N       = 10      # number of worst stations to plot


# ── S3 helpers ─────────────────────────────────────────────────────────────────
_s3_fs: pafs.S3FileSystem | None = None


def _fs() -> pafs.S3FileSystem:
    global _s3_fs
    if _s3_fs is None:
        _s3_fs = pafs.S3FileSystem()
    return _s3_fs


def read_s3(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(
        f"{BUCKET}/{PREFIX}{key}", filesystem=_fs(), columns=columns
    ).to_pandas()


def upload_png(fig: plt.Figure, filename: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    key = S3_OUT + filename
    boto3.client("s3").put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buf,
        ContentType="image/png",
    )
    print(f"  → s3://{BUCKET}/{key}")


def upload_csv(df: pd.DataFrame, filename: str) -> None:
    buf = io.BytesIO(df.to_csv().encode())
    key = S3_OUT + filename
    boto3.client("s3").put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buf,
        ContentType="text/csv",
    )
    print(f"  → s3://{BUCKET}/{key}")


# ── Event counting (no QC filter — intentionally exposing bad thresholds) ──────
def count_events(sf: pd.DataFrame, flow_stats: pd.DataFrame, q_col: str) -> pd.Series:
    """Return Series[site_no → n_events] exceeding ``q_col``, unfiltered.

    Events are declustered by a 24-hour gap: consecutive exceedances within
    24 h count as a single event.
    """
    thresholds = (
        flow_stats.set_index("site_no")[q_col]
        .dropna()
        .rename("threshold")
    )
    merged = sf.merge(thresholds, on="site_no", how="inner")
    exceed = (
        merged[merged["value_cfs"] >= merged["threshold"]]
        .sort_values(["site_no", "datetime"])
        .copy()
    )
    exceed["prev_dt"]   = exceed.groupby("site_no")["datetime"].shift(1)
    exceed["gap_hr"]    = (
        (exceed["datetime"] - exceed["prev_dt"]).dt.total_seconds() / 3600
    )
    exceed["new_event"] = exceed["prev_dt"].isna() | (exceed["gap_hr"] > 24)
    return exceed.groupby("site_no")["new_event"].sum()


# ── Per-station plot ───────────────────────────────────────────────────────────
def plot_station(
    site_no: str,
    n_events: dict[str, int],
    site_ts: pd.DataFrame,
    row: pd.Series,
) -> plt.Figure:
    median_q = float(site_ts.loc[site_ts["value_cfs"] >= 0, "value_cfs"].median())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        site_ts["datetime"], site_ts["value_cfs"],
        lw=0.5, color="steelblue", label="Observed flow",
    )
    ax.axhline(
        row["Q10"], color="gold", lw=1.5, ls="--",
        label=f"Q10  = {row['Q10']:.0f} cfs",
    )
    ax.axhline(
        row["Q50"], color="orange", lw=1.5, ls="--",
        label=f"Q50  = {row['Q50']:.0f} cfs",
    )
    ax.axhline(
        row["Q100"], color="red", lw=1.5, ls="--",
        label=f"Q100 = {row['Q100']:.0f} cfs",
    )
    ax.axhline(
        median_q, color="limegreen", lw=1.0, ls=":",
        label=f"Median = {median_q:.0f} cfs",
    )

    ax.set_title(
        f"Station {site_no}  |  source={row['source']}  |  "
        f"{n_events['Q10']} Q10 / {n_events['Q50']} Q50 / {n_events['Q100']} Q100 "
        f"events (2002–2022)\n"
        f"Q10={row['Q10']:.0f}  Q50={row['Q50']:.0f}  "
        f"Q100={row['Q100']:.0f}  Median={median_q:.0f}  cfs"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading flow statistics...")
    flow_stats = read_s3(
        "flow_stats/per_gauge_flow_stats.parquet",
        columns=["site_no", "source", "Q10", "Q25", "Q50", "Q100"],
    )
    flow_stats["site_no"] = flow_stats["site_no"].astype(str)

    print("Loading streamflow time series (may take a minute)...")
    sf = read_s3(
        "streamflow/instantaneous/all_gauges_long.parquet",
        columns=["site_no", "datetime", "value_cfs"],
    )
    sf["site_no"]   = sf["site_no"].astype(str)
    sf["datetime"]  = pd.to_datetime(sf["datetime"], utc=True)
    sf["value_cfs"] = pd.to_numeric(sf["value_cfs"], errors="coerce")
    sf = sf[
        (sf["datetime"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (sf["datetime"] <= pd.Timestamp(END_DATE, tz="UTC"))
    ]
    print(f"  {len(sf):,} IV records across {sf['site_no'].nunique()} stations")

    print("\nCounting Q10 / Q50 / Q100 events per station (no QC filter)...")
    counts = {
        rp: count_events(sf, flow_stats, rp) for rp in ("Q10", "Q50", "Q100")
    }

    # Build summary table — ranked by Q10 events (descending)
    summary = (
        counts["Q10"].rename("n_q10_events")
        .reset_index()
        .merge(flow_stats, on="site_no", how="left")
    )
    summary["n_q50_events"]  = summary["site_no"].map(counts["Q50"]).fillna(0).astype(int)
    summary["n_q100_events"] = summary["site_no"].map(counts["Q100"]).fillna(0).astype(int)
    summary["median_cfs"] = summary["site_no"].map(
        sf[sf["value_cfs"] >= 0].groupby("site_no")["value_cfs"].median()
    )
    summary["q10_vs_median"] = (summary["Q10"] / summary["median_cfs"]).round(2)
    summary = summary.sort_values("n_q10_events", ascending=False).reset_index(drop=True)

    cols = ["site_no", "n_q10_events", "n_q50_events", "n_q100_events",
            "source", "Q10", "Q50", "Q100", "median_cfs", "q10_vs_median"]

    print("\nTop 10 stations by Q10 events:")
    print(summary.head(10)[cols].to_string(index=False))

    print(f"\nSaving summary CSV...")
    upload_csv(summary[cols], "station_q10_event_ranking.csv")

    print(f"\nPlotting top {TOP_N} stations...")
    for rank, row_s in enumerate(summary.head(TOP_N).itertuples(), start=1):
        site_no   = row_s.site_no
        n_events  = {
            "Q10":  row_s.n_q10_events,
            "Q50":  row_s.n_q50_events,
            "Q100": row_s.n_q100_events,
        }
        fs_row    = flow_stats[flow_stats["site_no"] == site_no].iloc[0]
        site_ts   = sf[sf["site_no"] == site_no].sort_values("datetime")

        print(f"  [{rank}/{TOP_N}] {site_no}  "
              f"(Q10={n_events['Q10']} / Q50={n_events['Q50']} / Q100={n_events['Q100']} "
              f"events, source={fs_row['source']})")
        fig = plot_station(site_no, n_events, site_ts, fs_row)
        upload_png(fig, f"rank{rank:02d}_{site_no}.png")
        plt.close(fig)

    # ── Diagnostic 1: non-monotonic frequency curves ──────────────────────────
    # A valid LP3/flood-frequency curve must satisfy Q10 ≤ Q25 ≤ Q50 ≤ Q100.
    # Any violation means the quantiles are physically impossible.
    fs = flow_stats
    mono_cols = ["Q10", "Q25", "Q50", "Q100"]
    have_all  = fs[mono_cols].notna().all(axis=1)
    violation = (
        (fs["Q50"]  < fs["Q10"])
        | (fs["Q100"] < fs["Q50"])
        | (fs["Q100"] < fs["Q10"])
        | (fs["Q25"]  < fs["Q10"])
        | (fs["Q50"]  < fs["Q25"])
        | (fs["Q100"] < fs["Q25"])
    )
    non_mono = fs[have_all & violation].copy()

    print(f"\n{'='*70}")
    print(f"Non-monotonic frequency curves (Q10≤Q25≤Q50≤Q100 violated): "
          f"{len(non_mono)} stations")
    print(f"{'='*70}")
    if len(non_mono):
        print(non_mono[["site_no", "source"] + mono_cols].to_string(index=False))
        upload_csv(non_mono[["site_no", "source"] + mono_cols],
                   "non_monotonic_frequency_curves.csv")
    else:
        print("  None — all stations with a full Q set are monotonic.")

    # ── Diagnostic 2: gage_stats stations with gaps (possible stale fills) ─────
    # The new 04b skips every source=='gage_stats' station, so any gap that was
    # filled by the OLD regression was NOT overwritten with LP3.  Flag them.
    gage = fs[fs["source"] == "gage_stats"].copy()
    gage_gaps = gage[gage[mono_cols].isna().any(axis=1)]

    print(f"\n{'='*70}")
    print(f"gage_stats stations missing one or more of {mono_cols}: "
          f"{len(gage_gaps)} stations")
    print(f"(these were NOT reprocessed by 04b — gaps retain old values or null)")
    print(f"{'='*70}")
    if len(gage_gaps):
        print(gage_gaps[["site_no", "source"] + mono_cols].to_string(index=False))
        upload_csv(gage_gaps[["site_no", "source"] + mono_cols],
                   "gage_stats_with_gaps.csv")
    else:
        print("  None — all gage_stats stations have a complete Q set.")

    print("\nDone.")


if __name__ == "__main__":
    main()
