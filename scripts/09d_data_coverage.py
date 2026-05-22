"""09d_data_coverage.py

Reads actual date ranges from every temporal dataset in S3 and draws a
Gantt-style coverage chart.  All ranges come from the data itself, not
from documentation.

For the QPE nearest_pixel and watershed_mean parquets the record is split
into three source eras (Stage IV / GaugeCorr / MultiSensor Pass2) so the
methodological transitions are visible.

Writes:
    s3://<bucket>/<prefix>analysis/figures/data_coverage.png
"""
from __future__ import annotations

import io
import logging
from datetime import timezone

import boto3
import botocore.exceptions
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, write_bytes_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("09d_coverage")

OUT_KEY = "analysis/figures/data_coverage.png"

# Date boundaries that separate the three QPE source eras
# (must match the constants used in scripts 05b / 06b)
_STAGE4_START = pd.Timestamp("2002-01-01", tz="UTC")
_ISU_CUT      = pd.Timestamp("2015-01-01", tz="UTC")   # Stage IV → GaugeCorr QPE
_PDS_CUT      = pd.Timestamp("2020-10-14", tz="UTC")   # GaugeCorr → MultiSensor Pass2


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def date_range(bucket: str, key: str, col: str = "datetime_utc") -> tuple | None:
    """Return (min_date, max_date) by reading only the datetime column."""
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        df = pq.read_table(io.BytesIO(obj["Body"].read()),
                           columns=[col]).to_pandas()
        df[col] = pd.to_datetime(df[col], utc=True)
        return df[col].min(), df[col].max()
    except botocore.exceptions.ClientError:
        log.warning("Not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def date_range_era(
    bucket: str,
    key: str,
    col: str,
    era_start: pd.Timestamp | None,
    era_end: pd.Timestamp | None,
) -> tuple | None:
    """Return (min_date, max_date) for rows within [era_start, era_end).

    Uses PyArrow predicate pushdown so only matching row groups are read —
    efficient even on a 40M+ row parquet.
    """
    filters: list = []
    if era_start is not None:
        filters.append((col, ">=", era_start))
    if era_end is not None:
        filters.append((col, "<", era_end))
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        t = pq.read_table(
            io.BytesIO(obj["Body"].read()),
            columns=[col],
            filters=filters or None,
        )
        if t.num_rows == 0:
            return None
        arr = pd.to_datetime(t[col].to_pandas(), utc=True)
        r = arr.min(), arr.max()
        log.info("  %-55s  %s → %s  [era %s → %s]",
                 key,
                 r[0].date(), r[1].date(),
                 era_start.date() if era_start else "start",
                 era_end.date()   if era_end   else "end")
        return r
    except botocore.exceptions.ClientError:
        log.warning("Not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read era from %s: %s", key, e)
        return None


def date_range_streamflow(bucket: str, key: str) -> tuple | None:
    for col in ["datetime_utc", "datetime"]:
        r = date_range(bucket, key, col=col)
        if r is not None:
            return r
    return None


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(rows: list[dict]) -> plt.Figure:
    """
    rows: list of dicts with keys:
        label   str   — y-axis label
        start   pd.Timestamp
        end     pd.Timestamp
        color   str
        group   str   — used to draw group separators
    """
    fig, ax = plt.subplots(figsize=(14, max(6, len(rows) * 0.55 + 2)))

    groups_seen: list[str] = []
    yticks, ylabels = [], []

    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        yticks.append(y)
        ylabels.append(row["label"])

        if row["start"] is None or row["end"] is None:
            ax.text(pd.Timestamp("2000-01-01"), y, "  data not found",
                    va="center", fontsize=8, color="#aaa", style="italic")
            continue

        ax.barh(y, row["end"] - row["start"],
                left=row["start"], height=0.55,
                color=row["color"], edgecolor="white", linewidth=0.4,
                alpha=0.88, zorder=2)

        span_years = (row["end"] - row["start"]).days / 365.25
        label_txt = (f'{row["start"].strftime("%Y-%m")} → '
                     f'{row["end"].strftime("%Y-%m")} '
                     f'({span_years:.1f} yr)')
        ax.text(row["end"] + pd.Timedelta(days=40), y, label_txt,
                va="center", fontsize=7.5, color="#333")

        # Group separator line
        if row["group"] not in groups_seen:
            if groups_seen:
                ax.axhline(y + 0.75, color="#ccc", linewidth=0.8, zorder=1)
            groups_seen.append(row["group"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", which="major", color="#ddd", linewidth=0.8, zorder=0)
    ax.grid(axis="x", which="minor", color="#eee", linewidth=0.4, zorder=0)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_title("INDOT Pipeline — Data Coverage by Dataset\n"
                 "(date ranges read from S3 parquets)", fontsize=12, fontweight="bold")

    fig.tight_layout(pad=1.2)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    log.info("Reading date ranges from S3...")

    def get(key, col="datetime_utc"):
        r = date_range(bucket, f"{prefix}{key}", col=col)
        if r:
            log.info("  %-55s  %s → %s", key, r[0].date(), r[1].date())
        return r

    def get_era(key, era_start, era_end):
        return date_range_era(bucket, f"{prefix}{key}", "datetime_utc",
                              era_start, era_end)

    # ── USGS streamflow ───────────────────────────────────────────────────────
    sf = date_range_streamflow(bucket, f"{prefix}streamflow/instantaneous/all_gauges_long.parquet")
    if sf:
        log.info("  %-55s  %s → %s",
                 "streamflow/instantaneous/all_gauges_long.parquet",
                 sf[0].date(), sf[1].date())

    # ── QPE nearest pixel — three source eras ────────────────────────────────
    pixel_key = "mrms/QPE_01H_Pass2/nearest_pixel.parquet"
    log.info("Reading QPE nearest pixel eras...")
    pix_stage4  = get_era(pixel_key, _STAGE4_START, _ISU_CUT)
    pix_gaugecc = get_era(pixel_key, _ISU_CUT,      _PDS_CUT)
    pix_multi   = get_era(pixel_key, _PDS_CUT,      None)

    # ── QPE watershed mean — three source eras ───────────────────────────────
    ws_key = "mrms/QPE_01H_Pass2/watershed_mean.parquet"
    log.info("Reading QPE watershed mean eras...")
    ws_stage4   = get_era(ws_key, _STAGE4_START, _ISU_CUT)
    ws_gaugecc  = get_era(ws_key, _ISU_CUT,      _PDS_CUT)
    ws_multi    = get_era(ws_key, _PDS_CUT,      None)

    # ── NWM ───────────────────────────────────────────────────────────────────
    nwm_retro = get("nwm/retrospective.parquet")
    nwm_assim = get("nwm/analysis_assim.parquet")
    nwm_loop  = get("nwm/open_loop.parquet")

    # ── Supplementary precipitation ───────────────────────────────────────────
    isd    = get("precip/noaa/isd_hourly.parquet")
    ghcnh  = get("precip/noaa/ghcnh_hourly.parquet")
    usgs_p = get("precip/usgs/precip_iv.parquet")

    # ── Analysis output ───────────────────────────────────────────────────────
    ta_start = ta_end = None
    try:
        obj = _s3().get_object(Bucket=bucket,
                               Key=f"{prefix}analysis/trigger_analysis.parquet")
        ta_df = pq.read_table(io.BytesIO(obj["Body"].read()),
                              columns=["common_start", "common_end"]).to_pandas()
        ta_df["common_start"] = pd.to_datetime(ta_df["common_start"], utc=True, errors="coerce")
        ta_df["common_end"]   = pd.to_datetime(ta_df["common_end"],   utc=True, errors="coerce")
        ta_start = ta_df["common_start"].min()
        ta_end   = ta_df["common_end"].max()
        log.info("  %-55s  %s → %s",
                 "analysis/trigger_analysis.parquet (common period)",
                 ta_start.date(), ta_end.date())
    except Exception as e:
        log.warning("trigger_analysis: %s", e)

    # ── Build rows for chart ───────────────────────────────────────────────────
    C_SF      = "#2E86C1"
    C_STAGE4  = "#F0A500"   # amber  — coarsest / oldest source
    C_GAUGEC  = "#E67E22"   # orange — ISU GaugeCorr
    C_MULTI   = "#BA4A00"   # dark orange — MultiSensor Pass2
    C_NWM     = "#27AE60"
    C_PRECIP  = "#8E44AD"
    C_ANAL    = "#7F8C8D"

    def row(label, rng, color, group):
        s, e = (rng[0], rng[1]) if rng else (None, None)
        return dict(label=label, start=s, end=e, color=color, group=group)

    rows = [
        row("USGS Streamflow\n(all_gauges_long)",              sf,          C_SF,     "Streamflow"),

        row("QPE Nearest · Stage IV 4km\n(mesonet IEM)",       pix_stage4,  C_STAGE4, "QPE Nearest"),
        row("QPE Nearest · GaugeCorr QPE\n(ISU MRMS archive)", pix_gaugecc, C_GAUGEC, "QPE Nearest"),
        row("QPE Nearest · MultiSensor Pass2\n(noaa-mrms-pds)",pix_multi,   C_MULTI,  "QPE Nearest"),

        row("QPE Watershed · Stage IV 4km\n(mesonet IEM)",     ws_stage4,   C_STAGE4, "QPE Watershed"),
        row("QPE Watershed · GaugeCorr QPE\n(ISU MRMS archive)",ws_gaugecc, C_GAUGEC, "QPE Watershed"),
        row("QPE Watershed · MultiSensor Pass2\n(noaa-mrms-pds)",ws_multi,  C_MULTI,  "QPE Watershed"),

        row("NWM Retrospective v3.0",                          nwm_retro,   C_NWM,    "NWM"),
        row("NWM Analysis & Assimilation",                     nwm_assim,   C_NWM,    "NWM"),
        row("NWM Open Loop (no DA)",                           nwm_loop,    C_NWM,    "NWM"),

        row("NOAA ISD/LCD Hourly Precip",                      isd,         C_PRECIP, "Precip"),
        row("NOAA GHCNh Hourly Precip",                        ghcnh,       C_PRECIP, "Precip"),
        row("USGS IV Precipitation (param 00045)",             usgs_p,      C_PRECIP, "Precip"),

        row("Trigger Analysis\n(common MRMS × Streamflow period)",
            (ta_start, ta_end) if ta_start else None,          C_ANAL,      "Analysis"),
    ]

    log.info("Building coverage figure...")
    fig = build_figure(rows)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    key = f"{prefix}{OUT_KEY}"
    write_bytes_to_s3(buf.getvalue(), bucket, key)
    log.info("Saved s3://%s/%s", bucket, key)


if __name__ == "__main__":
    main()
