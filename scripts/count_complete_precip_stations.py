"""count_complete_precip_stations.py

How many ISD / GHCNh precip stations actually have a (near-)complete HOURLY record
over the MRMS era (2002-2026)?  "Coverage" = fraction of hours in the window that
carry a precip value (non-NaN after flooring to the hour) — i.e. real data density,
not just endpoint-bracketing.

Usage:
    python scripts/count_complete_precip_stations.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
START = pd.Timestamp("2002-01-01", tz="UTC")
END   = pd.Timestamp("2026-06-30", tz="UTC")
SOURCES = ["isd", "ghcnh"]
THRESHOLDS = [99, 95, 90, 80, 50]

TOTAL_HOURS = int((END - START).total_seconds() // 3600) + 1


def _read(key: str, columns) -> pd.DataFrame:
    o = boto3.client("s3").get_object(Bucket=BUCKET, Key=PREFIX + key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def main() -> None:
    print(f"Window: {START.date()} → {END.date()}  ({TOTAL_HOURS:,} hours)\n")

    frames = []
    for src in SOURCES:
        try:
            df = _read(f"precip/noaa/{src}_hourly.parquet",
                       ["station_id", "datetime_utc", "precip_in"])
        except Exception as e:
            print(f"  {src}: {e}")
            continue
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        df = df[(df["datetime_utc"] >= START) & (df["datetime_utc"] <= END)
                & df["precip_in"].notna()]                       # hours with a value
        df["hour"] = df["datetime_utc"].dt.floor("h")
        cov = df.groupby("station_id").agg(
            covered_hours=("hour", "nunique"),
            span_start=("hour", "min"),
            span_end=("hour", "max"),
        ).reset_index()
        cov["source"] = src
        frames.append(cov)
        print(f"  loaded {src}: {cov['station_id'].nunique()} stations")

    if not frames:
        print("No precip data."); return

    allcov = pd.concat(frames, ignore_index=True)
    allcov["span_hours"] = (((allcov["span_end"] - allcov["span_start"]).dt.total_seconds() // 3600) + 1).astype(int)
    allcov["coverage_full"]     = (allcov["covered_hours"] / TOTAL_HOURS * 100).round(1)
    allcov["coverage_internal"] = (allcov["covered_hours"] / allcov["span_hours"] * 100).round(1)

    print(f"\nTotal precip stations with any 2002-2026 data: {len(allcov)}")

    print("\n[A] FULL-WINDOW coverage = covered hours / full 2002-2026 window")
    print("    (penalises stations that start late OR end early)")
    for t in THRESHOLDS:
        print(f"    >= {t:3d}% : {int((allcov['coverage_full'] >= t).sum()):4d}")

    print("\n[B] INTERNAL coverage = covered hours / station's own start→end span")
    print("    (how gap-free a station is WITHIN its record, ignoring span length)")
    for t in THRESHOLDS:
        print(f"    >= {t:3d}% : {int((allcov['coverage_internal'] >= t).sum()):4d}")

    print("\n[C] Genuinely complete over 2002-2026 (spans the window AND dense):")
    spans = ((allcov["span_start"] <= pd.Timestamp("2003-01-01", tz="UTC"))
             & (allcov["span_end"] >= pd.Timestamp("2025-01-01", tz="UTC")))
    print(f"    starts <=2003 AND ends >=2025          : {int(spans.sum())}")
    print(f"    ...AND >=90% internal coverage          : {int((spans & (allcov['coverage_internal'] >= 90)).sum())}")
    print(f"    ...AND >=80% internal coverage          : {int((spans & (allcov['coverage_internal'] >= 80)).sum())}")

    print("\nBy source (full-window coverage):")
    print(allcov.groupby("source")["coverage_full"].describe()[["count", "mean", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
