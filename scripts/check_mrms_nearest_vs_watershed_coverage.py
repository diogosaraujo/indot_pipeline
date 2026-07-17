#!/usr/bin/env python3
"""check_mrms_nearest_vs_watershed_coverage.py

Confirm the MRMS WATERSHED-MEAN record covers the SAME period as the
NEAREST-PIXEL record, per station — a prerequisite for the areal (watershed-mean)
frequency analysis.

Why this matters: the pre-2020 Stage IV / ISU-MRMS backfill (2002 -> 2020-10-13)
was applied to the two products by SEPARATE scripts:
    nearest_pixel.parquet   <- 05  (+ 05b historical backfill)
    watershed_mean.parquet  <- 06  (+ 06b historical backfill)
So a station can easily have 2002->present nearest-pixel but only 2020->present
watershed-mean if 06b wasn't run over the same universe. This flags any such gap.

Coverage = presence of rows (hours) per station; the value column is irrelevant
here, so only site_no + datetime_utc are read (fast, low memory).

Run on EC2 (instance role must read the private output bucket):
    python scripts/check_mrms_nearest_vs_watershed_coverage.py
    python scripts/check_mrms_nearest_vs_watershed_coverage.py --tol-days 31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_cfg(repo_root: Path) -> dict:
    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)


def coverage(path: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Per-station (first, last, distinct-hours) plus overall span."""
    df = pd.read_parquet(path, columns=["site_no", "datetime_utc"])
    site = df["site_no"].astype(str)
    hour = pd.to_datetime(df["datetime_utc"], utc=True).dt.floor("h")
    d = pd.DataFrame({"site_no": site.values, "hour": hour.values})
    cov = d.groupby("site_no")["hour"].agg(first="min", last="max", hours="nunique")
    return cov, d["hour"].min(), d["hour"].max()


def fmt(ts) -> str:
    return "—" if pd.isna(ts) else f"{ts:%Y-%m-%d %Hz}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--product", default="QPE_01H_Pass2")
    ap.add_argument("--tol-days", type=int, default=31,
                    help="stations whose spans differ by more than this are flagged")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    base = f"s3://{bucket}/{prefix}mrms/{args.product}"
    near_path = f"{base}/nearest_pixel.parquet"
    ws_path   = f"{base}/watershed_mean.parquet"

    print(f"nearest : {near_path}")
    print(f"wshed   : {ws_path}\n")
    cov_n, n_lo, n_hi = coverage(near_path)
    cov_w, w_lo, w_hi = coverage(ws_path)

    print("=== Overall coverage ===")
    print(f"  nearest_pixel  : {fmt(n_lo)} -> {fmt(n_hi)} | {len(cov_n):>4} stations")
    print(f"  watershed_mean : {fmt(w_lo)} -> {fmt(w_hi)} | {len(cov_w):>4} stations\n")

    # ── Per-station join ─────────────────────────────────────────────────────
    m = cov_n.join(cov_w, lsuffix="_near", rsuffix="_ws", how="outer")
    m["start_diff_days"] = (m["first_ws"] - m["first_near"]).dt.total_seconds() / 86400
    m["end_diff_days"]   = (m["last_ws"]  - m["last_near"]).dt.total_seconds() / 86400

    in_both   = m[["first_near", "first_ws"]].notna().all(axis=1)
    only_near = m["first_near"].notna() & m["first_ws"].isna()
    only_ws   = m["first_ws"].notna()  & m["first_near"].isna()

    tol = args.tol_days
    ws_starts_late = in_both & (m["start_diff_days"] > tol)     # <-- the backfill-gap case
    ws_ends_early  = in_both & (m["end_diff_days"] < -tol)
    matched        = in_both & (m["start_diff_days"].abs() <= tol) & (m["end_diff_days"].abs() <= tol)

    print("=== Per-station comparison (joined on site_no) ===")
    print(f"  stations in BOTH products         : {int(in_both.sum())}")
    print(f"  only in nearest_pixel             : {int(only_near.sum())}   (watershed-mean missing entirely)")
    print(f"  only in watershed_mean            : {int(only_ws.sum())}")
    print(f"  matching span (<= {tol}d both ends): {int(matched.sum())}")
    print(f"  watershed starts LATER by > {tol}d : {int(ws_starts_late.sum())}   <-- backfill gap")
    print(f"  watershed ends EARLIER by > {tol}d : {int(ws_ends_early.sum())}\n")

    problem = m[ws_starts_late | ws_ends_early | only_near | only_ws].copy()
    if len(problem):
        problem = problem.sort_values("start_diff_days", ascending=False, na_position="first")
        show = problem[["first_near", "last_near", "first_ws", "last_ws",
                        "start_diff_days", "end_diff_days"]].copy()
        for c in ("first_near", "last_near", "first_ws", "last_ws"):
            show[c] = show[c].apply(fmt)
        for c in ("start_diff_days", "end_diff_days"):
            show[c] = show[c].round(0)
        print(f"Stations with a coverage mismatch ({len(problem)}):")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(show.to_string())
        out = Path(args.repo) / "mrms_nearest_vs_watershed_coverage.csv"
        show.to_csv(out)
        print(f"\nWrote {out}")
    else:
        print("✓ Every station's watershed-mean span matches its nearest-pixel span "
              f"(within {tol} days). The areal record is aligned with the point record.")


if __name__ == "__main__":
    main()
