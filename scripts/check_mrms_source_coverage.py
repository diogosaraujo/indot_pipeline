#!/usr/bin/env python3
"""check_mrms_source_coverage.py

Report exactly which date range / how many station-hours in the merged MRMS
nearest_pixel.parquet came from each download source:

    NOAA MRMS PDS   (s3://noaa-mrms-pds)      2020-10-14 -> present   [script 05]
    ISU MRMS        (mtarchive.geol.iastate)  ~2015-01-01 -> 2020-10-13 [05b]
    NOAA Stage IV   (mesonet.agron.iastate)   2002-01-01 -> ~2015     [05b fallback]

Provenance is NOT stored per row (schema is datetime_utc, site_no, value), so
source is inferred from the timestamp against the extractor's archive boundaries.
The NOAA-PDS boundary is exact. The ISU-MRMS-vs-Stage-IV split in the 2015-2020
overlap is a per-hour fallback in 05b, so pass --probe to re-query the ISU MRMS
archive and attribute those days definitively.

Run on the EC2 box (instance role must read your private output bucket):
    python scripts/check_mrms_source_coverage.py                      # fast, date-based
    python scripts/check_mrms_source_coverage.py --probe              # + resolve Stage IV
    python scripts/check_mrms_source_coverage.py --dataset watershed  # watershed_mean.parquet
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

# Boundaries copied verbatim from 05b_extract_historical_precip_nearest.py
STAGE4_START   = date(2002,  1,  1)
ISU_MRMS_START = date(2015,  1,  1)
MRMS_PDS_START = date(2020, 10, 14)

ISU_MRMS_BASE   = "https://mtarchive.geol.iastate.edu"
ISU_MRMS_FOLDER = "GaugeCorr_QPE_01H"
ISU_MRMS_FSTEM  = "GaugeCorr_QPE_01H_00.00"


def isu_mrms_url(dt) -> str:
    fname = f"{ISU_MRMS_FSTEM}_{dt.strftime('%Y%m%d')}-{dt.strftime('%H')}0000.grib2.gz"
    return (f"{ISU_MRMS_BASE}/{dt.year}/{dt.month:02d}/{dt.day:02d}"
            f"/mrms/ncep/{ISU_MRMS_FOLDER}/{fname}")


def load_cfg(repo_root: Path) -> dict:
    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)


def bucket_span(ts: pd.Series) -> dict:
    """Summarize one source's slice of timestamps."""
    if ts.empty:
        return dict(rows=0, hours=0, first=None, last=None, completeness=None)
    hours = ts.dt.floor("h").nunique()
    first, last = ts.min(), ts.max()
    expected = int((last.floor("h") - first.floor("h")).total_seconds() // 3600) + 1
    return dict(rows=len(ts), hours=hours, first=first, last=last,
                completeness=hours / expected if expected else None)


def report(name: str, s: dict) -> None:
    if s["rows"] == 0:
        print(f"  {name:<26} (none)")
        return
    comp = f"{s['completeness']*100:5.1f}%" if s["completeness"] is not None else "  n/a"
    print(f"  {name:<26} {s['first']:%Y-%m-%d %Hz} -> {s['last']:%Y-%m-%d %Hz}  "
          f"| {s['hours']:>6,} distinct hrs ({comp} of span)  "
          f"| {s['rows']:>10,} station-hours")


def probe_isu_mrms(days: list[date], workers: int = 32) -> set[date]:
    """Return the subset of days for which the ISU MRMS archive has a file
    (checked at 12Z as a day representative). Days NOT returned are Stage IV."""
    have = set()

    def check(d: date) -> tuple[date, bool]:
        url = isu_mrms_url(pd.Timestamp(d) + pd.Timedelta(hours=12))
        try:
            r = requests.head(url, timeout=30, allow_redirects=True)
            if r.status_code == 405:                      # server refuses HEAD
                r = requests.get(url, timeout=30, stream=True)
            return d, r.status_code == 200
        except requests.RequestException:
            return d, False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(check, d) for d in days]
        for i, fut in enumerate(as_completed(futs), 1):
            d, ok = fut.result()
            if ok:
                have.add(d)
            if i % 200 == 0:
                print(f"    probed {i}/{len(days)} days...")
    return have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="QPE_01H_Pass2")
    ap.add_argument("--dataset", choices=["nearest", "watershed"], default="nearest")
    ap.add_argument("--probe", action="store_true",
                    help="re-query ISU MRMS archive to split MRMS vs Stage IV")
    ap.add_argument("--repo", default=".", help="path to repo root (for config.yaml)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    fname = "nearest_pixel.parquet" if args.dataset == "nearest" else "watershed_mean.parquet"
    path = f"s3://{bucket}/{prefix}mrms/{args.product}/{fname}"

    print(f"Reading {path}")
    df = pd.read_parquet(path, columns=["datetime_utc"])   # only the time column
    ts = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_localize(None)
    print(f"Total: {len(ts):,} station-hours | "
          f"{ts.dt.floor('h').nunique():,} distinct hours | "
          f"{ts.min()} -> {ts.max()}\n")

    d = ts.dt.date
    pds     = ts[d >= MRMS_PDS_START]
    overlap = ts[(d >= ISU_MRMS_START) & (d < MRMS_PDS_START)]   # ISU MRMS or Stage IV
    early   = ts[(d >= STAGE4_START)   & (d < ISU_MRMS_START)]   # Stage IV only
    pre     = ts[d < STAGE4_START]

    print("By source (date-based inference):")
    report("NOAA MRMS PDS  [05]",       bucket_span(pds))
    report("ISU MRMS+StageIV [05b]",    bucket_span(overlap))
    report("Stage IV only  [05b]",      bucket_span(early))
    if len(pre):
        report("PRE-2002 (unexpected!)", bucket_span(pre))

    if args.probe and len(overlap):
        days = sorted(set(overlap.dt.date))
        print(f"\nProbing ISU MRMS archive for {len(days)} overlap days "
              f"(12Z representative)...")
        mrms_days = probe_isu_mrms(days)
        od = overlap.dt.date
        isu     = overlap[od.isin(mrms_days)]
        stage4o = overlap[~od.isin(mrms_days)]
        print("\nOverlap window resolved:")
        report("  -> ISU MRMS",         bucket_span(isu))
        report("  -> Stage IV fallback", bucket_span(stage4o))
        print("\nDefinitive Stage IV total (early + overlap-fallback):")
        report("Stage IV (all)", bucket_span(pd.concat([early, stage4o])))


if __name__ == "__main__":
    main()
