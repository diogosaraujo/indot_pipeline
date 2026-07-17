#!/usr/bin/env python3
"""check_106_in_mrms_products.py

Confirm the FINAL retained station universe (the 106 gauges kept by 08c, and
reused by 08f) exists in BOTH MRMS products — nearest_pixel AND watershed_mean —
so the areal (watershed-mean) frequency analysis can run on exactly the same
stations as the point (nearest-pixel) analysis.

The 106 universe = distinct site_no in analysis/event_confusion_matrix_tc.parquet
(08c output; 08f labels it "→ the 106 gauges").

Run on EC2:
    python scripts/check_106_in_mrms_products.py
    python scripts/check_106_in_mrms_products.py --universe-key analysis/event_confusion_matrix_nwm.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_cfg(repo_root: Path) -> dict:
    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)


def sites(path: str) -> set[str]:
    return set(pd.read_parquet(path, columns=["site_no"])["site_no"].astype(str).unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--product", default="QPE_01H_Pass2")
    ap.add_argument("--universe-key", default="analysis/event_confusion_matrix_tc.parquet",
                    help="parquet whose distinct site_no defines the retained universe")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    base = f"s3://{bucket}/{prefix}mrms/{args.product}"

    universe = sites(f"s3://{bucket}/{prefix}{args.universe_key}")
    near     = sites(f"{base}/nearest_pixel.parquet")
    wshed    = sites(f"{base}/watershed_mean.parquet")

    print(f"Retained universe ({args.universe_key}): {len(universe)} stations\n")
    print(f"  in nearest_pixel  : {len(universe & near):>3} / {len(universe)}")
    print(f"  in watershed_mean : {len(universe & wshed):>3} / {len(universe)}")
    print(f"  in BOTH           : {len(universe & near & wshed):>3} / {len(universe)}")

    miss_near = sorted(universe - near)
    miss_ws   = sorted(universe - wshed)
    if miss_near:
        print(f"\nMissing from nearest_pixel ({len(miss_near)}): {miss_near}")
    if miss_ws:
        print(f"\nMissing from watershed_mean ({len(miss_ws)}): {miss_ws}")
    if not miss_ws and not miss_near:
        print("\n✓ All retained stations are in both products — the areal analysis "
              "can run on the full retained universe.")


if __name__ == "__main__":
    main()
