#!/usr/bin/env python3
"""compare_atlas14_vs_areal_ddf.py

Compare the Atlas 14 POINT precipitation-frequency table (script 07) with the
watershed-mean AREAL DDF (script 07a) to confirm they overlap well enough for a
fair point-vs-areal trigger comparison.

Two things matter:
  1. DURATION OVERLAP — 08c/08h interpolate Atlas 14 to each station's Kirpich Tc;
     08g interpolates the areal DDF. If Atlas 14's stored durations don't reach a
     station's Tc but the areal DDF does, the point side CLAMPS while the areal
     side interpolates → the comparison is biased at that station. This flags any
     such station.
  2. VALUE SANITY — where both have the same (duration, return period), the areal
     depth should sit BELOW the point depth (areal reduction). The areal/point
     ratio is an empirical ARF; ratios > 1 are physically suspect.

Run on EC2:
    python scripts/compare_atlas14_vs_areal_ddf.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_cfg(repo_root: Path) -> dict:
    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)


def ddf(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["site_no", "duration_hr", "return_period_yr", "depth_in"])
    df["site_no"] = df["site_no"].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--product", default="QPE_01H_Pass2")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    b = f"s3://{bucket}/{prefix}"

    a14   = ddf(f"{b}atlas14/precipitation_frequency.parquet")
    areal = ddf(f"{b}mrms/{args.product}/areal_precip_frequency.parquet")
    tc    = pd.read_parquet(f"{b}watersheds/basin_characteristics.parquet",
                            columns=["site_no", "tc_hr"])
    tc["site_no"] = tc["site_no"].astype(str)
    tc = tc.dropna(subset=["tc_hr"]).set_index("site_no")["tc_hr"]
    univ = set(pd.read_parquet(f"{b}analysis/event_confusion_matrix_tc.parquet",
                               columns=["site_no"])["site_no"].astype(str))

    # ── Global duration / RP coverage ─────────────────────────────────────────
    da = sorted(a14["duration_hr"].unique())
    dr = sorted(areal["duration_hr"].unique())
    print("=== Global duration coverage (hours) ===")
    print(f"  Atlas 14 : {da}")
    print(f"  Areal DDF: {dr}")
    print(f"  common       : {sorted(set(da) & set(dr))}")
    print(f"  Atlas14-only : {sorted(set(da) - set(dr))}")
    print(f"  Areal-only   : {sorted(set(dr) - set(da))}\n")
    print("=== Return periods (yr) ===")
    print(f"  Atlas 14 : {sorted(a14['return_period_yr'].unique())}")
    print(f"  Areal DDF: {sorted(areal['return_period_yr'].unique())}   (no P1 = AMS-GEV expected)\n")

    # ── Per-station Tc bracketing ─────────────────────────────────────────────
    a14_dur = a14.groupby("site_no")["duration_hr"].agg(["min", "max"])
    are_dur = areal.groupby("site_no")["duration_hr"].agg(["min", "max"])
    rows = []
    for s in sorted(univ):
        if s not in tc.index:
            continue
        tc_h = max(1.0, round(float(tc[s])))
        a_min = a14_dur["min"].get(s, np.nan); a_max = a14_dur["max"].get(s, np.nan)
        r_min = are_dur["min"].get(s, np.nan); r_max = are_dur["max"].get(s, np.nan)
        a_ok = (not np.isnan(a_max)) and (a_min <= tc_h <= a_max)
        r_ok = (not np.isnan(r_max)) and (r_min <= tc_h <= r_max)
        rows.append({"site_no": s, "tc_hr": tc_h,
                     "a14_dur_max": a_max, "areal_dur_max": r_max,
                     "a14_brackets": a_ok, "areal_brackets": r_ok,
                     "a14_clamps_only": (not a_ok) and r_ok})
    B = pd.DataFrame(rows)
    print("=== Per-station Tc bracketing ===")
    print(f"  stations checked          : {len(B)}")
    print(f"  Tc range (h)              : {int(B.tc_hr.min())} .. {int(B.tc_hr.max())}")
    print(f"  both DDFs bracket Tc      : {int((B.a14_brackets & B.areal_brackets).sum())}")
    print(f"  Atlas14 CLAMPS, areal OK  : {int(B.a14_clamps_only.sum())}   <-- comparison bias")
    print(f"  areal clamps (Tc>max)     : {int((~B.areal_brackets).sum())}\n")
    bias = B[B.a14_clamps_only].sort_values("tc_hr", ascending=False)
    if len(bias):
        print(f"Stations where Atlas 14 clamps but the areal DDF interpolates ({len(bias)}):")
        print(bias[["site_no", "tc_hr", "a14_dur_max", "areal_dur_max"]].to_string(index=False))
        print()

    # ── Value sanity: areal / point ratio at shared (duration, RP) ────────────
    merged = a14.merge(areal, on=["site_no", "duration_hr", "return_period_yr"],
                       suffixes=("_point", "_areal"))
    merged = merged[merged["site_no"].isin(univ) & (merged["depth_in_point"] > 0)]
    merged["ratio"] = merged["depth_in_areal"] / merged["depth_in_point"]
    print("=== Areal / point depth ratio (empirical ARF) at shared (duration, RP) ===")
    print(f"  shared rows: {len(merged):,} | overall median ratio: {merged['ratio'].median():.3f}")
    by_dur = merged.groupby("duration_hr")["ratio"].median().round(3)
    print("  median ratio by duration (h):")
    print(by_dur.to_string())
    n_gt1 = int((merged["ratio"] > 1.0).sum())
    print(f"\n  rows with areal > point (ratio>1, physically suspect): {n_gt1} / {len(merged)}")
    if n_gt1:
        worst = merged.sort_values("ratio", ascending=False).head(10)
        print(worst[["site_no", "duration_hr", "return_period_yr",
                     "depth_in_point", "depth_in_areal", "ratio"]].to_string(index=False))

    out = Path(args.repo) / "atlas14_vs_areal_ddf_bracketing.csv"
    B.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
