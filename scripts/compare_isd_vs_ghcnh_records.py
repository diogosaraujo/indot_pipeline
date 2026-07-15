#!/usr/bin/env python3
"""compare_isd_vs_ghcnh_records.py

Record-level comparison of the 165 ISD<->GHCNh station pairs that matched on
WBAN (see compare_isd_vs_ghcnh_stations.py). Station identity already agrees;
this asks whether the DATA agrees:

  * period of record (first/last valid hour) on each side
  * count of valid precip-hours on each side
  * hour-by-hour overlap: shared hours, ISD-only hours, GHCNh-only hours
      -> "ISD-only hours" is the key number: hours ISD carries that GHCNh is
         missing at the SAME station. If ~0 everywhere, GHCNh fully stands in.
  * value agreement on shared hours: Pearson r, mean-abs-diff (inches),
    and wet-hour detection agreement (both >= 0.01 in)

Precip is floored to the hour and summed within each hour before comparison
(both feeds are ~hourly; sub-hourly LCD special reports get collapsed). Trace
is 0.001 in and NaN is missing, per script 12's schema.

Run on EC2 (instance role must read the private output bucket):
    python scripts/compare_isd_vs_ghcnh_records.py
    python scripts/compare_isd_vs_ghcnh_records.py --wet-threshold 0.01
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


def isd_wban(sid: str):
    tail = str(sid).strip()[-5:]
    return int(tail) if tail.isdigit() and int(tail) != 99999 else None


def ghcnh_wban(sid: str):
    s = str(sid).strip()
    if len(s) == 11 and s[2] == "W" and s[3:].isdigit():
        w = int(s[3:])
        return w if w != 0 else None
    return None


def hourly_sum(df: pd.DataFrame) -> pd.Series:
    """(station_id, hour) -> summed precip_in over valid rows only."""
    df = df.dropna(subset=["precip_in"])
    hour = df["datetime_utc"].dt.floor("h")
    return df.groupby([df["station_id"], hour])["precip_in"].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--wet-threshold", type=float, default=0.01,
                    help="inches at/above which an hour counts as 'wet'")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    base = f"s3://{bucket}/{prefix}precip/noaa"

    # ── station crosswalk via WBAN ────────────────────────────────────────────
    isd_meta   = pd.read_parquet(f"{base}/isd_hourly.parquet",
                                 columns=["station_id", "name"]).drop_duplicates("station_id")
    ghcnh_ids  = pd.read_parquet(f"{base}/ghcnh_hourly.parquet",
                                 columns=["station_id"]).drop_duplicates()["station_id"]

    g_wban = {}
    for sid in ghcnh_ids:
        w = ghcnh_wban(sid)
        if w is not None:
            g_wban.setdefault(w, sid)          # first wins on the rare collision

    pairs = []
    for _, r in isd_meta.iterrows():
        w = isd_wban(r["station_id"])
        if w is not None and w in g_wban:
            pairs.append((r["station_id"], g_wban[w], w, r["name"]))
    print(f"Matched pairs to compare: {len(pairs)}\n")
    isd_ids   = {p[0] for p in pairs}
    ghcnh_sel = {p[1] for p in pairs}

    # ── hourly series for matched stations only ───────────────────────────────
    isd = pd.read_parquet(f"{base}/isd_hourly.parquet",
                          columns=["station_id", "datetime_utc", "precip_in"])
    isd = isd[isd["station_id"].isin(isd_ids)]
    isd["datetime_utc"] = pd.to_datetime(isd["datetime_utc"], utc=True)

    ghc = pd.read_parquet(f"{base}/ghcnh_hourly.parquet",
                          columns=["station_id", "datetime_utc", "precip_in"])
    ghc = ghc[ghc["station_id"].isin(ghcnh_sel)]
    ghc["datetime_utc"] = pd.to_datetime(ghc["datetime_utc"], utc=True)

    isd_h = hourly_sum(isd)
    ghc_h = hourly_sum(ghc)

    wt = args.wet_threshold
    rows = []
    for isd_id, ghcnh_id, wban, name in pairs:
        a = isd_h.loc[isd_id] if isd_id in isd_h.index.get_level_values(0) else pd.Series(dtype=float)
        b = ghc_h.loc[ghcnh_id] if ghcnh_id in ghc_h.index.get_level_values(0) else pd.Series(dtype=float)

        ai, bi = a.index, b.index
        shared = ai.intersection(bi)
        isd_only = ai.difference(bi)
        ghc_only = bi.difference(ai)

        corr = mae = np.nan
        wet_isd = wet_both = 0
        if len(shared):
            av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
            mae = float(np.abs(av - bv).mean())
            if len(shared) > 1 and av.std() > 0 and bv.std() > 0:
                corr = float(np.corrcoef(av, bv)[0, 1])
            wet_isd  = int((av >= wt).sum())
            wet_both = int(((av >= wt) & (bv >= wt)).sum())

        rows.append(dict(
            isd_id=isd_id, ghcnh_id=ghcnh_id, wban=wban, name=name,
            isd_first=ai.min() if len(ai) else pd.NaT,
            isd_last=ai.max() if len(ai) else pd.NaT,
            ghcnh_first=bi.min() if len(bi) else pd.NaT,
            ghcnh_last=bi.max() if len(bi) else pd.NaT,
            isd_hours=len(ai), ghcnh_hours=len(bi),
            shared_hours=len(shared),
            isd_only_hours=len(isd_only), ghcnh_only_hours=len(ghc_only),
            overlap_frac_isd=round(len(shared) / len(ai), 4) if len(ai) else np.nan,
            corr=round(corr, 4) if corr == corr else np.nan,
            mae_in=round(mae, 4) if mae == mae else np.nan,
            wet_detect_agree=round(wet_both / wet_isd, 4) if wet_isd else np.nan,
        ))

    df = pd.DataFrame(rows)

    # ── summary ───────────────────────────────────────────────────────────────
    print("=== Coverage (summed over the 165 pairs) ===")
    print(f"  ISD valid hours        : {df.isd_hours.sum():,}")
    print(f"  GHCNh valid hours      : {df.ghcnh_hours.sum():,}")
    print(f"  Shared hours           : {df.shared_hours.sum():,}")
    print(f"  ISD-only hours         : {df.isd_only_hours.sum():,}   "
          f"(hours ISD has that GHCNh lacks, same station)")
    print(f"  GHCNh-only hours       : {df.ghcnh_only_hours.sum():,}\n")

    print("=== Per-station distribution ===")
    print(f"  median ISD/GHCNh overlap frac : {df.overlap_frac_isd.median():.3f}")
    print(f"  median corr on shared hours   : {df.corr.median():.3f}")
    print(f"  median MAE on shared hours    : {df.mae_in.median():.4f} in")
    print(f"  median wet-hour detect agree  : {df.wet_detect_agree.median():.3f}")
    print(f"  stations ISD starts earlier   : {(df.isd_first < df.ghcnh_first).sum()}")
    print(f"  stations ISD ends later       : {(df.isd_last  > df.ghcnh_last ).sum()}")
    for thr in (100, 1000, 8760):
        print(f"  stations with >{thr:>4} ISD-only hrs : {(df.isd_only_hours > thr).sum()}")

    out = Path(args.repo) / "isd_vs_ghcnh_record_comparison.csv"
    df.sort_values("isd_only_hours", ascending=False).to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")
    print("\nTop 10 stations by ISD-only hours:")
    cols = ["isd_id", "ghcnh_id", "name", "isd_hours", "ghcnh_hours",
            "shared_hours", "isd_only_hours", "corr", "mae_in"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.sort_values("isd_only_hours", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
