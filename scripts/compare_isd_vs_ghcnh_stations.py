#!/usr/bin/env python3
"""compare_isd_vs_ghcnh_stations.py

Answer: do we have any ISD station that has NO GHCNh counterpart?

Script 12 downloaded NOAA hourly precip from two sources with DIFFERENT id
schemes, so a plain set-difference on station_id is wrong:

    ISD   station_id = USAF(6) + WBAN(5)      e.g. 72530094846  -> WBAN 94846
    GHCNh station_id = 11-char GHCN id        e.g. USW00094846  (W-net embeds WBAN)
                                                   USC00xxxxxx  (COOP, no WBAN)

Matching strategy (ISD -> GHCNh):
    1. WBAN crosswalk: ISD WBAN vs GHCNh W-network WBAN (id[2]=='W').
    2. Spatial fallback for anything unmatched: nearest GHCNh station within
       --tol-km (default 1.5 km) counts as the same site (catches COOP re-ids
       and small coordinate rounding).
ISD stations matched by neither are "in ISD, not in GHCNh".

By default compares the HOURLY parquets (stations we actually have DATA for).
Use --inventory to compare the selected-station lists instead.

Run on EC2 (instance role must read the private output bucket):
    python scripts/compare_isd_vs_ghcnh_stations.py
    python scripts/compare_isd_vs_ghcnh_stations.py --tol-km 3
    python scripts/compare_isd_vs_ghcnh_stations.py --inventory
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


def distinct_stations(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["station_id", "name", "latitude", "longitude"])
    return (df.dropna(subset=["latitude", "longitude"])
              .drop_duplicates("station_id")
              .reset_index(drop=True))


def isd_wban(sid: str) -> int | None:
    s = str(sid).strip()
    tail = s[-5:]
    return int(tail) if tail.isdigit() and int(tail) != 99999 else None


def ghcnh_wban(sid: str) -> int | None:
    s = str(sid).strip()
    if len(s) == 11 and s[2] == "W" and s[3:].isdigit():   # US WBAN-network station
        w = int(s[3:])
        return w if w != 0 else None
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tol-km", type=float, default=1.5,
                    help="spatial-fallback radius for calling two stations the same site")
    ap.add_argument("--inventory", action="store_true",
                    help="compare selected-station lists instead of the hourly data")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.repo))
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    base = f"s3://{bucket}/{prefix}precip/noaa"
    isd_key   = "stations_isd.parquet"   if args.inventory else "isd_hourly.parquet"
    ghcnh_key = "stations_ghcnh.parquet" if args.inventory else "ghcnh_hourly.parquet"

    print(f"ISD   : {base}/{isd_key}")
    print(f"GHCNh : {base}/{ghcnh_key}\n")
    isd   = distinct_stations(f"{base}/{isd_key}")
    ghcnh = distinct_stations(f"{base}/{ghcnh_key}")
    print(f"Distinct stations -> ISD: {len(isd):,} | GHCNh: {len(ghcnh):,}\n")

    # 1. WBAN crosswalk
    isd["wban"] = isd["station_id"].map(isd_wban)
    ghcnh_wbans = set(w for w in ghcnh["station_id"].map(ghcnh_wban) if w is not None)
    isd["match_wban"] = isd["wban"].map(lambda w: w in ghcnh_wbans if w is not None else False)

    # 2. spatial fallback for the rest
    glat = ghcnh["latitude"].to_numpy()
    glon = ghcnh["longitude"].to_numpy()
    nearest_id, nearest_km, matched_spatial = [], [], []
    for _, row in isd.iterrows():
        d = haversine_km(row["latitude"], row["longitude"], glat, glon)
        j = int(np.argmin(d))
        nearest_id.append(ghcnh.iloc[j]["station_id"])
        nearest_km.append(float(d[j]))
        matched_spatial.append(bool(d[j] <= args.tol_km))
    isd["nearest_ghcnh"] = nearest_id
    isd["nearest_km"]    = np.round(nearest_km, 3)
    isd["match_spatial"] = matched_spatial

    isd["in_ghcnh"] = isd["match_wban"] | isd["match_spatial"]
    only = isd[~isd["in_ghcnh"]].sort_values("nearest_km", ascending=False)

    n_wban = int(isd["match_wban"].sum())
    n_spat = int((~isd["match_wban"] & isd["match_spatial"]).sum())
    print(f"Matched to GHCNh by WBAN     : {n_wban:,}")
    print(f"Matched to GHCNh spatially   : {n_spat:,}  (<= {args.tol_km} km)")
    print(f"ISD stations NOT in GHCNh    : {len(only):,}\n")

    if len(only):
        show = only[["station_id", "wban", "name", "latitude", "longitude",
                     "nearest_ghcnh", "nearest_km"]]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(show.to_string(index=False))
        out = Path(args.repo) / "isd_not_in_ghcnh.csv"
        show.to_csv(out, index=False)
        print(f"\nWrote {out}")
    else:
        print("Every ISD station has a GHCNh counterpart (by WBAN or proximity).")


if __name__ == "__main__":
    main()
