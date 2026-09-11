#!/usr/bin/env python3
"""Per-bridge audit: which reach is nearest, and does the NWM actually route it?

The flow trigger reads streamflow on a bridge's assigned COMID. Three separate
things can go wrong, and they need separate numbers because they need separate
fixes:

  1. the bridge is assigned to a reach far from it          -> re-snap
  2. the nearest reach exists in NHDPlus but NWM does not
     route it, so it carries no streamflow                  -> cannot be monitored by flow
  3. the assigned reach has an implausible LP3 fit          -> null the quantiles

Distances are nearest-vertex over the full Indiana NHDPlus V2.1 export
(results/indiana_nwm_comids.shp, from export_indiana_comids_shp.py), which holds
every reach, not only the ~6.9k a bridge was assigned to.

Writes  results/bridge_reach_snapping_audit.csv    (one row per bridge)
        results/bridge_reach_snapping_audit.parquet

Usage:
    python scripts/audit_bridge_reach_snapping.py
"""
from __future__ import annotations

import argparse
import io
import pathlib
from collections import defaultdict

import boto3
import numpy as np
import pandas as pd
import shapefile

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
R_KM = 6371.0
CELL = 0.02                 # ~2 km grid bucket
BORDER_DEG = 0.05           # how close to the clip edge counts as "border"
IN_BBOX = (-88.20, 37.70, -84.70, 41.90)


def s3_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}{key}")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def build_index(shp_path):
    r = shapefile.Reader(str(shp_path))
    recs = r.records()
    meta = {int(x["COMID"]): x for x in recs}
    cid, lo, la = [], [], []
    for rec, sh in zip(recs, r.shapes()):
        c = int(rec["COMID"])
        for x, y in sh.points:
            cid.append(c); lo.append(x); la.append(y)
    cid = np.asarray(cid, dtype="int64")
    lo = np.asarray(lo); la = np.asarray(la)
    routed = np.asarray([meta[c]["IN_NWM"] == 1 for c in cid])
    grid = defaultdict(list)
    for i, (x, y) in enumerate(zip(lo, la)):
        grid[(int(x / CELL), int(y / CELL))].append(i)
    grid = {k: np.asarray(v) for k, v in grid.items()}
    print(f"  {len(recs):,} reaches | {len(lo):,} vertices | "
          f"{routed.sum():,} on NWM-routed reaches")
    return meta, cid, lo, la, routed, grid


def make_nearest(cid, lo, la, routed, grid):
    def nearest(lat, lon, routed_only=False, max_rings=40):
        """Expanding-ring search so a sparse area still resolves rather than
        silently returning 'nothing within the first box'."""
        gx, gy = int(lon / CELL), int(lat / CELL)
        for rings in (2, 5, 10, 20, max_rings):
            idx = [grid[(gx + dx, gy + dy)]
                   for dx in range(-rings, rings + 1)
                   for dy in range(-rings, rings + 1)
                   if (gx + dx, gy + dy) in grid]
            if not idx:
                continue
            idx = np.concatenate(idx)
            if routed_only:
                idx = idx[routed[idx]]
            if not len(idx):
                continue
            d = np.hypot(np.radians(la[idx] - lat),
                         np.radians(lo[idx] - lon) * np.cos(np.radians(lat))) * R_KM
            j = int(np.argmin(d))
            # a hit inside the searched box is only trustworthy if it is closer
            # than the box edge; otherwise widen and try again
            if d[j] <= rings * CELL * 85 or rings == max_rings:
                return int(cid[idx[j]]), float(d[j])
        return None, float("nan")
    return nearest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default="results/indiana_nwm_comids")
    ap.add_argument("--out", default="results/bridge_reach_snapping_audit")
    args = ap.parse_args()

    print("Loading ...")
    cfg = s3_parquet("monitor/bridge_monitor_config.parquet")
    meta, cid, lo, la, routed, grid = build_index(args.shp)
    nearest = make_nearest(cid, lo, la, routed, grid)

    b = cfg.copy()
    b["assigned"] = pd.to_numeric(b["comid"], errors="coerce")
    b = b.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  {len(b):,} bridges")

    # distance to the ASSIGNED reach
    byc = {}
    for c in b.assigned.dropna().astype("int64").unique():
        w = np.where(cid == c)[0]
        if len(w):
            byc[c] = w
    d_assigned = np.full(len(b), np.nan)
    for i, (lat, lon, c) in enumerate(zip(b.lat, b.lon, b.assigned)):
        if pd.isna(c):
            continue
        w = byc.get(int(c))
        if w is None:
            continue
        d = np.hypot(np.radians(la[w] - lat),
                     np.radians(lo[w] - lon) * np.cos(np.radians(lat))) * R_KM
        d_assigned[i] = d.min()
    b["d_assigned_km"] = d_assigned

    print("Resolving nearest reaches (this is the slow part) ...")
    na, da, nr, dr = [], [], [], []
    for k, (lat, lon) in enumerate(zip(b.lat, b.lon)):
        c1, d1 = nearest(lat, lon)
        c2, d2 = nearest(lat, lon, routed_only=True)
        na.append(c1); da.append(d1); nr.append(c2); dr.append(d2)
        if k % 2000 == 0:
            print(f"    {k:,} / {len(b):,}", end="\r")
    print(f"    {len(b):,} / {len(b):,}        ")

    b["nearest_comid"] = na
    b["d_nearest_km"] = da
    b["nearest_routed_comid"] = nr
    b["d_nearest_routed_km"] = dr
    b["nearest_in_nwm"] = [bool(meta[c]["IN_NWM"] == 1) if c in meta else False for c in na]
    b["nearest_ftype"] = [meta[c]["FTYPE_NAME"] if c in meta else "" for c in na]
    b["assigned_in_nwm"] = [bool(meta[int(c)]["IN_NWM"] == 1)
                            if pd.notna(c) and int(c) in meta else False for c in b.assigned]
    b["has_flow_gate"] = b["Q50_cfs"].notna()
    # Distance to the STATE LINE, not to the bounding box. The reach export is
    # clipped to Indiana, so for a bridge near the border the true nearest reach
    # may sit in Illinois/Ohio/Michigan/Kentucky and be missing here. Measuring
    # against the bbox instead would report a reassuring zero and mean nothing.
    counties = s3_parquet("monitor/assets/in_counties.parquet")
    cl_lon = counties.lon.to_numpy(); cl_lat = counties.lat.to_numpy()
    d_border = np.empty(len(b))
    for i, (lat, lon) in enumerate(zip(b.lat.to_numpy(), b.lon.to_numpy())):
        m = (np.abs(cl_lat - lat) < 0.6) & (np.abs(cl_lon - lon) < 0.6)
        if not m.any():
            d_border[i] = np.inf; continue
        d_border[i] = (np.hypot(np.radians(cl_lat[m] - lat),
                                np.radians(cl_lon[m] - lon) * np.cos(np.radians(lat))).min()
                       * R_KM)
    b["km_to_state_line"] = d_border
    b["border"] = b.km_to_state_line < b.d_nearest_km

    cols = ["bridge_id", "Asset Name", "lat", "lon", "scour_critical", "has_flow_gate",
            "assigned", "d_assigned_km", "assigned_in_nwm",
            "nearest_comid", "d_nearest_km", "nearest_in_nwm", "nearest_ftype",
            "nearest_routed_comid", "d_nearest_routed_km", "km_to_state_line", "border",
            "Q10_cfs", "Q50_cfs", "Q100_cfs"]
    out = b[[c for c in cols if c in b.columns]].copy()
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p.with_suffix(".csv"), index=False)
    out.to_parquet(p.with_suffix(".parquet"), index=False)

    # ---- report ------------------------------------------------------------
    n = len(b)
    sc = b.scour_critical.astype(bool)
    print(f"\n{'='*72}\nFLEET-WIDE SNAPPING AUDIT — {n:,} bridges\n{'='*72}")

    print("\nDistance to the ASSIGNED reach")
    for lab, m in [("<= 200 ft (61 m)", b.d_assigned_km <= .061),
                   ("61 m - 200 m", b.d_assigned_km.between(.061, .2, "right")),
                   ("200 m - 1 km", b.d_assigned_km.between(.2, 1., "right")),
                   ("1 - 5 km", b.d_assigned_km.between(1., 5., "right")),
                   ("> 5 km", b.d_assigned_km > 5.)]:
        print(f"  {lab:>18} : {int(m.sum()):6,} ({m.mean()*100:5.1f}%)  scour {int((m & sc).sum()):3,}")

    print("\nIs the bridge's NEAREST reach routed by the NWM?")
    yes = b.nearest_in_nwm
    print(f"  nearest reach IS NWM-routed     : {int(yes.sum()):6,} ({yes.mean()*100:5.1f}%)")
    print(f"  nearest reach NOT NWM-routed    : {int((~yes).sum()):6,} ({(~yes).mean()*100:5.1f}%)"
          f"  scour {int((~yes & sc).sum()):,}")

    un = b[~yes]
    print(f"\nThe {len(un):,} whose nearest reach is NOT routed:")
    print(f"  median distance to that nearest reach   : {un.d_nearest_km.median():6.3f} km")
    print(f"  median distance to nearest ROUTED reach : {un.d_nearest_routed_km.median():6.3f} km")
    for cut in (.061, .2, .5, 1., 2.):
        m = un.d_nearest_routed_km <= cut
        print(f"    routed reach within {cut:>5} km : {int(m.sum()):5,} ({m.mean()*100:5.1f}%)")
    print(f"  their nearest reach is a ...")
    print("   " + un.nearest_ftype.value_counts().to_string().replace("\n", "\n   "))
    print(f"  currently carry a flow gate anyway      : {int(un.has_flow_gate.sum()):,}")
    print(f"  closer to the STATE LINE than to that reach: {int(un.border.sum()):,}"
          "  (an out-of-state reach could be nearer still)")

    print("\nHow the assigned reach compares with the nearest one")
    better = b.d_assigned_km > b.d_nearest_km + .05
    print(f"  a strictly nearer reach exists          : {int(better.sum()):6,} ({better.mean()*100:5.1f}%)")
    m = better & b.nearest_in_nwm
    print(f"    ... and it IS NWM-routed             : {int(m.sum()):6,}  <- re-snap fixes these")
    m2 = better & ~b.nearest_in_nwm
    print(f"    ... but it is NOT routed             : {int(m2.sum()):6,}  <- no flow data either way")

    print(f"\nWrote {p}.csv and {p}.parquet")


if __name__ == "__main__":
    main()
