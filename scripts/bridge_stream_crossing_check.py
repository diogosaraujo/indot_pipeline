"""Which bridges actually cross an NWM-routed stream?

The monitor assigns every over-water bridge an NHDPlus V2.1 COMID and fires its
flow trigger against that reach's LP3 thresholds. But a COMID is assigned
whether or not the bridge is on that stream, and the failure is invisible:

  49-1912L sits on Devon Creek (~9 km2) and is assigned Fall Creek COMID
  18476443 (320 mi2, Q50 = 9,170 cfs). It is 392 m from Fall Creek, which
  passes every distance check we had, and its gate can never be reached. A
  bridge that cannot fire looks exactly like a bridge that never needed to.

NHDPlus V2.1 (medium resolution, ~1:100k) is what the NWM routes on; it simply
does not contain Devon Creek. NHDPlus HR (high resolution, ~1:24k) does. So HR
cannot give thresholds, but it can say which stream a bridge is actually on.

The test
────────
  1. nearest HR flowline to the bridge, by TRUE PERPENDICULAR distance
     (point-to-segment). NOT a nearest-vertex snap: vertex snapping picks the
     wrong reach wherever vertices are sparse, which is how this class of error
     was introduced in the first place.
  2. compare that flowline's `reachcode` with the REACHCODE of the bridge's
     assigned V2.1 COMID.

ReachCode is the ONLY key the two product lines share. HR has no COMID and V2.1
has no NHDPlusID, so there is no other join.

Deliberately NOT used for matching: geometry and drainage area. HR breaks a
single V2.1 reach into many short segments carrying different `totdasqkm`
(COMID 18476443 -> 10 HR segments), so neither is comparable across products.
`totdasqkm` is carried into the output as CONTEXT ONLY — it is what makes a
mismatch legible ("crosses a 9 km2 creek, assigned a 320 mi2 river"), never
what decides one.

STATUS (binary)
  CROSSING      the two ReachCodes are equal — the NWM network contains the
                stream this bridge is nearest to.
  NOT_CROSSING  they differ (or either is missing).

Nothing else enters the decision. There is no distance tolerance, the assigned
COMID is not consulted, and no geometry or drainage area is compared. Distance
is used only to pick which reach is nearest within each product — the nearest
search is unbounded — and is never tested against a threshold. V2_NEAR_M,
HR_DIST_M, V2_DIST_M and HR_DA_KM2 are carried into the output for context and
are not inputs to STATUS.

Outputs (results/)
    bridges_crossing_nwm.shp       STATUS == CROSSING
    bridges_not_crossing_nwm.shp   STATUS == NOT_CROSSING
    bridge_stream_crossing.csv     every bridge, all fields
    bridge_stream_crossing.parquet

Usage
    python scripts/bridge_stream_crossing_check.py
    python scripts/bridge_stream_crossing_check.py --hr-gdb "Z:/.../x.gdb"
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crossing")

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GDB = (r"Z:\INDOT - Flooding\National Water Model"
               r"\nhdplus_epasnapshot2022_in_fgdb\nhdplus_epasnapshot2022_in.gdb")
V2_SHP = REPO / "results" / "indiana_nwm_comids.shp"
OUT_DIR = REPO / "results"

CONFIG_KEY = "monitor/bridge_monitor_config.parquet"
WORK_CRS = 26916          # UTM 16N — metres, the CRS used elsewhere in the repo

# Step 1 tolerance: 200 ft, the pipeline's own documented snapping rule.
V2_TOL_M = 61.0

# A bridge does not cross a buried line, so Pipeline (428) and Underground
# Conduit (420) are not candidates for "the stream this bridge crosses".
HR_FTYPE_KEEP = {460, 558, 336, 334}   # StreamRiver, ArtificialPath, CanalDitch, Connector


def load_bridges() -> gpd.GeoDataFrame:
    import sys
    sys.path.insert(0, str(REPO / "monitor"))
    from monitor_common import config as mcfg
    from monitor_common.s3io import read_parquet
    b, p = mcfg.bucket_prefix()
    d = read_parquet(b, f"{p}{CONFIG_KEY}")
    d["bridge_id"] = d["bridge_id"].astype(str).str.strip()
    d["comid"] = pd.to_numeric(d["comid"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["lat", "lon"])
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["lon"], d["lat"]),
                         crs=4326)
    log.info("bridges: %d  (%d with an assigned COMID)", len(g), int(g["comid"].notna().sum()))
    return g


def load_v2() -> pd.DataFrame:
    v = gpd.read_file(V2_SHP, columns=["COMID", "REACHCODE", "GNIS_NAME"])
    v = pd.DataFrame(v.drop(columns="geometry"))
    v["COMID"] = pd.to_numeric(v["COMID"], errors="coerce").astype("Int64")
    v["REACHCODE"] = v["REACHCODE"].astype(str).str.zfill(14)
    log.info("V2.1 reaches: %d  (%d distinct reachcodes)", len(v), v["REACHCODE"].nunique())
    return v


def load_hr(gdb: str) -> gpd.GeoDataFrame:
    log.info("reading HR flowlines (this is ~2.3M features)...")
    h = pyogrio.read_dataframe(
        gdb, layer="nhdflowline_in",
        columns=["nhdplusid", "reachcode", "gnis_name", "ftype"])
    log.info("  read %d; filtering to watercourse ftypes %s", len(h), sorted(HR_FTYPE_KEEP))
    h = h[h["ftype"].isin(HR_FTYPE_KEEP)].copy()
    h["reachcode"] = h["reachcode"].astype(str).str.zfill(14)
    h = h[h.geometry.notna() & ~h.geometry.is_empty]
    log.info("  %d candidate flowlines", len(h))
    return h


def attach_vaa(h: gpd.GeoDataFrame, gdb: str) -> gpd.GeoDataFrame:
    """totdasqkm / streamorde — CONTEXT ONLY, never a matching criterion."""
    v = pyogrio.read_dataframe(gdb, layer="nhdplusflowlinevaa_in",
                               read_geometry=False,
                               columns=["nhdplusid", "totdasqkm", "streamorde"])
    out = h.merge(v, on="nhdplusid", how="left")
    log.info("  VAA attached (%d/%d have totdasqkm)", int(out["totdasqkm"].notna().sum()), len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr-gdb", default=DEFAULT_GDB)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    br = load_bridges()
    v2 = load_v2()
    hr = attach_vaa(load_hr(args.hr_gdb), args.hr_gdb)

    v2_codes = set(v2["REACHCODE"])
    br = br.merge(v2.rename(columns={"COMID": "comid", "REACHCODE": "v2_reach",
                                     "GNIS_NAME": "v2_name"}),
                  on="comid", how="left")

    log.info("nearest HR flowline by perpendicular distance...")
    brp = br.to_crs(WORK_CRS)
    hrp = hr.to_crs(WORK_CRS)
    # Nearest is UNBOUNDED: no max_distance. Distance is used only to decide
    # which reach is nearest and is never compared against a threshold.
    # sjoin_nearest measures point-to-GEOMETRY distance, i.e. to the nearest
    # point ON the segment — the perpendicular foot, not the nearest vertex.
    j = gpd.sjoin_nearest(brp[["bridge_id", "geometry"]], hrp,
                          how="left", distance_col="hr_dist_m")
    j = (j.sort_values("hr_dist_m")
           .drop_duplicates("bridge_id")                 # ties -> keep the closest
           [["bridge_id", "nhdplusid", "reachcode", "gnis_name", "ftype",
             "totdasqkm", "streamorde", "hr_dist_m"]])
    d = br.merge(j, on="bridge_id", how="left")

    # ── nearest V2.1 reach (unbounded), plus its ReachCode ───────────────────
    v2g = gpd.read_file(V2_SHP, columns=["COMID", "REACHCODE"]).to_crs(WORK_CRS)
    v2g["COMID"] = pd.to_numeric(v2g["COMID"], errors="coerce").astype("Int64")
    v2g["REACHCODE"] = v2g["REACHCODE"].astype(str).str.zfill(14)
    geom_by_comid = dict(zip(v2g["COMID"], v2g.geometry))
    # distance to the ASSIGNED reach — reported only, not part of the test
    d["v2_dist_m"] = [
        (geom_by_comid[c].distance(p) if (pd.notna(c) and c in geom_by_comid) else np.nan)
        for c, p in zip(d["comid"], brp.geometry)]
    jv = gpd.sjoin_nearest(brp[["bridge_id", "geometry"]], v2g,
                           how="left", distance_col="v2_near_m")
    jv = (jv.sort_values("v2_near_m").drop_duplicates("bridge_id")
            [["bridge_id", "COMID", "REACHCODE", "v2_near_m"]]
            .rename(columns={"COMID": "v2_near_comid",
                             "REACHCODE": "v2_near_reach"}))
    d = d.merge(jv, on="bridge_id", how="left")

    # ── THE TEST ─────────────────────────────────────────────────────────────
    # Three steps, in order:
    #   1  the nearest V2.1 reach is within V2_TOL_M        -> CROSSING
    #   2  else, ReachCode(nearest V2.1) == ReachCode(nearest HR) -> CROSSING
    #   3  everything else                                  -> NOT_CROSSING
    #
    # The two steps rescue each other's failures. Step 1 alone misses a bridge
    # whose V2.1 line is drawn beyond the tolerance from the real channel
    # (149-64-04025 C: 93.9 m off, same stream). Step 2 alone misses a bridge
    # sitting on a V2.1 line that the two products number differently
    # (45-00248 B: 2.2 m away, but V2.1 says 0712... and HR says 0404...).
    #
    # Step 1 uses the NEAREST V2.1 reach, not the assigned COMID: the question
    # is whether the NWM network contains this stream, not whether the existing
    # assignment points at the right reach.
    hr_rc = d["reachcode"].astype(str).str.zfill(14)
    v2_rc = d["v2_near_reach"].astype(str).str.zfill(14)
    d["hr_near_reach"] = hr_rc

    step1 = (d["v2_near_m"].notna() & (d["v2_near_m"] <= V2_TOL_M)).to_numpy(dtype=bool)
    rc_same = (d["reachcode"].notna() & d["v2_near_reach"].notna()
               & (hr_rc == v2_rc)).to_numpy(dtype=bool)
    step2 = (~step1) & rc_same

    d["status"] = np.where(step1 | step2, "CROSSING", "NOT_CROSSING")
    # which step decided it — reporting only, does not affect STATUS
    d["rule"] = np.select([step1, step2], ["STEP1_DIST", "STEP2_RC"],
                          default="NEITHER")

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(d)
    log.info("=" * 70)
    log.info("IS THE BRIDGE ON A STREAM THE NWM NETWORK CONTAINS?")
    log.info("=" * 70)
    log.info("  step 1  nearest V2.1 reach within %.0f m                     -> CROSSING", V2_TOL_M)
    log.info("  step 2  else ReachCode(nearest V2.1) == ReachCode(nearest HR) -> CROSSING")
    log.info("  step 3  everything else                                       -> NOT_CROSSING")
    log.info("")
    for st in ("CROSSING", "NOT_CROSSING"):
        k = int((d["status"] == st).sum())
        sc = int(d.loc[d["status"] == st, "scour_critical"].astype(bool).sum()) \
            if "scour_critical" in d.columns else 0
        log.info("  %-13s %6d  (%5.1f%%)   scour-critical %4d", st, k, 100.0 * k / n, sc)
    log.info("")
    for rl, lab in (("STEP1_DIST", "step 1 (within %.0f m)" % V2_TOL_M),
                    ("STEP2_RC", "step 2 (ReachCode match)"),
                    ("NEITHER", "step 3 (neither)")):
        k = int((d["rule"] == rl).sum())
        log.info("    decided by %-26s %6d  (%5.1f%%)", lab, k, 100.0 * k / n)

    nc = d[d["status"] == "NOT_CROSSING"]
    if len(nc):
        log.info("")
        log.info("  NOT_CROSSING: the streams these bridges are nearest to in HR")
        for k_, c_ in nc["gnis_name"].fillna("(unnamed)").value_counts().head(8).items():
            log.info("    %-28s %4d bridges", k_, c_)
        log.info("    median HR drainage area %.1f km2", nc["totdasqkm"].median())
        log.info("")
        log.info("  context only, NOT used by the test:")
        log.info("    distance to nearest V2.1 reach  p50 %7.1f m   p90 %9.1f m",
                 nc["v2_near_m"].median(), nc["v2_near_m"].quantile(.9))
        log.info("    distance to nearest HR flowline p50 %7.1f m   p90 %9.1f m",
                 nc["hr_dist_m"].median(), nc["hr_dist_m"].quantile(.9))

    # ── write ────────────────────────────────────────────────────────────────
    cols = {"bridge_id": "BRIDGE_ID", "comid": "COMID", "v2_reach": "V2_REACH",
            "v2_name": "V2_NAME", "v2_dist_m": "V2_DIST_M", "nhdplusid": "HR_ID",
            "reachcode": "HR_REACH", "gnis_name": "HR_NAME", "ftype": "HR_FTYPE",
            "hr_dist_m": "HR_DIST_M", "totdasqkm": "HR_DA_KM2",
            "streamorde": "HR_ORDER",
            "v2_near_comid": "V2_NEAR_ID", "v2_near_m": "V2_NEAR_M",
            "v2_near_reach": "V2_NEAR_RC", "hr_near_reach": "HR_NEAR_RC",
            "status": "STATUS", "rule": "RULE",
            "scour_critical": "SCOUR", "Q50_cfs": "Q50_CFS", "lat": "LAT", "lon": "LON"}
    keep = [c for c in cols if c in d.columns]
    out = gpd.GeoDataFrame(d[keep].rename(columns=cols),
                           geometry=d.geometry, crs=4326)
    if "SCOUR" in out.columns:
        out["SCOUR"] = out["SCOUR"].astype(bool).astype(int)
    out["HR_ID"] = out["HR_ID"].astype("Float64").astype(str)   # 14-digit, keep exact

    # The split answers "is this bridge on a stream the NWM routes?", so a
    # WRONG_COMID bridge belongs with the crossing set: it IS on a routed reach,
    # just assigned a different one. 45-00248 B sits 2.2 m from COMID 13463811
    # while assigned one 4.6 km away — real coverage, wrong pointer. Filing it
    # as "not crossing" would read as no coverage at all. STATUS keeps the
    # distinction, so re-splitting on it is one attribute query.
    a = out[out["STATUS"] == "CROSSING"]
    b = out[out["STATUS"] == "NOT_CROSSING"]
    a.to_file(out_dir / "bridges_crossing_nwm.shp")
    b.to_file(out_dir / "bridges_not_crossing_nwm.shp")
    out.drop(columns="geometry").to_csv(out_dir / "bridge_stream_crossing.csv", index=False)
    out.drop(columns="geometry").to_parquet(out_dir / "bridge_stream_crossing.parquet", index=False)
    log.info("")
    log.info("wrote %s (%d) and %s (%d), plus the full table as .csv/.parquet",
             "bridges_crossing_nwm.shp", len(a), "bridges_not_crossing_nwm.shp", len(b))


if __name__ == "__main__":
    main()
