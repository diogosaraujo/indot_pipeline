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

STATUS
──────
  CROSSES_NWM  HR reachcode == the assigned COMID's reachcode. Correct.
  WRONG_COMID  HR reachcode is in V2.1 but belongs to a different COMID — the
               bridge is on a routed stream, just assigned the wrong reach.
               Recoverable by re-snapping.
  NOT_ROUTED   HR reachcode has no V2.1 counterpart — the NWM has no reach for
               this stream. No flow trigger is possible; re-snapping cannot help.
  NO_HR_NEAR   no HR flowline within MAX_SEARCH_M.

Outputs (results/)
    bridges_crossing_nwm.shp       STATUS == CROSSES_NWM
    bridges_not_crossing_nwm.shp   everything else
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
MAX_SEARCH_M = 5000.0
# How close a bridge must be to a V2.1 reach to count as "on" it. V2.1 is
# ~1:100k, so its geometry is generalised: bridges that genuinely sit on a
# V2.1 reach measure p50 11 m / p90 30 m, against ~2 m for HR at 1:24k.
# 61 m is the pipeline's own documented 200 ft rule.
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
    # sjoin_nearest measures point-to-GEOMETRY distance, i.e. to the nearest
    # point ON the segment — the perpendicular foot, not the nearest vertex.
    j = gpd.sjoin_nearest(brp[["bridge_id", "geometry"]], hrp,
                          how="left", max_distance=MAX_SEARCH_M,
                          distance_col="hr_dist_m")
    j = (j.sort_values("hr_dist_m")
           .drop_duplicates("bridge_id")                 # ties -> keep the closest
           [["bridge_id", "nhdplusid", "reachcode", "gnis_name", "ftype",
             "totdasqkm", "streamorde", "hr_dist_m"]])
    d = br.merge(j, on="bridge_id", how="left")

    # ── classification 1: the reachcode test, as specified ───────────────────
    hr_rc = d["reachcode"]
    same = hr_rc.notna() & d["v2_reach"].notna() & (hr_rc == d["v2_reach"])
    in_v2 = hr_rc.isin(v2_codes)
    d["status_rc"] = np.select(
        [hr_rc.isna(), same, in_v2],
        ["NO_HR_NEAR", "CROSSES_NWM", "WRONG_COMID"],
        default="NOT_ROUTED")

    # ── nearest V2.1 reach, and distance to the ASSIGNED one ─────────────────
    v2g = gpd.read_file(V2_SHP, columns=["COMID"]).to_crs(WORK_CRS)
    v2g["COMID"] = pd.to_numeric(v2g["COMID"], errors="coerce").astype("Int64")
    geom_by_comid = dict(zip(v2g["COMID"], v2g.geometry))
    d["v2_dist_m"] = [
        (geom_by_comid[c].distance(p) if (pd.notna(c) and c in geom_by_comid) else np.nan)
        for c, p in zip(d["comid"], brp.geometry)]
    jv = gpd.sjoin_nearest(brp[["bridge_id", "geometry"]], v2g,
                           how="left", max_distance=MAX_SEARCH_M,
                           distance_col="v2_near_m")
    jv = (jv.sort_values("v2_near_m").drop_duplicates("bridge_id")
            [["bridge_id", "COMID", "v2_near_m"]]
            .rename(columns={"COMID": "v2_near_comid"}))
    d = d.merge(jv, on="bridge_id", how="left")

    # ── classification 2: geometric, per product — the defensible one ────────
    # The reachcode test alone over-reports NOT_ROUTED: HR and V2.1 number
    # reaches differently, so the same watercourse can carry unrelated codes in
    # the two products. Asking each product separately "is there a stream here?"
    # avoids any cross-product comparison, which is exactly the comparability
    # problem that rules out geometry and drainage-area matching.
    #   NOT_ROUTED  no V2.1 reach within V2_TOL_M -> the NWM has no reach here
    #   CROSSES_NWM a V2.1 reach is here AND it is the assigned COMID
    #   WRONG_COMID a V2.1 reach is here but a different COMID was assigned
    no_v2 = (d["v2_near_m"].isna() | (d["v2_near_m"] > V2_TOL_M)).to_numpy(dtype=bool)
    # Comparing two nullable Int64 columns yields a masked boolean, which
    # np.select rejects; NA means "no nearest COMID", i.e. not a match.
    right = (~no_v2) & (d["v2_near_comid"].astype("Int64")
                        .eq(d["comid"].astype("Int64"))
                        .fillna(False).to_numpy(dtype=bool))
    d["status"] = np.select(
        [hr_rc.isna().to_numpy(dtype=bool), no_v2, right],
        ["NO_HR_NEAR", "NOT_ROUTED", "CROSSES_NWM"],
        default="WRONG_COMID")

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(d)
    log.info("=" * 70)
    log.info("DO BRIDGES CROSS THE NWM REACH THEY ARE ASSIGNED TO?")
    log.info("=" * 70)
    log.info("  %-12s %17s   %17s   %s", "", "reachcode test", "geometric test", "scour")
    for s in ("CROSSES_NWM", "WRONG_COMID", "NOT_ROUTED", "NO_HR_NEAR"):
        a = int((d["status_rc"] == s).sum())
        k = int((d["status"] == s).sum())
        sc = int(d.loc[d["status"] == s, "scour_critical"].astype(bool).sum()) \
            if "scour_critical" in d.columns else 0
        log.info("  %-12s %7d (%5.1f%%)   %7d (%5.1f%%)   %5d",
                 s, a, 100.0 * a / n, k, 100.0 * k / n, sc)
    log.info("  (the split and the shapefiles use the GEOMETRIC test; STATUS_RC "
             "carries the reachcode result)")
    ok = d["status"] == "CROSSES_NWM"
    log.info("")
    log.info("  covered by NWM     : %6d  (%.1f%%)", int(ok.sum()), 100.0 * ok.mean())
    log.info("  NOT covered        : %6d  (%.1f%%)", int((~ok).sum()), 100.0 * (~ok).mean())
    log.info("")
    log.info("  perpendicular distance to the HR stream the bridge is on:")
    for s in ("CROSSES_NWM", "WRONG_COMID", "NOT_ROUTED"):
        v = d.loc[d["status"] == s, "hr_dist_m"].dropna()
        if len(v):
            log.info("    %-12s p50 %6.1f m   p90 %7.1f m", s, v.median(), v.quantile(.9))
    nr = d[d["status"] == "NOT_ROUTED"]
    if len(nr):
        log.info("")
        log.info("  NOT_ROUTED: the streams these bridges are actually on")
        named = nr["gnis_name"].fillna("(unnamed)").value_counts().head(8)
        for k_, c_ in named.items():
            log.info("    %-28s %4d bridges", k_, c_)
        log.info("    median HR drainage area %.1f km2 vs assigned reach's NWM gate",
                 nr["totdasqkm"].median())

    # ── write ────────────────────────────────────────────────────────────────
    cols = {"bridge_id": "BRIDGE_ID", "comid": "COMID", "v2_reach": "V2_REACH",
            "v2_name": "V2_NAME", "v2_dist_m": "V2_DIST_M", "nhdplusid": "HR_ID",
            "reachcode": "HR_REACH", "gnis_name": "HR_NAME", "ftype": "HR_FTYPE",
            "hr_dist_m": "HR_DIST_M", "totdasqkm": "HR_DA_KM2",
            "streamorde": "HR_ORDER",
            "v2_near_comid": "V2_NEAR_ID", "v2_near_m": "V2_NEAR_M",
            "status": "STATUS", "status_rc": "STATUS_RC",
            "scour_critical": "SCOUR", "Q50_cfs": "Q50_CFS", "lat": "LAT", "lon": "LON"}
    keep = [c for c in cols if c in d.columns]
    out = gpd.GeoDataFrame(d[keep].rename(columns=cols),
                           geometry=d.geometry, crs=4326)
    if "SCOUR" in out.columns:
        out["SCOUR"] = out["SCOUR"].astype(bool).astype(int)
    out["HR_ID"] = out["HR_ID"].astype("Float64").astype(str)   # 14-digit, keep exact

    a = out[out["STATUS"] == "CROSSES_NWM"]
    b = out[out["STATUS"] != "CROSSES_NWM"]
    a.to_file(out_dir / "bridges_crossing_nwm.shp")
    b.to_file(out_dir / "bridges_not_crossing_nwm.shp")
    out.drop(columns="geometry").to_csv(out_dir / "bridge_stream_crossing.csv", index=False)
    out.drop(columns="geometry").to_parquet(out_dir / "bridge_stream_crossing.parquet", index=False)
    log.info("")
    log.info("wrote %s (%d) and %s (%d), plus the full table as .csv/.parquet",
             "bridges_crossing_nwm.shp", len(a), "bridges_not_crossing_nwm.shp", len(b))


if __name__ == "__main__":
    main()
