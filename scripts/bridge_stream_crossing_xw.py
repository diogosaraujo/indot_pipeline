"""Re-test bridge_stream_crossing_check.py with the USGS Mainstems HR<->V2 crosswalk.

The geometric rule (61 m, then ReachCode, then GNIS name) fails where V2.1 draws
the same stream far from the real channel:

  34-00027 sits on William Vogus Ditch. V2.1 COMID 20097961 is that ditch,
  digitised at 1:100k 61.16 m away, unnamed, and numbered differently from HR,
  so all three steps miss it.

USGS Mainstems v3.1 (Blodgett 2026, doi:10.5066/P13LNDDQ) links NHDPlus HR
flowlines to V2.1 COMIDs by tracing both networks headwater to outlet, not by
geometry or name. The bridge -> HR snap is already reliable (median ~2 m), so
the question becomes: which COMID, if any, does the crosswalk give the bridge's
HR flowline?

The test (per bridge, starting from its nearest HR flowline)
────────
  A  HR flowline is in the reach-level crosswalk (nhdplushr_to_nhdpv2.gpkg)
       matched COMID within SEARCH_M of the bridge  -> CROSSING      XW_REACH
       matched COMID farther away                    -> NOT_CROSSING  XW_FAR
         (HR runs on past the top of the V2.1 line: matched to the V2.1
          inlet, measure 100, with an offset of 1-2 km)
  B  HR flowline is NOT in the crosswalk
       every V2.1 mainstem within SEARCH_M IS matched -> NOT_CROSSING XW_UNMATCHED
         (the network was matched here and this stream is not part of it;
          also covers "no V2.1 reach within SEARCH_M")
       some V2.1 mainstem within SEARCH_M is unmatched -> current rule FALLBACK
         (the crosswalk has no information about it)

Drainage area is NOT used to decide; HR and V2.1 DA are reported as context.
16% of the V2.1 mainstems near Indiana bridges have no HR match at all, which
is why FALLBACK keeps the geometric rule for those bridges.

Data (git-ignored, in tmp/mainstems_v3.1/), from the ScienceBase items:
    nhdplushr_to_nhdpv2.gpkg   65cbc0b3d34ef4b119cb37e9 (Mainstems v3.1)
    nhdplushr.csv              65cbc0b3d34ef4b119cb37e9
    enhd_nhdplusatts.parquet   65cbbb98d34ef4b119cb37c9 (lp_mainstem_v3 == levelpathi)
Download with https://www.sciencebase.gov/catalog/file/get/<item>?name=<file>
(the /manager/ links return the web app, not the file).

Outputs (results/)
    bridge_stream_crossing_xw.{csv,parquet}  every bridge, old + new status
    bridges_xw_changed.shp                   STATUS changed, for visual review

Usage
    python scripts/bridge_stream_crossing_check.py   # first
    python scripts/bridge_stream_crossing_xw.py
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
import pyarrow.csv as pacsv
import pyogrio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crossing_xw")

REPO = pathlib.Path(__file__).resolve().parents[1]
MS_DIR = REPO / "tmp" / "mainstems_v3.1"
V2_SHP = REPO / "results" / "indiana_nwm_comids.shp"
IN_PARQUET = REPO / "results" / "bridge_stream_crossing.parquet"
OUT_DIR = REPO / "results"

WORK_CRS = 26916          # UTM 16N, as in bridge_stream_crossing_check.py
SEARCH_M = 1000.0         # daraujo, 2026-10-21: "keep using 1 km for now"
IN_BBOX = (-88.3, 37.6, -84.6, 41.9)   # Indiana + margin, lon/lat


def load_reach_xw(gpkg: pathlib.Path) -> pd.DataFrame:
    """HR flowline start/end points -> V2.1 COMID, clipped to Indiana."""
    m = pyogrio.read_dataframe(gpkg, layer="matches", bbox=IN_BBOX, read_geometry=False)
    # nhdplushr_id is stored as TEXT in the gpkg
    m["nhdplushr_id"] = m["nhdplushr_id"].astype("int64")
    m["nhdplusv2_comid"] = m["nhdplusv2_comid"].astype("int64")
    log.info("reach crosswalk: %d HR endpoints, %d HR flowlines",
             len(m), m["nhdplushr_id"].nunique())
    return m


def matched_mainstems(csv: pathlib.Path) -> set:
    t = pacsv.read_csv(csv, convert_options=pacsv.ConvertOptions(
        include_columns=["lp_mainstem_v3"]))
    s = set(pd.Series(t.column(0).to_numpy(zero_copy_only=False)).dropna().astype("int64"))
    log.info("mainstems with an HR match: %d", len(s))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mainstems-dir", default=str(MS_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    ms = pathlib.Path(args.mainstems_dir)
    out_dir = pathlib.Path(args.out_dir)

    d = pd.read_parquet(IN_PARQUET)
    d["hr_id"] = pd.to_numeric(d["HR_ID"], errors="coerce").round().astype("Int64")
    br = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["LON"], d["LAT"]),
                          crs=4326).to_crs(WORK_CRS)
    log.info("bridges: %d", len(br))

    xw = load_reach_xw(ms / "nhdplushr_to_nhdpv2.gpkg")
    lp_ok = matched_mainstems(ms / "nhdplushr.csv")
    enhd = pd.read_parquet(ms / "enhd_nhdplusatts.parquet",
                           columns=["comid", "levelpathi", "totdasqkm"])
    lp_of = dict(zip(enhd["comid"].astype("int64"), enhd["levelpathi"]))
    da_of = dict(zip(enhd["comid"].astype("int64"), enhd["totdasqkm"]))

    v2 = gpd.read_file(V2_SHP, columns=["COMID"]).to_crs(WORK_CRS)
    v2["COMID"] = pd.to_numeric(v2["COMID"], errors="coerce").astype("int64")
    geom_by_comid = dict(zip(v2["COMID"], v2.geometry))

    # ── A: HR flowline is in the reach-level crosswalk ───────────────────────
    cand = xw[xw["nhdplushr_id"].isin(set(br["hr_id"].dropna()))]
    by_hr = cand.groupby("nhdplushr_id")
    xw_comids = by_hr["nhdplusv2_comid"].apply(lambda s: sorted(set(s))).to_dict()
    xw_off = by_hr["nhdplusv2_offset"].min().to_dict()
    xw_meas = by_hr["nhdplusv2_comid_measure"].max().to_dict()

    best_c, best_d = [], []
    for h, p in zip(br["hr_id"], br.geometry):
        cs = xw_comids.get(h) if pd.notna(h) else None
        if not cs:
            best_c.append(pd.NA); best_d.append(np.nan); continue
        ds = [(geom_by_comid[c].distance(p), c) for c in cs if c in geom_by_comid]
        if not ds:                       # matched COMID absent from our V2.1 layer
            best_c.append(cs[0]); best_d.append(np.nan); continue
        dist, c = min(ds)
        best_c.append(c); best_d.append(dist)
    br["xw_comid"] = pd.array(best_c, dtype="Int64")
    br["xw_dist_m"] = best_d
    br["xw_off_m"] = br["hr_id"].map(xw_off)
    br["xw_meas"] = br["hr_id"].map(xw_meas)
    in_xw = br["hr_id"].isin(xw_comids.keys()).to_numpy(dtype=bool)

    # ── B: V2.1 mainstems within SEARCH_M, and whether all are matched ───────
    near = gpd.sjoin(br[["BRIDGE_ID", "geometry"]].assign(geometry=br.buffer(SEARCH_M)),
                     v2[["COMID", "geometry"]], how="left", predicate="intersects")
    near["lp"] = near["COMID"].map(lp_of)
    # a COMID missing from enhd (NWM-modified) counts as unmatched -> no information
    near["lp_matched"] = near["lp"].isin(lp_ok)
    g = near.dropna(subset=["COMID"]).groupby("BRIDGE_ID")
    n_v2 = g.size()
    all_matched = g["lp_matched"].all()
    br["v2_within_n"] = br["BRIDGE_ID"].map(n_v2).fillna(0).astype(int)
    br["v2_all_matched"] = br["BRIDGE_ID"].map(all_matched).fillna(True).astype(bool)

    # ── THE TEST ─────────────────────────────────────────────────────────────
    a_near = in_xw & (br["xw_dist_m"].fillna(np.inf) <= SEARCH_M).to_numpy(dtype=bool)
    a_far = in_xw & ~a_near
    b_unm = (~in_xw) & br["v2_all_matched"].to_numpy(dtype=bool)
    br["rule_xw"] = np.select([a_near, a_far, b_unm],
                              ["XW_REACH", "XW_FAR", "XW_UNMATCHED"], default="FALLBACK")
    br["status_xw"] = np.select([a_near, a_far | b_unm], ["CROSSING", "NOT_CROSSING"],
                                default=br["STATUS"].to_numpy())
    br["xw_da_km2"] = br["xw_comid"].map(lambda c: da_of.get(c) if pd.notna(c) else np.nan)
    br["changed"] = br["status_xw"] != br["STATUS"]

    # ── summary ──────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("CROSSWALK RULE (search %.0f m) vs CURRENT GEOMETRIC RULE", SEARCH_M)
    log.info("=" * 70)
    t = pd.crosstab([br["rule_xw"], br["status_xw"]], br["STATUS"], margins=True)
    for line in t.to_string().splitlines():
        log.info("  %s", line)
    for st in ("CROSSING", "NOT_CROSSING"):
        k0 = int((br["STATUS"] == st).sum()); k1 = int((br["status_xw"] == st).sum())
        log.info("  %-13s current %6d  ->  crosswalk %6d", st, k0, k1)
    ch = br[br["changed"]]
    log.info("  changed: %d  (%d scour-critical)", len(ch), int(ch["SCOUR"].sum()))
    for (o, n), k in ch.groupby(["STATUS", "status_xw"]).size().items():
        log.info("    %-12s -> %-12s %5d", o, n, k)
    cc = br[(br["status_xw"] == "CROSSING") & br["xw_comid"].notna()
            & (br["xw_comid"] != br["V2_NEAR_ID"])]
    log.info("  crossing with a crosswalk COMID other than the nearest V2.1 reach: %d", len(cc))

    # ── write ────────────────────────────────────────────────────────────────
    cols = ["BRIDGE_ID", "COMID", "STATUS", "RULE", "status_xw", "rule_xw", "changed",
            "HR_ID", "HR_NAME", "HR_FTYPE", "HR_DA_KM2", "V2_NEAR_ID", "V2_NEAR_M",
            "V2_NEAR_NM", "xw_comid", "xw_dist_m", "xw_off_m", "xw_meas", "xw_da_km2",
            "v2_within_n", "v2_all_matched", "SCOUR", "Q50_CFS", "LAT", "LON"]
    out = br[cols].rename(columns={
        "status_xw": "STATUS_XW", "rule_xw": "RULE_XW", "changed": "CHANGED",
        "xw_comid": "XW_COMID", "xw_dist_m": "XW_DIST_M", "xw_off_m": "XW_OFF_M",
        "xw_meas": "XW_MEAS", "xw_da_km2": "XW_DA_KM2", "v2_within_n": "V2_1KM_N",
        "v2_all_matched": "V2_ALL_MS"})
    out.to_csv(out_dir / "bridge_stream_crossing_xw.csv", index=False)
    out.to_parquet(out_dir / "bridge_stream_crossing_xw.parquet", index=False)
    shp = gpd.GeoDataFrame(out[out["CHANGED"]].drop(columns="CHANGED"),
                           geometry=br.loc[br["changed"]].geometry.to_crs(4326).values,
                           crs=4326)
    shp["V2_ALL_MS"] = shp["V2_ALL_MS"].astype(int)
    shp.to_file(out_dir / "bridges_xw_changed.shp")
    log.info("wrote bridge_stream_crossing_xw.{csv,parquet} (%d) and bridges_xw_changed.shp (%d)",
             len(out), len(shp))


if __name__ == "__main__":
    main()
