"""nwm_comid_geojson.py

Two GeoJSONs for the NWM COMIDs matched to our USGS gauges (10_download_nwm):
  1. nwm_reaches.geojson        NHDPlus flowline POLYLINES for each used COMID
  2. nwm_comid_outlets.geojson  the reach OUTLET points (downstream end of each
                                flowline — where NWM reports the reach streamflow)

Geometry is fetched from NLDI (api.water.usgs.gov/nldi/linked-data/comid/{comid}),
the same service the download used.  NHDPlusV2 flowlines are digitised in the flow
direction, so the LAST vertex is the downstream outlet.

Also prints the NWM coverage funnel (inventory -> COMID resolved -> has NWM data).

Writes:
    s3://<bucket>/<prefix>analysis/nwm/nwm_reaches.geojson
    s3://<bucket>/<prefix>analysis/nwm/nwm_comid_outlets.geojson

Usage:
    python scripts/nwm_comid_geojson.py
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import load_config, s3_client

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
INV_KEY   = "stations/indiana_streamflow_sites.parquet"
CL_KEY    = "nwm/comid_locations.parquet"
RETRO_KEY = "nwm/retrospective.parquet"
AA_KEY    = "nwm/analysis_assim.parquet"
TC_KEY    = "analysis/event_confusion_matrix_tc.parquet"   # 08c analysis stations
REACH_STEM  = "analysis/nwm/nwm_reaches"
OUTLET_STEM = "analysis/nwm/nwm_comid_outlets"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nwm_geojson")


def _pq(bucket, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def _unique_comids(bucket, prefix, key):
    try:
        return set(int(c) for c in _pq(bucket, f"{prefix}{key}", ["comid"])["comid"].unique())
    except Exception as e:                                   # noqa: BLE001
        log.warning("could not read %s (%s)", key, e)
        return set()


def fetch_flowline(comid: int, timeout: int):
    """Return (geom_type, [line, ...]) from NLDI, or None."""
    try:
        r = requests.get(f"{NLDI_BASE}/comid/{comid}", timeout=timeout)
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
        if not feats:
            return None
        geom = feats[0].get("geometry", {})
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "LineString":
            return gtype, [coords]
        if gtype == "MultiLineString":
            return gtype, coords
        return None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-only", action="store_true",
                    help="restrict to the 08c analysis stations; write *_analysis.geojson")
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    timeout = cfg["streamstats"]["request_timeout_sec"]
    workers = cfg["execution"]["max_workers_io"]

    n_inv = len(_pq(bucket, f"{prefix}{INV_KEY}", ["site_no"]))
    cl = _pq(bucket, f"{prefix}{CL_KEY}")
    cl["site_no"] = cl["site_no"].astype(str)
    cl["comid"] = cl["comid"].astype(int)

    retro_c = _unique_comids(bucket, prefix, RETRO_KEY)
    aa_c    = _unique_comids(bucket, prefix, AA_KEY)

    suffix = ""
    if args.analysis_only:
        asites = set(_pq(bucket, f"{prefix}{TC_KEY}", ["site_no"])["site_no"].astype(str))
        resolved = set(cl["site_no"])
        site_comid = dict(zip(cl["site_no"], cl["comid"]))
        res_a     = asites & resolved
        with_retro = {s for s in res_a if site_comid[s] in retro_c}
        with_aa    = {s for s in res_a if site_comid[s] in aa_c}
        no_comid   = sorted(asites - resolved)
        no_retro   = sorted(s for s in res_a if site_comid[s] not in retro_c)

        print("\n=== NWM coverage of the 08c analysis stations ===")
        print(f"  analysis stations (08c)          : {len(asites)}")
        print(f"  with a resolved COMID            : {len(res_a)}")
        print(f"  with NWM retrospective data      : {len(with_retro)}")
        print(f"  with NWM analysis_assim data     : {len(with_aa)}")
        complete = len(with_retro) == len(asites)
        print(f"  COMPLETE retrospective coverage? : {'YES' if complete else 'NO'}")
        if no_comid:
            print(f"  no COMID ({len(no_comid)}): {no_comid}")
        if no_retro:
            print(f"  COMID but no retro ({len(no_retro)}): {no_retro}")
        print()

        cl = cl[cl["site_no"].isin(asites)]
        suffix = "_analysis"

    reach_out  = f"{REACH_STEM}{suffix}.geojson"
    outlet_out = f"{OUTLET_STEM}{suffix}.geojson"

    # one record per COMID (a COMID may serve >1 gauge)
    by_comid = cl.groupby("comid").agg(
        site_no=("site_no", lambda s: ",".join(sorted(set(s)))),
        n_sites=("site_no", "nunique"),
        distance_km=("distance_km", "min"),
    ).reset_index()

    log.info("Fetching %d NHDPlus flowlines from NLDI...", len(by_comid))
    geoms: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_flowline, int(c), timeout): int(c) for c in by_comid["comid"]}
        for i, fut in enumerate(as_completed(futs), 1):
            g = fut.result()
            if g is not None:
                geoms[futs[fut]] = g
            if i % 50 == 0:
                log.info("  %d / %d", i, len(futs))

    reach_feats, outlet_feats = [], []
    for r in by_comid.itertuples(index=False):
        comid = int(r.comid)
        g = geoms.get(comid)
        if g is None:
            continue
        gtype, lines = g
        outlet = lines[-1][-1]                              # downstream end [lon, lat]
        props = {
            "comid": comid, "site_no": r.site_no, "n_sites": int(r.n_sites),
            "distance_km": (round(float(r.distance_km), 4) if pd.notna(r.distance_km) else None),
            "has_retro": comid in retro_c, "has_analysis_assim": comid in aa_c,
        }
        reach_geom = ({"type": "LineString", "coordinates": lines[0]} if gtype == "LineString"
                      else {"type": "MultiLineString", "coordinates": lines})
        reach_feats.append({"type": "Feature", "geometry": reach_geom, "properties": props})
        outlet_feats.append({"type": "Feature",
                             "geometry": {"type": "Point", "coordinates": [outlet[0], outlet[1]]},
                             "properties": props})

    for key, feats in [(reach_out, reach_feats), (outlet_out, outlet_feats)]:
        body = json.dumps({"type": "FeatureCollection", "features": feats}, indent=1).encode()
        s3_client().put_object(Bucket=bucket, Key=f"{prefix}{key}", Body=body,
                               ContentType="application/geo+json")
        log.info("Wrote s3://%s/%s%s (%d features)", bucket, prefix, key, len(feats))

    # ── coverage funnel (for the set being exported) ──
    scope = "08c analysis stations" if suffix else "all inventory"
    print(f"\n=== NWM export funnel ({scope}) ===")
    print(f"  inventory stations              : {n_inv}")
    print(f"  COMIDs exported                  : {cl['site_no'].nunique()} stations, "
          f"{cl['comid'].nunique()} distinct COMIDs")
    print(f"  COMIDs with retrospective data   : {len(retro_c)} (global)")
    print(f"  COMIDs with analysis_assim data  : {len(aa_c)} (global)")
    print(f"  flowline geometry fetched        : {len(geoms)} / {len(by_comid)}")
    print(f"\nDownload:  aws s3 cp s3://{bucket}/{prefix}{reach_out} .")
    print(f"           aws s3 cp s3://{bucket}/{prefix}{outlet_out} .")


if __name__ == "__main__":
    main()
