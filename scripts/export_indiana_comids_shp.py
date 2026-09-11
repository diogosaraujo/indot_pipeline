#!/usr/bin/env python3
"""Export every NWM/NHDPlus-V2 reach in Indiana as an ESRI shapefile.

Built for opening in ArcGIS and checking, by eye, which bridges are snapped to a
reach they do not cross. The monitor's own flowlines.parquet holds only the ~6.9k
reaches a bridge was assigned to, which is exactly the set that cannot reveal a
bad assignment — so the geometry comes from the full network instead.

Source is the EPA WATERS NHDPlus_NP21 snapshot (NHDPlus V2.1), the vintage whose
COMIDs the National Water Model routes on. Reaches are clipped to the Indiana
county boundary rather than a bounding box, and joined to the monitor's own
tables so the layer carries what an investigation needs:

    IN_NWM     is the reach in the NWM domain (from a live channel_rt feature_id list)
    N_BRIDGES  how many monitored bridges are assigned to it
    Q10/50/100 the retro-LP3 quantiles the flow trigger fires against
    UNIT_Q100  Q100 per square mile — the number that exposes a broken fit
    FLAG_LOWQ  1 where UNIT_Q100 < 5 cfs/mi2 (implausible; see COMID 13437963)

Writes results/indiana_nwm_comids.{shp,shx,dbf,prj,cpg}

Usage:
    python scripts/export_indiana_comids_shp.py
    python scripts/export_indiana_comids_shp.py --no-nwm-check   # skip the 12 MB download
"""
from __future__ import annotations

import argparse
import io
import pathlib
import time

import boto3
import numpy as np
import pandas as pd
import requests
import shapefile
from botocore import UNSIGNED
from botocore.client import Config
from matplotlib.path import Path

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
SERVICE = ("https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/"
           "NHDSnapshot_NP21/MapServer/0/query")
IN_BBOX = (-88.20, 37.70, -84.70, 41.90)

# The service returns FTYPE as a bare NHD code. Nobody filters on "460" in
# ArcGIS, so ship the label alongside it.
FTYPE_NAME = {
    "334": "Connector", "336": "CanalDitch", "420": "UndergroundConduit",
    "428": "Pipeline", "460": "StreamRiver", "558": "ArtificialPath",
    "566": "Coastline", "334.0": "Connector",
}
PAGE = 1000

# WGS84, so ArcGIS does not ask. Without a .prj the layer loads with an unknown
# CRS and silently fails to line up with anything else.
PRJ_WGS84 = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def s3_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}{key}")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def fetch_flowlines(bbox, timeout=120):
    """All flowlines intersecting bbox, paginated. -> list of dicts."""
    base = {
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "COMID,GNIS_NAME,LENGTHKM,FTYPE,REACHCODE",
        "returnGeometry": "true", "f": "geojson",
    }
    n = requests.get(SERVICE, params={**base, "returnCountOnly": "true", "f": "json"},
                     timeout=timeout).json().get("count", 0)
    print(f"  {n:,} flowlines intersect the bounding box")

    out, offset = [], 0
    while offset < n:
        for attempt in range(4):
            try:
                r = requests.get(SERVICE, params={**base, "resultOffset": offset,
                                                  "resultRecordCount": PAGE},
                                 timeout=timeout)
                r.raise_for_status()
                feats = r.json().get("features", [])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"    retry {attempt+1} at offset {offset}: {e}")
                time.sleep(2 + 3 * attempt)
        if not feats:
            break
        out.extend(feats)
        offset += PAGE
        print(f"    {len(out):,} / {n:,}", end="\r")
    print(f"    fetched {len(out):,} features            ")
    return out


def indiana_mask(counties: pd.DataFrame):
    """Point-in-Indiana test built from the county rings the monitor already ships."""
    paths = [Path(np.column_stack([g.lon.to_numpy(), g.lat.to_numpy()]))
             for _, g in counties.groupby("part_id") if len(g) > 3]
    print(f"  Indiana boundary: {len(paths)} county rings")

    def inside(pts: np.ndarray) -> bool:
        for p in paths:
            if p.contains_points(pts).any():
                return True
        return False
    return inside


def nwm_feature_ids(timeout=180) -> set[int]:
    """feature_id set from one live NWM channel_rt file (anonymous, ~12 MB)."""
    import h5py
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    day = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    key = (f"nwm.{day:%Y%m%d}/analysis_assim/"
           f"nwm.t00z.analysis_assim.channel_rt.tm00.conus.nc")
    print(f"  NWM domain from s3://noaa-nwm-pds/{key}")
    raw = s3.get_object(Bucket="noaa-nwm-pds", Key=key)["Body"].read()
    with h5py.File(io.BytesIO(raw), "r") as h:
        ids = np.asarray(h["feature_id"][:]).astype("int64")
    print(f"    {len(ids):,} reaches in the NWM domain (CONUS)")
    return set(ids.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/indiana_nwm_comids")
    ap.add_argument("--no-nwm-check", action="store_true")
    args = ap.parse_args()

    print("Loading monitor tables ...")
    cfg = s3_parquet("monitor/bridge_monitor_config.parquet")
    lp3 = s3_parquet("monitor/precompute/bridge_comid_lp3.parquet").set_index("comid")
    area = s3_parquet("monitor/precompute/bridge_watershed_area.parquet").set_index("comid")["area_mi2"]
    counties = s3_parquet("monitor/assets/in_counties.parquet")
    nb = pd.to_numeric(cfg["comid"], errors="coerce").value_counts()

    print("Fetching NHDPlus V2.1 flowlines ...")
    feats = fetch_flowlines(IN_BBOX)

    print("Clipping to Indiana ...")
    inside = indiana_mask(counties)
    nwm = set() if args.no_nwm_check else nwm_feature_ids()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = shapefile.Writer(str(out), shapeType=shapefile.POLYLINE)
    w.field("COMID", "N", 10, 0)
    w.field("GNIS_NAME", "C", 65)
    w.field("LENGTHKM", "N", 12, 4)
    w.field("FTYPE", "C", 8)
    w.field("FTYPE_NAME", "C", 24)
    w.field("REACHCODE", "C", 14)
    w.field("IN_NWM", "N", 1, 0)
    w.field("N_BRIDGES", "N", 5, 0)
    w.field("Q10_CFS", "N", 16, 3)
    w.field("Q50_CFS", "N", 16, 3)
    w.field("Q100_CFS", "N", 16, 3)
    w.field("AREA_MI2", "N", 16, 3)
    w.field("UNIT_Q100", "N", 14, 4)
    w.field("FLAG_LOWQ", "N", 1, 0)

    kept = dropped = 0
    n_nwm = n_br = n_flag = 0
    for f in feats:
        geom = f.get("geometry") or {}
        gt = geom.get("type")
        if gt == "LineString":
            parts = [geom["coordinates"]]
        elif gt == "MultiLineString":
            parts = geom["coordinates"]
        else:
            dropped += 1
            continue
        parts = [p for p in parts if len(p) >= 2]
        if not parts:
            dropped += 1
            continue
        pts = np.asarray([c[:2] for p in parts for c in p], float)
        if not inside(pts):
            dropped += 1
            continue

        a = f.get("properties", {})
        comid = int(a.get("COMID") or 0)
        q = lp3.loc[comid] if comid in lp3.index else None
        ar = float(area.get(comid, np.nan))
        q100 = float(q["Q100_cfs"]) if q is not None and pd.notna(q["Q100_cfs"]) else np.nan
        unit = q100 / ar if (ar == ar and ar > 0 and q100 == q100) else np.nan
        flag = int(unit == unit and unit < 5.0)
        innwm = int(comid in nwm) if nwm else 0
        nbr = int(nb.get(comid, 0))
        n_nwm += innwm; n_br += (nbr > 0); n_flag += flag

        # The service returns FTYPE as a coded integer and REACHCODE as a number,
        # so coerce rather than assume text.
        def txt(key, n):
            v = a.get(key)
            return "" if v is None else str(v)[:n]

        w.line([[list(map(float, c[:2])) for c in p] for p in parts])
        w.record(comid, txt("GNIS_NAME", 65),
                 float(a.get("LENGTHKM") or 0), txt("FTYPE", 8),
                 FTYPE_NAME.get(txt("FTYPE", 8), "other"),
                 txt("REACHCODE", 14), innwm, nbr,
                 float(q["Q10_cfs"]) if q is not None else 0.0,
                 float(q["Q50_cfs"]) if q is not None else 0.0,
                 0.0 if q100 != q100 else q100,
                 0.0 if ar != ar else ar,
                 0.0 if unit != unit else unit, flag)
        kept += 1
    w.close()

    out.with_suffix(".prj").write_text(PRJ_WGS84, encoding="utf-8")
    out.with_suffix(".cpg").write_text("UTF-8", encoding="utf-8")

    size = sum(p.stat().st_size for p in out.parent.glob(out.name + ".*"))
    print(f"\nWrote {out}.shp  ({kept:,} reaches, {size/1e6:.1f} MB total)")
    print(f"  dropped outside Indiana / no geometry : {dropped:,}")
    if nwm:
        print(f"  IN_NWM = 1                            : {n_nwm:,}")
    print(f"  carrying at least one monitored bridge : {n_br:,}")
    print(f"  FLAG_LOWQ = 1 (unit Q100 < 5 cfs/mi2)  : {n_flag:,}")
    print("  files:", ", ".join(sorted(p.name for p in out.parent.glob(out.name + ".*"))))


if __name__ == "__main__":
    main()
