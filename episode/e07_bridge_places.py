"""e07 — county, nearest city, and river name for every bridge.

The report tables carry location context, but none of the three fields exists in
the bridge inventory (checked: 110 columns, no county/place/feature-intersected),
so each is derived here once:

  county  point-in-polygon against Census TIGER counties
  city    NEAREST Census place, with the distance carried alongside — bridges are
          usually outside any city limit, so "contained by" would be null for most
          of them and a bare nearest-name would imply a precision it lacks
  river   NHDPlus GNIS_NAME for the bridge's COMID, from the EPA WATERS service
          (NLDI's /comid/ endpoint returns identifiers only, no name)

Unnamed NHDPlus reaches are common for small tributaries; they stay blank rather
than being filled with the downstream river's name, which would be wrong.

Writes  episode/bridge_places.parquet
            bridge_id, county, city, city_mi, river

Usage:
    python episode/e07_bridge_places.py
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import pandas as pd

from common import bucket, ep_key, load_config
from monitor_common.s3io import write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e07")

COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
PLACES = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_18_place_500k.zip"
WATERS = ("https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/"
          "NHDSnapshot_NP21/MapServer/0/query")
CRS_M = 26916                      # NAD83 / UTM 16N — metres, for distances


def river_names(comids: list[int], batch: int = 100, pause: float = 0.2) -> dict:
    """COMID -> GNIS_NAME, batched. Unknown/unnamed reaches are simply absent."""
    out: dict[int, str] = {}
    for i in range(0, len(comids), batch):
        chunk = comids[i:i + batch]
        q = urllib.parse.urlencode({
            "where": f"COMID IN ({','.join(map(str, chunk))})",
            "outFields": "COMID,GNIS_NAME", "returnGeometry": "false", "f": "json"})
        try:
            with urllib.request.urlopen(f"{WATERS}?{q}", timeout=60) as r:
                feats = json.load(r).get("features", [])
        except Exception as e:  # noqa: BLE001
            log.warning("WATERS batch %d failed (%s) — those reaches stay unnamed",
                        i // batch, type(e).__name__)
            continue
        for f in feats:
            a = f.get("attributes", {})
            nm = a.get("GNIS_NAME")
            if nm and str(nm).strip():
                out[int(a["COMID"])] = str(nm).strip()
        if (i // batch) % 10 == 0:
            log.info("  rivers %d/%d comids, %d named so far",
                     min(i + batch, len(comids)), len(comids), len(out))
        time.sleep(pause)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    g = gpd.GeoDataFrame(cfg[["bridge_id", "lat", "lon", "comid"]].copy(),
                         geometry=gpd.points_from_xy(cfg["lon"], cfg["lat"]),
                         crs=4326)

    log.info("County lookup (%d bridges)...", len(g))
    cty = gpd.read_file(COUNTIES)
    cty = cty[cty["STATEFP"] == "18"][["NAME", "geometry"]].rename(columns={"NAME": "county"})
    g = gpd.sjoin(g, cty.to_crs(4326), how="left", predicate="within").drop(columns="index_right")
    log.info("  matched %d/%d", int(g["county"].notna().sum()), len(g))

    log.info("Nearest-city lookup...")
    plc = gpd.read_file(PLACES)[["NAME", "geometry"]].rename(columns={"NAME": "city"})
    gm, pm = g.to_crs(CRS_M), plc.to_crs(CRS_M)
    near = gpd.sjoin_nearest(gm, pm, how="left", distance_col="_m").drop(columns="index_right")
    near = near.drop_duplicates("bridge_id")
    g = g.merge(near[["bridge_id", "city", "_m"]], on="bridge_id", how="left")
    g["city_mi"] = (g["_m"] / 1609.34).round(1)
    g = g.drop(columns="_m")

    comids = sorted(int(c) for c in cfg["comid"].dropna().unique())
    log.info("River names for %d COMIDs...", len(comids))
    names = river_names(comids, args.batch)
    g["river"] = pd.to_numeric(g["comid"], errors="coerce").map(names)
    log.info("  %d/%d COMIDs named -> %d/%d bridges have a river",
             len(names), len(comids), int(g["river"].notna().sum()), len(g))

    out = pd.DataFrame(g.drop(columns="geometry"))[
        ["bridge_id", "county", "city", "city_mi", "river"]]
    write_parquet(out, bucket(), ep_key("bridge_places.parquet"))
    log.info("Wrote episode/bridge_places.parquet")
    log.info("\n%s", out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
