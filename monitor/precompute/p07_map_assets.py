"""p07 — Indiana county outlines as a plain lon/lat table for the alert map.

The digest PDF draws a statewide map, but the Lambda image has NO geopandas /
shapely / pyproj (see env-lambda.yml) — adding them would roughly double the
image. So the boundary geometry is flattened HERE, once, into a long table of
polygon vertices that matplotlib can draw directly:

    part_id  int    one per ring (county polygons; islands get their own ring)
    lon      float
    lat      float

The alerter groups by part_id and calls ax.plot / Polygon on each ring. Map
scale is handled with an equirectangular aspect correction (1/cos(lat)) rather
than a projection, which is accurate enough at one-state extent.

Source: US Census TIGER/Line cartographic boundary, 20 m generalization.
Writes  monitor/assets/in_counties.parquet
"""
from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from common import config
from monitor_common.s3io import write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p07")

TIGER = ("https://www2.census.gov/geo/tiger/GENZ2022/shp/"
         "cb_2022_us_county_20m.zip")
STATEFP = "18"                       # Indiana
OUT = "monitor/assets/in_counties.parquet"


def rings(geom) -> list:
    """Exterior rings of a (Multi)Polygon, as coordinate lists."""
    if isinstance(geom, Polygon):
        return [list(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [list(g.exterior.coords) for g in geom.geoms]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simplify", type=float, default=0.004,
                    help="degrees; 0 disables. 0.004 ~ 400 m, invisible at state scale")
    args = ap.parse_args()

    log.info("Fetching %s", TIGER)
    gdf = gpd.read_file(TIGER)
    ind = gdf[gdf["STATEFP"] == STATEFP].to_crs(4326)
    log.info("Indiana counties: %d", len(ind))

    if args.simplify > 0:
        ind["geometry"] = ind.geometry.simplify(args.simplify, preserve_topology=True)

    rows, pid = [], 0
    for geom in ind.geometry:
        for ring in rings(geom):
            for lon, lat in ring:
                rows.append((pid, lon, lat))
            pid += 1
    df = pd.DataFrame(rows, columns=["part_id", "lon", "lat"])
    log.info("Flattened to %d rings, %d vertices (%.0f KB in memory)",
             pid, len(df), df.memory_usage(deep=True).sum() / 1024)

    b, prefix = config.bucket_prefix()
    write_parquet(df, b, f"{prefix}{OUT}")
    log.info("Wrote s3://%s/%s%s", b, prefix, OUT)


if __name__ == "__main__":
    main()
