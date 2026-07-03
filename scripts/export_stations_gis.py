"""export_stations_gis.py

Export the point locations of the stations that survived the funnel — the set
"Evaluated in 08c" (the final band of the station_funnel figure) — as GeoJSON,
to match the other station layers this pipeline writes (see 01_get_indiana_stations).

Authoritative station set:
    The distinct site_no in the 08c output
    (analysis/event_confusion_matrix_tc.parquet).  That is exactly what the
    station_funnel figure counts as "Evaluated in 08c", so the count is whatever
    the current run produced — nothing is hardcoded here.

Coordinates come from the station inventory
    (stations/indiana_streamflow_sites.parquet): dec_lat_va / dec_long_va,
    decimal degrees in EPSG:4326 (WGS84).  GeoJSON opens directly in ArcGIS Pro.

Writes:
    ./exports/stations_08c.geojson                       (local, for ArcGIS)
    s3://<bucket>/<prefix>stations/stations_08c.geojson  (matches the other layers)

Usage:
    python scripts/export_stations_gis.py
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import Point

from utils import load_config, s3_client, write_bytes_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("export_gis")

TC_KEY   = "analysis/event_confusion_matrix_tc.parquet"          # 08c output
INV_KEY  = "stations/indiana_streamflow_sites.parquet"
OUT_KEY  = "stations/stations_08c.geojson"                       # matches other layers
LOCAL_GJ = Path("exports/stations_08c.geojson")

# Inventory attributes to carry onto each point.
ATTR_COLS = ["site_no", "station_nm", "dec_lat_va", "dec_long_va",
             "drain_area_va", "huc_cd", "begin_date", "end_date"]


def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    # ── 1. Final filtered set: distinct stations evaluated in 08c ────────────
    used = _read_parquet_s3(bucket, f"{prefix}{TC_KEY}", columns=["site_no"])
    site_ids = sorted(set(used["site_no"].astype(str)))
    log.info("Stations evaluated in 08c (final funnel count): %d", len(site_ids))

    # ── 2. Attach coordinates + attributes from the inventory ────────────────
    inv = _read_parquet_s3(bucket, f"{prefix}{INV_KEY}")
    inv["site_no"] = inv["site_no"].astype(str)
    cols = [c for c in ATTR_COLS if c in inv.columns]
    sub = inv[inv["site_no"].isin(site_ids)][cols].copy()

    missing = sorted(set(site_ids) - set(sub["site_no"]))
    if missing:
        log.warning("%d station(s) had no inventory coordinates and are dropped: %s",
                    len(missing), ", ".join(missing))

    sub["dec_lat_va"]  = pd.to_numeric(sub["dec_lat_va"], errors="coerce")
    sub["dec_long_va"] = pd.to_numeric(sub["dec_long_va"], errors="coerce")
    sub = sub.dropna(subset=["dec_lat_va", "dec_long_va"]).sort_values("site_no")
    log.info("Points with valid coordinates: %d", len(sub))

    # Dates → strings for clean GeoJSON serialization.
    for col in ("begin_date", "end_date"):
        if col in sub.columns:
            sub[col] = pd.to_datetime(sub[col], errors="coerce").dt.strftime("%Y-%m-%d")

    gdf = gpd.GeoDataFrame(
        sub,
        geometry=[Point(xy) for xy in zip(sub["dec_long_va"], sub["dec_lat_va"])],
        crs="EPSG:4326",
    )

    # ── 3. Write GeoJSON — local (for ArcGIS) + S3 (matches other layers) ────
    gj_bytes = gdf.to_json().encode()

    LOCAL_GJ.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_GJ.write_bytes(gj_bytes)
    log.info("Wrote local GeoJSON: %s", LOCAL_GJ.resolve())

    write_bytes_to_s3(gj_bytes, bucket, f"{prefix}{OUT_KEY}")
    log.info("Wrote GeoJSON: s3://%s/%s%s", bucket, prefix, OUT_KEY)

    log.info("Done. %d stations exported.", len(gdf))


if __name__ == "__main__":
    main()
