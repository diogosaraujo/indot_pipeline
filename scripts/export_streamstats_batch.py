"""export_streamstats_batch.py

Read the station inventory produced by script 01 and write zipped point
shapefiles ready for submission to the USGS StreamStats Batch Processor.

StreamStats limit is 250 points per submission. If the inventory exceeds 250
stations, multiple zip files are produced (one per batch).

Each zip contains the four files required by the uploader:
    .shp  geometry
    .shx  shape index
    .dbf  attributes (site_no ID field)
    .prj  coordinate system

Coordinate system: NAD83 geographic (EPSG:4269), which matches the Indiana
StreamStats stream grid.

Reads:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet

Writes (local):
    streamstats_batches/streamstats_batch_01.zip
    streamstats_batches/streamstats_batch_02.zip  (if > 250 stations)
    ...
"""
from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import Point

from utils import load_config, s3_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("export_streamstats_batch")

BATCH_SIZE = 250


def read_station_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def export_streamstats_batch_shapefiles(df: pd.DataFrame, output_dir: str = "streamstats_batches") -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gdf = gpd.GeoDataFrame(
        df[["site_no"]].copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df["dec_long_va"], df["dec_lat_va"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:4269")

    n_batches = (len(gdf) + BATCH_SIZE - 1) // BATCH_SIZE
    zip_paths: list[str] = []

    for i in range(n_batches):
        batch = gdf.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].copy()
        batch_num = i + 1
        zip_path = output_path / f"streamstats_batch_{batch_num:02d}.zip"

        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / f"stations_batch_{batch_num:02d}.shp"
            batch.to_file(shp_path, driver="ESRI Shapefile")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for ext in (".shp", ".shx", ".dbf", ".prj"):
                    f = shp_path.with_suffix(ext)
                    if f.exists():
                        zf.write(f, f.name)

        log.info(
            "Wrote %s  (%d points, batch %d/%d)",
            zip_path, len(batch), batch_num, n_batches,
        )
        zip_paths.append(str(zip_path))

    return zip_paths


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    sites = read_station_inventory(bucket, prefix)
    log.info("Loaded %d stations from S3", len(sites))

    zip_paths = export_streamstats_batch_shapefiles(sites)
    log.info(
        "Done. %d zip file(s) written to streamstats_batches/:",
        len(zip_paths),
    )
    for p in zip_paths:
        log.info("  %s", p)


if __name__ == "__main__":
    main()
