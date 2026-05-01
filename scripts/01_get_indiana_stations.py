"""01_get_indiana_stations.py

Fetch the complete inventory of USGS streamflow gauges in Indiana with
instantaneous/unit-value discharge data, and write a single Parquet
table (and matching GeoJSON) to S3.

Uses two paths:
  1. `nwis.what_sites()` for the basic site inventory
  2. direct NWIS REST `seriesCatalogOutput=true` for record dates, counts,
     and series type filtering

Output schema (one row per gauge):
    site_no, station_nm, dec_lat_va, dec_long_va, drain_area_va,
    huc_cd, begin_date, end_date, count_nu

Writes:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.geojson
"""
from __future__ import annotations

import logging
from io import StringIO

import geopandas as gpd
import pandas as pd
import requests
from dataretrieval import nwis
from shapely.geometry import Point

from utils import load_config, write_bytes_to_s3, write_parquet_to_s3

log = logging.getLogger("01_stations")


def fetch_basic_sites(state_code: str, parameter_code: str) -> pd.DataFrame:
    """Query NWIS for all Indiana stream sites with discharge data."""
    log.info("Querying NWIS what_sites for state=%s parameterCd=%s", state_code, parameter_code)
    df, _ = nwis.what_sites(
        stateCd=state_code,
        parameterCd=parameter_code,
        siteType="ST",
    )
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    return df.reset_index(drop=True)


def fetch_series_catalog(state_code: str, parameter_code: str) -> pd.DataFrame:
    """Fetch raw series-catalog rows and keep only instantaneous discharge."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "stateCd": state_code,
        "parameterCd": parameter_code,
        "siteType": "ST",
        "seriesCatalogOutput": "true",
        "siteStatus": "all",
    }
    log.info("Fetching NWIS series catalog for instantaneous discharge filtering")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()

    lines = [line for line in r.text.splitlines() if not line.startswith("#")]
    if len(lines) < 3:
        raise ValueError("Series catalog response was shorter than expected")

    rdb_text = lines[0] + "\n" + "\n".join(lines[2:])
    cat = pd.read_csv(StringIO(rdb_text), sep="\t", dtype=str)

    if "parm_cd" in cat.columns:
        cat = cat[cat["parm_cd"] == parameter_code].copy()
    if "data_type_cd" in cat.columns:
        cat = cat[cat["data_type_cd"].str.lower() == "uv"].copy()
    else:
        raise ValueError("Series catalog response did not include data_type_cd")

    if "count_nu" in cat.columns:
        cat["count_nu"] = pd.to_numeric(cat["count_nu"], errors="coerce").fillna(0)
    else:
        cat["count_nu"] = 0

    cat = cat.sort_values("count_nu", ascending=False).drop_duplicates("site_no", keep="first")
    keep = [c for c in ["site_no", "begin_date", "end_date", "count_nu", "data_type_cd", "parm_cd"] if c in cat.columns]
    log.info("Series catalog yielded %d instantaneous sites", len(cat))
    return cat[keep].reset_index(drop=True)


def fetch_indiana_streamflow_sites(state_code: str, parameter_code: str) -> pd.DataFrame:
    basic = fetch_basic_sites(state_code, parameter_code)
    catalog = fetch_series_catalog(state_code, parameter_code)

    basic["site_no"] = basic["site_no"].astype(str)
    catalog["site_no"] = catalog["site_no"].astype(str)
    sites = basic.merge(catalog, on="site_no", how="inner")
    log.info("Merged inventory contains %d instantaneous discharge sites", len(sites))
    return sites


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df["dec_lat_va"] = pd.to_numeric(df["dec_lat_va"], errors="coerce")
    df["dec_long_va"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
    df = df.dropna(subset=["dec_lat_va", "dec_long_va"])
    geom = [Point(xy) for xy in zip(df["dec_long_va"], df["dec_lat_va"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def main() -> None:
    cfg = load_config()
    sites = fetch_indiana_streamflow_sites(
        cfg["usgs"]["state_code"], cfg["usgs"]["parameter_code"]
    )
    keep_cols = [
        "site_no", "station_nm", "dec_lat_va", "dec_long_va",
        "drain_area_va", "huc_cd", "begin_date", "end_date",
        "count_nu", "data_type_cd", "parm_cd",
    ]
    keep_cols = [c for c in keep_cols if c in sites.columns]
    sites = sites[keep_cols].reset_index(drop=True)

    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    write_parquet_to_s3(sites, bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    log.info("Wrote Parquet: s3://%s/%sstations/indiana_streamflow_sites.parquet", bucket, prefix)

    gdf = to_geodataframe(sites)
    geojson = gdf.to_json().encode()
    write_bytes_to_s3(geojson, bucket, f"{prefix}stations/indiana_streamflow_sites.geojson")
    log.info("Wrote GeoJSON: s3://%s/%sstations/indiana_streamflow_sites.geojson", bucket, prefix)
    log.info("Done. %d sites total.", len(sites))


if __name__ == "__main__":
    main()
