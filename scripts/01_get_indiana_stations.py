"""01_get_indiana_stations.py

Fetch the complete inventory of USGS streamflow gauges in Indiana with
instantaneous/unit-value discharge data, and write a single Parquet
table (and matching GeoJSON) to S3.

Uses Water Data API endpoints:
  1. `waterdata.get_monitoring_locations()` for location metadata
  2. `waterdata.get_time_series_metadata()` for instantaneous discharge
     series coverage

Output schema (one row per gauge):
    site_no, station_nm, dec_lat_va, dec_long_va, drain_area_va,
    huc_cd, begin_date, end_date, count_nu

Writes:
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.geojson
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
from dataretrieval import waterdata
from shapely.geometry import Point

from utils import load_config, write_bytes_to_s3, write_parquet_to_s3

log = logging.getLogger("01_stations")


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def fetch_monitoring_locations(state_code: str) -> pd.DataFrame:
    """Fetch Indiana stream monitoring locations from the Water Data API."""
    log.info("Querying Water Data monitoring locations for state=%s", state_code)
    df, _ = waterdata.get_monitoring_locations(
        agency_code=["USGS"],
        state_code=[state_code],
        site_type_code=["ST"],
    )
    return df.reset_index(drop=True)


def fetch_iv_series_metadata(state_code: str, parameter_code: str) -> pd.DataFrame:
    """Fetch IV streamflow time-series metadata for Indiana gauges."""
    log.info("Querying Water Data time-series metadata for IV streamflow in %s", state_code)
    df, _ = waterdata.get_time_series_metadata(
        parameter_code=parameter_code,
        statistic_id="00011",
        state_name="Indiana",
    )
    if df.empty:
        return df

    loc_col = first_present(df, ["monitoring_location_id"])
    begin_col = first_present(df, ["begin", "begin_utc"])
    end_col = first_present(df, ["end", "end_utc"])
    primary_col = first_present(df, ["primary"])

    if loc_col is None:
        raise ValueError("Water Data time-series metadata did not include monitoring_location_id")

    series = df.copy()
    series = series[series[loc_col].astype(str).str.startswith("USGS-")].copy()
    if primary_col is not None:
        primary_vals = series[primary_col].astype(str).str.lower()
        series = series[primary_vals.isin(["true", "t", "1", "yes"])].copy()

    series["site_no"] = series[loc_col].astype(str).str.replace("USGS-", "", regex=False)
    series["begin_date"] = pd.to_datetime(series[begin_col], errors="coerce").dt.date if begin_col else pd.NaT
    series["end_date"] = pd.to_datetime(series[end_col], errors="coerce").dt.date if end_col else pd.NaT
    series["count_nu"] = pd.NA
    series = series.sort_values(["site_no", "begin_date", "end_date"]).drop_duplicates("site_no", keep="first")
    keep = ["site_no", "begin_date", "end_date", "count_nu"]
    log.info("Time-series metadata yielded %d IV sites", len(series))
    return series[keep].reset_index(drop=True)


def standardize_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    site_id_col = first_present(df, ["monitoring_location_id", "site_no"])
    name_col = first_present(df, ["monitoring_location_name", "station_nm"])
    lat_col = first_present(df, ["latitude", "dec_lat_va"])
    lon_col = first_present(df, ["longitude", "dec_long_va"])
    drainage_col = first_present(df, ["drainage_area", "drain_area_va"])
    huc_col = first_present(df, ["hydrologic_unit_code", "huc_cd"])

    if site_id_col is None:
        raise ValueError("Monitoring locations response did not include a site identifier")

    out = pd.DataFrame()
    raw_site = df[site_id_col].astype(str)
    out["site_no"] = raw_site.str.replace("USGS-", "", regex=False)
    out["station_nm"] = df[name_col] if name_col else None

    if lat_col and lon_col:
        out["dec_lat_va"] = pd.to_numeric(df[lat_col], errors="coerce")
        out["dec_long_va"] = pd.to_numeric(df[lon_col], errors="coerce")
    elif "geometry" in df.columns:
        out["dec_lat_va"] = df.geometry.y
        out["dec_long_va"] = df.geometry.x
    else:
        out["dec_lat_va"] = pd.NA
        out["dec_long_va"] = pd.NA

    out["drain_area_va"] = pd.to_numeric(df[drainage_col], errors="coerce") if drainage_col else pd.NA
    out["huc_cd"] = df[huc_col] if huc_col else None
    return out


def fetch_indiana_streamflow_sites(state_code: str, parameter_code: str) -> pd.DataFrame:
    locations = standardize_locations(fetch_monitoring_locations(state_code))
    series = fetch_iv_series_metadata(state_code, parameter_code)
    sites = locations.merge(series, on="site_no", how="inner")
    sites = sites.drop_duplicates("site_no").reset_index(drop=True)
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
        "drain_area_va", "huc_cd", "begin_date", "end_date", "count_nu",
    ]
    keep_cols = [c for c in keep_cols if c in sites.columns]
    sites = sites[keep_cols].reset_index(drop=True)

    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    write_parquet_to_s3(sites, bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    log.info("Wrote Parquet: s3://%s/%sstations/indiana_streamflow_sites.parquet", bucket, prefix)

    gdf = to_geodataframe(sites)
    write_bytes_to_s3(
        gdf.to_json().encode(),
        bucket,
        f"{prefix}stations/indiana_streamflow_sites.geojson",
    )
    log.info("Wrote GeoJSON: s3://%s/%sstations/indiana_streamflow_sites.geojson", bucket, prefix)
    log.info("Done. %d sites total.", len(sites))


if __name__ == "__main__":
    main()
