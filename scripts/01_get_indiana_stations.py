"""01_get_indiana_stations.py

Fetch the complete inventory of USGS streamflow gauges in Indiana with
instantaneous/unit-value discharge data, and write a single Parquet
table (and matching GeoJSON) to S3.

Uses Water Data API endpoints:
  1. `waterdata.get_time_series_metadata()` for streamflow series coverage
  2. `waterdata.get_monitoring_locations()` for stream-site filtering and
     extra location metadata

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


def normalize_monitoring_location_ids(df: pd.DataFrame) -> pd.Series:
    """Return Water Data monitoring location IDs like USGS-03339000."""
    if "monitoring_location_id" in df.columns:
        ids = df["monitoring_location_id"].astype(str)
        ids = ids.where(ids.str.strip().ne(""), pd.NA)
        if ids.notna().any():
            return ids

    agency_col = first_present(df, ["agency_code", "agency_cd"])
    number_col = first_present(
        df,
        ["monitoring_location_number", "site_no", "site_number"],
    )
    if agency_col is not None and number_col is not None:
        return (
            df[agency_col].astype(str).str.strip()
            + "-"
            + df[number_col].astype(str).str.strip()
        )
    if number_col is not None:
        return "USGS-" + df[number_col].astype(str).str.strip()
    return pd.Series(pd.NA, index=df.index, dtype="object")


def fetch_monitoring_locations(state_code: str) -> pd.DataFrame:
    """Fetch Indiana stream monitoring locations from the Water Data API."""
    log.info("Querying Water Data monitoring locations for state=%s", state_code)
    df, _ = waterdata.get_monitoring_locations(
        agency_code=["USGS"],
        state_code=[state_code],
        site_type_code=["ST"],
    )
    df = df.reset_index(drop=True)
    df["monitoring_location_id"] = normalize_monitoring_location_ids(df)
    return df


def fetch_streamflow_series_metadata(state_name: str, parameter_code: str) -> pd.DataFrame:
    """Fetch streamflow time-series metadata for Indiana and reduce to one row per site."""
    log.info("Querying Water Data time-series metadata for streamflow in %s", state_name)
    df, _ = waterdata.get_time_series_metadata(
        state_name=state_name,
        parameter_code=parameter_code,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["site_no", "begin_date", "end_date", "count_nu"])

    loc_col = first_present(df, ["monitoring_location_id"])
    begin_col = first_present(df, ["begin", "begin_utc"])
    end_col = first_present(df, ["end", "end_utc"])
    primary_col = first_present(df, ["primary"])
    period_col = first_present(df, ["computation_period_identifier", "computation_period"])
    statistic_col = first_present(df, ["statistic_id"])

    if loc_col is None:
        raise ValueError("Water Data time-series metadata did not include monitoring_location_id")

    series = df.copy()
    series = series[series[loc_col].astype(str).str.startswith("USGS-")].copy()
    is_continuous = pd.Series(False, index=series.index)
    if period_col is not None:
        period_vals = series[period_col].astype(str).str.lower()
        is_continuous = is_continuous | period_vals.str.contains(
            "instant|continuous|unit|iv",
            regex=True,
            na=False,
        )
    if statistic_col is not None:
        is_continuous = is_continuous | series[statistic_col].astype(str).eq("00011")
    if is_continuous.any():
        series = series[is_continuous].copy()
    else:
        log.warning(
            "Could not identify continuous-only rows from metadata labels; keeping all %s streamflow series",
            parameter_code,
        )

    if primary_col is not None:
        primary_vals = series[primary_col].astype(str).str.lower()
        series = series[primary_vals.isin(["true", "t", "1", "yes"])].copy()

    series["site_no"] = series[loc_col].astype(str).str.replace("USGS-", "", regex=False)
    series["begin_date"] = pd.to_datetime(series[begin_col], errors="coerce").dt.date if begin_col else pd.NaT
    series["end_date"] = pd.to_datetime(series[end_col], errors="coerce").dt.date if end_col else pd.NaT
    series["count_nu"] = pd.NA
    series = series.sort_values(["site_no", "begin_date", "end_date"]).drop_duplicates("site_no", keep="first")
    keep = ["site_no", "begin_date", "end_date", "count_nu"]
    log.info("Time-series metadata yielded %d streamflow sites", len(series))
    return series[keep].reset_index(drop=True)


def standardize_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["monitoring_location_id"] = normalize_monitoring_location_ids(df)
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
        coords = df["geometry"].apply(_extract_point_coords)
        out["dec_long_va"] = pd.to_numeric(coords.str[0], errors="coerce")
        out["dec_lat_va"] = pd.to_numeric(coords.str[1], errors="coerce")
    else:
        out["dec_lat_va"] = pd.NA
        out["dec_long_va"] = pd.NA

    out["drain_area_va"] = pd.to_numeric(df[drainage_col], errors="coerce") if drainage_col else pd.NA
    out["huc_cd"] = df[huc_col] if huc_col else None
    return out


def _extract_point_coords(value) -> tuple[object, object]:
    """Return (lon, lat) from a geometry-like object when explicit columns are absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (pd.NA, pd.NA)
    if hasattr(value, "x") and hasattr(value, "y"):
        return (value.x, value.y)
    if isinstance(value, dict):
        coords = value.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return (coords[0], coords[1])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("POINT"):
            inner = stripped[stripped.find("(") + 1:stripped.rfind(")")]
            parts = inner.replace(",", " ").split()
            if len(parts) >= 2:
                return (parts[0], parts[1])
    return (pd.NA, pd.NA)


def fetch_indiana_streamflow_sites(state_code: str, parameter_code: str) -> pd.DataFrame:
    raw_locations = fetch_monitoring_locations(state_code)
    locations = standardize_locations(raw_locations)
    series = fetch_streamflow_series_metadata("Indiana", parameter_code)
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
