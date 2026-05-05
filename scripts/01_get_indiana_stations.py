"""01_get_indiana_stations.py

Fetch the inventory of USGS streamflow stations in Indiana from the USGS
Water Data API and write a single Parquet table (and matching GeoJSON) to S3.

This follows the same approach as the prior local station-listing script:
query `waterdata.get_time_series_metadata(state_name="Indiana",
parameter_code="00060")`, then reduce the returned metadata to one row per
monitoring location.

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


def _extract_point_coords(value) -> tuple[object, object]:
    """Return (lon, lat) from a geometry-like object."""
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
        if stripped.upper().startswith("POINT"):
            inner = stripped[stripped.find("(") + 1:stripped.rfind(")")]
            parts = inner.replace(",", " ").split()
            if len(parts) >= 2:
                return (parts[0], parts[1])
    return (pd.NA, pd.NA)


def _site_number(monitoring_location_id: object) -> str:
    value = str(monitoring_location_id).strip()
    return value.replace("USGS-", "", 1) if value.startswith("USGS-") else value


def fetch_indiana_streamflow_sites(state_name: str, parameter_code: str) -> pd.DataFrame:
    """Query Water Data metadata and reduce it to station-level rows."""
    log.info(
        "Querying Water Data time-series metadata for state=%s parameter_code=%s",
        state_name,
        parameter_code,
    )
    df, _ = waterdata.get_time_series_metadata(
        state_name=state_name,
        parameter_code=parameter_code,
    )
    if df is None or df.empty:
        log.warning("Water Data returned no streamflow time-series metadata")
        return pd.DataFrame(columns=[
            "site_no", "station_nm", "dec_lat_va", "dec_long_va",
            "drain_area_va", "huc_cd", "begin_date", "end_date", "count_nu",
        ])

    log.info("Water Data returned %d streamflow metadata rows", len(df))
    loc_col = first_present(df, ["monitoring_location_id"])
    if loc_col is None:
        raise ValueError("Water Data metadata did not include monitoring_location_id")

    name_col = first_present(df, ["monitoring_location_name", "station_nm"])
    begin_col = first_present(df, ["begin", "begin_utc"])
    end_col = first_present(df, ["end", "end_utc"])
    huc_col = first_present(df, ["hydrologic_unit_code", "huc_cd"])
    drainage_col = first_present(df, ["drainage_area", "drain_area_va"])
    period_col = first_present(df, ["computation_period_identifier", "computation_period"])
    geom_col = first_present(df, ["geometry"])

    if period_col is None:
        raise ValueError("Water Data metadata did not include computation_period_identifier")
    before = len(df)
    df = df[df[period_col].astype(str).str.lower() == "points"].copy()
    log.info("Kept %d/%d metadata rows with computation_period_identifier=Points", len(df), before)

    records: list[dict] = []
    for station_id, station_data in df.groupby(loc_col, dropna=True):
        row = station_data.iloc[0]
        lon, lat = _extract_point_coords(row[geom_col]) if geom_col else (pd.NA, pd.NA)

        records.append({
            "site_no": _site_number(station_id),
            "station_nm": row[name_col] if name_col else None,
            "dec_lat_va": lat,
            "dec_long_va": lon,
            "drain_area_va": row[drainage_col] if drainage_col else pd.NA,
            "huc_cd": row[huc_col] if huc_col else None,
            "begin_date": station_data[begin_col].min() if begin_col else pd.NaT,
            "end_date": station_data[end_col].max() if end_col else pd.NaT,
            "count_nu": pd.NA,
            "monitoring_location_id": station_id,
            "computation_period_identifier": row[period_col] if period_col else None,
        })

    out = pd.DataFrame.from_records(records)
    out["dec_lat_va"] = pd.to_numeric(out["dec_lat_va"], errors="coerce")
    out["dec_long_va"] = pd.to_numeric(out["dec_long_va"], errors="coerce")
    out["drain_area_va"] = pd.to_numeric(out["drain_area_va"], errors="coerce")
    out = out.dropna(subset=["dec_lat_va", "dec_long_va"]).sort_values("site_no")
    log.info("Extracted %d unique Indiana streamflow stations", len(out))
    return out.reset_index(drop=True)


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    for col in ["begin_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    geom = [Point(xy) for xy in zip(df["dec_long_va"], df["dec_lat_va"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def main() -> None:
    cfg = load_config()
    sites = fetch_indiana_streamflow_sites(
        state_name="Indiana",
        parameter_code=cfg["usgs"]["parameter_code"],
    )
    keep_cols = [
        "site_no", "station_nm", "dec_lat_va", "dec_long_va",
        "drain_area_va", "huc_cd", "begin_date", "end_date", "count_nu",
        "monitoring_location_id", "computation_period_identifier",
    ]
    keep_cols = [c for c in keep_cols if c in sites.columns]
    sites = sites[keep_cols].reset_index(drop=True)

    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    # --- Full inventory (all stations, used by scripts 02 and 03) ---
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

    # --- Active inventory (end_date >= 2018-01-01, used by scripts 05 and 06) ---
    # Filters to stations with data in the MRMS/NWM era for precipitation pairing.
    # Stations with earlier end dates are still downloaded by script 02 for future use.
    active_cutoff = pd.Timestamp("2018-01-01")
    sites_active = sites[
        pd.to_datetime(sites["end_date"], errors="coerce") >= active_cutoff
    ].copy().reset_index(drop=True)
    log.info(
        "Active stations (end_date >= 2018-01-01): %d / %d",
        len(sites_active), len(sites),
    )

    write_parquet_to_s3(
        sites_active, bucket, f"{prefix}stations/indiana_streamflow_sites_active.parquet"
    )
    log.info(
        "Wrote Parquet: s3://%s/%sstations/indiana_streamflow_sites_active.parquet",
        bucket, prefix,
    )

    gdf_active = to_geodataframe(sites_active)
    write_bytes_to_s3(
        gdf_active.to_json().encode(),
        bucket,
        f"{prefix}stations/indiana_streamflow_sites_active.geojson",
    )
    log.info(
        "Wrote GeoJSON: s3://%s/%sstations/indiana_streamflow_sites_active.geojson",
        bucket, prefix,
    )


if __name__ == "__main__":
    main()
