"""01_get_stations_gcp.py

GCP variant of 01_get_indiana_stations.py.
Identical data logic; reads config_gcp.yaml and writes outputs to GCS.

Writes:
    gs://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet
    gs://<bucket>/<prefix>stations/indiana_streamflow_sites.geojson
    gs://<bucket>/<prefix>stations/indiana_streamflow_sites_active.parquet
    gs://<bucket>/<prefix>stations/indiana_streamflow_sites_active.geojson
"""
from __future__ import annotations

import io
import logging
import os

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dataretrieval import waterdata
from google.cloud import storage
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("01_stations_gcp")


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ["config_gcp.yaml", os.path.join(script_dir, "..", "config_gcp.yaml")]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config_gcp.yaml not found; run from the project root directory.")


# ── GCS helpers ───────────────────────────────────────────────────────────────

_gcs: storage.Client | None = None


def _gcs_client() -> storage.Client:
    global _gcs
    if _gcs is None:
        _gcs = storage.Client()
    return _gcs


def _write_parquet(df: pd.DataFrame, bucket_name: str, blob_name: str) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    buf.seek(0)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_file(
        buf, content_type="application/octet-stream"
    )


def _write_bytes(data: bytes, bucket_name: str, blob_name: str) -> None:
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_string(data)


# ── Station fetch (mirrors 01_get_indiana_stations.py) ────────────────────────

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _extract_point_coords(value) -> tuple:
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
            inner = stripped[stripped.find("(") + 1 : stripped.rfind(")")]
            parts = inner.replace(",", " ").split()
            if len(parts) >= 2:
                return (parts[0], parts[1])
    return (pd.NA, pd.NA)


def _site_number(monitoring_location_id: object) -> str:
    value = str(monitoring_location_id).strip()
    return value.replace("USGS-", "", 1) if value.startswith("USGS-") else value


def fetch_indiana_streamflow_sites(state_name: str, parameter_code: str) -> pd.DataFrame:
    log.info(
        "Querying Water Data metadata: state=%s parameter_code=%s", state_name, parameter_code
    )
    df, _ = waterdata.get_time_series_metadata(
        state_name=state_name, parameter_code=parameter_code
    )
    if df is None or df.empty:
        log.warning("Water Data returned no results")
        return pd.DataFrame(
            columns=[
                "site_no", "station_nm", "dec_lat_va", "dec_long_va",
                "drain_area_va", "huc_cd", "begin_date", "end_date", "count_nu",
            ]
        )

    log.info("Water Data returned %d rows", len(df))
    loc_col      = _first_present(df, ["monitoring_location_id"])
    name_col     = _first_present(df, ["monitoring_location_name", "station_nm"])
    begin_col    = _first_present(df, ["begin", "begin_utc"])
    end_col      = _first_present(df, ["end", "end_utc"])
    huc_col      = _first_present(df, ["hydrologic_unit_code", "huc_cd"])
    drainage_col = _first_present(df, ["drainage_area", "drain_area_va"])
    period_col   = _first_present(df, ["computation_period_identifier", "computation_period"])
    geom_col     = _first_present(df, ["geometry"])

    if loc_col is None:
        raise ValueError("Water Data metadata missing monitoring_location_id")
    if period_col is None:
        raise ValueError("Water Data metadata missing computation_period_identifier")

    before = len(df)
    df = df[df[period_col].astype(str).str.lower() == "points"].copy()
    log.info("Kept %d/%d rows with computation_period_identifier=Points", len(df), before)

    records: list[dict] = []
    for station_id, station_data in df.groupby(loc_col, dropna=True):
        row = station_data.iloc[0]
        lon, lat = _extract_point_coords(row[geom_col]) if geom_col else (pd.NA, pd.NA)
        records.append({
            "site_no":      _site_number(station_id),
            "station_nm":   row[name_col] if name_col else None,
            "dec_lat_va":   lat,
            "dec_long_va":  lon,
            "drain_area_va": row[drainage_col] if drainage_col else pd.NA,
            "huc_cd":       row[huc_col] if huc_col else None,
            "begin_date":   station_data[begin_col].min() if begin_col else pd.NaT,
            "end_date":     station_data[end_col].max() if end_col else pd.NaT,
            "count_nu":     pd.NA,
            "monitoring_location_id":         station_id,
            "computation_period_identifier":  row[period_col] if period_col else None,
        })

    out = pd.DataFrame.from_records(records)
    out["dec_lat_va"]    = pd.to_numeric(out["dec_lat_va"],    errors="coerce")
    out["dec_long_va"]   = pd.to_numeric(out["dec_long_va"],   errors="coerce")
    out["drain_area_va"] = pd.to_numeric(out["drain_area_va"], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["dec_lat_va", "dec_long_va", "begin_date", "end_date"]).sort_values("site_no")
    if before - len(out):
        log.info("Dropped %d stations with missing coordinates or begin/end date", before - len(out))
    log.info("Extracted %d unique Indiana streamflow stations", len(out))
    return out.reset_index(drop=True)


def _to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    for col in ["begin_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    geom = [Point(xy) for xy in zip(df["dec_long_va"], df["dec_lat_va"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = _load_config()
    bucket = cfg["gcp"]["output_bucket"]
    prefix = cfg["gcp"]["output_prefix"]
    param  = cfg["usgs"]["parameter_code"]

    sites = fetch_indiana_streamflow_sites(state_name="Indiana", parameter_code=param)

    keep_cols = [
        "site_no", "station_nm", "dec_lat_va", "dec_long_va",
        "drain_area_va", "huc_cd", "begin_date", "end_date", "count_nu",
        "monitoring_location_id", "computation_period_identifier",
    ]
    sites = sites[[c for c in keep_cols if c in sites.columns]].reset_index(drop=True)

    # Full inventory
    blob = f"{prefix}stations/indiana_streamflow_sites.parquet"
    _write_parquet(sites, bucket, blob)
    log.info("Wrote gs://%s/%s", bucket, blob)

    gdf = _to_geodataframe(sites)
    geojson_blob = f"{prefix}stations/indiana_streamflow_sites.geojson"
    _write_bytes(gdf.to_json().encode(), bucket, geojson_blob)
    log.info("Wrote gs://%s/%s", bucket, geojson_blob)

    # Active inventory (end_date >= 2020-10-14)
    active_cutoff = pd.Timestamp("2020-10-14")
    sites_active = sites[
        pd.to_datetime(sites["end_date"], errors="coerce") >= active_cutoff
    ].copy().reset_index(drop=True)
    log.info("Active stations (end_date >= 2020-10-14): %d / %d", len(sites_active), len(sites))

    blob_a = f"{prefix}stations/indiana_streamflow_sites_active.parquet"
    _write_parquet(sites_active, bucket, blob_a)
    log.info("Wrote gs://%s/%s", bucket, blob_a)

    gdf_active = _to_geodataframe(sites_active)
    geojson_blob_a = f"{prefix}stations/indiana_streamflow_sites_active.geojson"
    _write_bytes(gdf_active.to_json().encode(), bucket, geojson_blob_a)
    log.info("Wrote gs://%s/%s", bucket, geojson_blob_a)

    log.info("Done. %d sites total, %d active.", len(sites), len(sites_active))


if __name__ == "__main__":
    main()
