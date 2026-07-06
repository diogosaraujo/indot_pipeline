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

Nearest precipitation station:
    Each exported point also carries the nearest NOAA hourly precip gauge
    (ISD/LCD + GHCNh, the same networks scored by count_complete_precip_stations)
    that has a (near-)complete record over the MRMS era 2002-2026 — at least 80%
    of all hours in that window carry a precip value.  Coverage is measured the
    same way as count_complete_precip_stations' "full-window" metric:
    distinct hours with a non-NaN value / total hours in 2002-2026.  Added
    properties: precip_site_id, precip_name, precip_source, precip_coverage_pct,
    precip_dist_mi.

Writes (two layers to S3; download from S3 for ArcGIS):
    s3://<bucket>/<prefix>stations/stations_08c.geojson         (streamflow)
    s3://<bucket>/<prefix>stations/precip_stations_08c.geojson  (paired precip)

The streamflow layer carries the precip_* pairing columns; the precip layer has
one point per gauge that was paired, with n_streamflow_paired.  Join the two on
precip_site_id in ArcGIS to draw the pairings.

GeoJSON is written via OGR (RFC 7946) so ArcGIS 'GeoJSON To Features' imports it
cleanly — not GeoDataFrame.to_json(), whose NaN tokens can yield empty output.

Usage:
    python scripts/export_stations_gis.py
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
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

# Two output layers: the evaluated streamflow stations, and the precip gauges
# they were paired to. Both match the other station layers this pipeline writes.
FLOW_OUT_KEY     = "stations/stations_08c.geojson"
PRECIP_OUT_KEY   = "stations/precip_stations_08c.geojson"

# Inventory attributes to carry onto each point.
ATTR_COLS = ["site_no", "station_nm", "dec_lat_va", "dec_long_va",
             "drain_area_va", "huc_cd", "begin_date", "end_date"]

# ── Nearest-precip-station settings ───────────────────────────────────────────
# NOAA hourly precip parquets (written by 12_download_noaa_precip); both carry
# station_id / name / latitude / longitude / datetime_utc / precip_in.
PRECIP_KEYS = {
    "isd":   "precip/noaa/isd_hourly.parquet",
    "ghcnh": "precip/noaa/ghcnh_hourly.parquet",
}
# MRMS era window — same as count_complete_precip_stations.
PRECIP_START = pd.Timestamp("2002-01-01", tz="UTC")
PRECIP_END   = pd.Timestamp("2026-06-30", tz="UTC")
PRECIP_TOTAL_HOURS = int((PRECIP_END - PRECIP_START).total_seconds() // 3600) + 1
# Minimum full-window coverage (% of all hours in 2002-2026 carrying a value).
PRECIP_MIN_COVERAGE = 80.0
EARTH_RADIUS_MI = 3958.7613


def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


_FS = None


def _fs():
    global _FS
    if _FS is None:
        _FS = pafs.S3FileSystem()
    return _FS


def qualifying_precip_stations(bucket: str, prefix: str) -> pd.DataFrame:
    """Precip gauges with >= PRECIP_MIN_COVERAGE of the 2002-2026 window covered.

    Coverage = distinct hours carrying a non-NaN precip value / total hours in
    the window (the "full-window" metric of count_complete_precip_stations).
    Computed in Arrow (group_by), so the full GHCNh table (~200M rows) is never
    materialized as pandas objects — the old to_pandas() version OOM'd and
    silently skipped GHCNh.  Returns one row per station_id with coordinates,
    name, source, coverage_pct.  Only precip presence (not value) is used, so
    GHCNh's millimetre units and COOP accumulation lumps don't matter here.
    """
    start = pa.scalar(PRECIP_START, type=pa.timestamp("us", "UTC"))
    end   = pa.scalar(PRECIP_END,   type=pa.timestamp("us", "UTC"))
    frames: list[pd.DataFrame] = []
    for src, key in PRECIP_KEYS.items():
        try:
            tbl = pq.read_table(
                f"{bucket}/{prefix}{key}", filesystem=_fs(),
                columns=["station_id", "name", "latitude", "longitude",
                         "datetime_utc", "precip_in"])
        except Exception as e:                                   # noqa: BLE001
            log.warning("Precip source %s unavailable, skipping: %s", src, e)
            continue

        dt = pc.cast(tbl["datetime_utc"], pa.timestamp("us", "UTC"))
        tbl = tbl.filter(pc.and_(pc.and_(pc.is_valid(tbl["precip_in"]),
                                         pc.greater_equal(dt, start)),
                                 pc.less_equal(dt, end)))
        if tbl.num_rows == 0:
            continue
        tbl = tbl.append_column(
            "hour", pc.floor_temporal(pc.cast(tbl["datetime_utc"], pa.timestamp("us", "UTC")), unit="hour"))
        covered = (tbl.select(["station_id", "hour"]).group_by(["station_id", "hour"]).aggregate([])
                   .group_by("station_id").aggregate([("hour", "count")]).to_pandas())
        meta = tbl.group_by("station_id").aggregate(
            [("latitude", "min"), ("longitude", "min"), ("name", "min")]).to_pandas()
        cov = covered.merge(meta, on="station_id").rename(columns={
            "hour_count": "covered_hours", "latitude_min": "latitude",
            "longitude_min": "longitude", "name_min": "name"})
        cov["station_id"] = cov["station_id"].astype(str)
        cov["source"] = src
        frames.append(cov)
        log.info("Precip source %s: %d stations with data", src, len(cov))

    if not frames:
        return pd.DataFrame()

    allcov = pd.concat(frames, ignore_index=True)
    allcov["coverage_pct"] = (allcov["covered_hours"] / PRECIP_TOTAL_HOURS * 100).round(1)
    allcov["latitude"]  = pd.to_numeric(allcov["latitude"], errors="coerce")
    allcov["longitude"] = pd.to_numeric(allcov["longitude"], errors="coerce")
    q = allcov[(allcov["coverage_pct"] >= PRECIP_MIN_COVERAGE)].copy()
    q = q.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return q


def _haversine_mi(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles; lat2/lon2 may be arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def attach_nearest_precip(sub: pd.DataFrame, precip: pd.DataFrame) -> pd.DataFrame:
    """Add the nearest qualifying precip gauge to each streamflow point."""
    cols = ["precip_site_id", "precip_name", "precip_source",
            "precip_coverage_pct", "precip_dist_mi"]
    sub = sub.reset_index(drop=True)
    if precip.empty:
        log.warning("No precip stations meet the >= %.0f%% 2002-2026 coverage "
                    "bar; nearest-precip columns will be empty.", PRECIP_MIN_COVERAGE)
        for c in cols:
            sub[c] = None
        return sub

    plat = precip["latitude"].to_numpy(dtype=float)
    plon = precip["longitude"].to_numpy(dtype=float)
    rows: list[dict] = []
    for _, r in sub.iterrows():
        d = _haversine_mi(r["dec_lat_va"], r["dec_long_va"], plat, plon)
        j = int(np.argmin(d))
        p = precip.iloc[j]
        rows.append({
            "precip_site_id":      str(p["station_id"]),
            "precip_name":         p["name"],
            "precip_source":       p["source"],
            "precip_coverage_pct": float(p["coverage_pct"]),
            "precip_dist_mi":      round(float(d[j]), 2),
        })
    return pd.concat([sub, pd.DataFrame(rows, columns=cols)], axis=1)


def build_precip_layer(paired: pd.DataFrame, precip: pd.DataFrame) -> gpd.GeoDataFrame:
    """One point per precip gauge that got paired to >= 1 streamflow station.

    Carries the gauge's coverage plus n_streamflow_paired (how many evaluated
    streamflow stations picked it as their nearest qualifying gauge).
    """
    empty = gpd.GeoDataFrame(
        columns=["precip_site_id", "precip_name", "precip_source",
                 "precip_coverage_pct", "n_streamflow_paired", "geometry"],
        geometry="geometry", crs="EPSG:4326",
    )
    if precip.empty or "precip_site_id" not in paired.columns:
        return empty
    counts = (paired["precip_site_id"].dropna().astype(str)
              .value_counts().rename("n_streamflow_paired"))
    if counts.empty:
        return empty

    sel = precip.copy()
    sel["station_id"] = sel["station_id"].astype(str)
    sel = sel[sel["station_id"].isin(counts.index)]
    sel = sel.merge(counts, left_on="station_id", right_index=True, how="left")

    out = pd.DataFrame({
        "precip_site_id":      sel["station_id"].values,
        "precip_name":         sel["name"].values,
        "precip_source":       sel["source"].values,
        "precip_coverage_pct": pd.to_numeric(sel["coverage_pct"], errors="coerce").values,
        "n_streamflow_paired": sel["n_streamflow_paired"].astype(int).values,
    }).sort_values("precip_site_id").reset_index(drop=True)

    return gpd.GeoDataFrame(
        out,
        geometry=[Point(xy) for xy in zip(sel["longitude"], sel["latitude"])],
        crs="EPSG:4326",
    )


def write_geojson(gdf: gpd.GeoDataFrame, out_key: str,
                  bucket: str, prefix: str, label: str) -> None:
    """Serialize a GeoDataFrame to GeoJSON (via OGR) and upload it to S3.

    OGR-driven output is RFC 7946 compliant — no invalid ``NaN`` tokens, which
    is what makes ArcGIS 'GeoJSON To Features' silently drop everything.  OGR
    needs a real file path, so we render to an ephemeral temp file and upload the
    bytes; nothing persists locally (download from S3 for ArcGIS).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "layer.geojson"
        gdf.to_file(tmp, driver="GeoJSON")
        data = tmp.read_bytes()
    write_bytes_to_s3(data, bucket, f"{prefix}{out_key}")
    log.info("Wrote GeoJSON (%s): s3://%s/%s%s  [%d features]",
             label, bucket, prefix, out_key, len(gdf))


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

    # ── 3. Nearest qualifying precip station per point ───────────────────────
    precip = qualifying_precip_stations(bucket, prefix)
    log.info("Precip gauges with >= %.0f%% 2002-2026 coverage: %d",
             PRECIP_MIN_COVERAGE, len(precip))
    sub = attach_nearest_precip(sub, precip)

    # ── 4. Build the two layers ──────────────────────────────────────────────
    flow_gdf = gpd.GeoDataFrame(
        sub,
        geometry=[Point(xy) for xy in zip(sub["dec_long_va"], sub["dec_lat_va"])],
        crs="EPSG:4326",
    )
    precip_gdf = build_precip_layer(sub, precip)

    # ── 5. Write GeoJSON — local (for ArcGIS) + S3 (matches other layers) ─────
    write_geojson(flow_gdf,   FLOW_OUT_KEY,   bucket, prefix, "streamflow stations")
    write_geojson(precip_gdf, PRECIP_OUT_KEY, bucket, prefix, "paired precip stations")

    log.info("Done. %d streamflow stations, %d paired precip stations.",
             len(flow_gdf), len(precip_gdf))


if __name__ == "__main__":
    main()
