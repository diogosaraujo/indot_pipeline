"""Build the Indiana bridge coverage figure from bridge, USGS, and NWM data.

This script updates the bridge coverage accounting using the project S3 bucket:

* Bridges come from ``bridge_data/bridge_location_csv`` by default.
* Bridges over waterways are NBI item 113 values 0..8.
* Scour critical bridges are NBI item 113 values <= 3.
* USGS streamflow stations are counted only if the local streamflow parquet has
  2026 records. Each active station covers its nearest bridge.
* NWM stream coverage is based on a bridge being within 200 ft of an NWM/NHDPlus
  flowline, not just a reach outlet. If a full flowline file is supplied, it is
  used directly. Otherwise the script queries NLDI per bridge to find the nearest
  COMID, fetches that COMID flowline, and verifies the true line distance.

Writes:
    s3://<bucket>/<prefix>analysis/bridge_coverage/bridge_coverage_flags.parquet
    s3://<bucket>/<prefix>analysis/bridge_coverage/bridge_coverage_summary.csv
    s3://<bucket>/<prefix>analysis/figures/bridge_coverage_updated.png

Usage on EC2:
    python scripts/14_bridge_coverage_figure.py

Optional, preferred when you already have a complete Indiana NWM/NHDPlus
flowline file:
    python scripts/14_bridge_coverage_figure.py \
        --nwm-flowlines s3://my-bucket/v1/bridge_data/indiana_nwm_flowlines.gpkg
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import boto3
import botocore.exceptions
import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import requests
from shapely.geometry import shape

from utils import load_config, s3_client, write_bytes_to_s3, write_parquet_to_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("14_bridge_coverage")

BRIDGE_KEY_DEFAULT = "bridge_data/bridge_location_csv"
STREAMFLOW_KEY = "streamflow/instantaneous/all_gauges_long.parquet"
STATION_KEY = "stations/indiana_streamflow_sites.parquet"
FLAGS_KEY = "analysis/bridge_coverage/bridge_coverage_flags.parquet"
SUMMARY_KEY = "analysis/bridge_coverage/bridge_coverage_summary.csv"
FIG_KEY = "analysis/figures/bridge_coverage_updated.png"

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
STATE_PLANE_EAST_FT = "EPSG:2965"
WGS84 = "EPSG:4326"


def _read_s3_bytes(bucket: str, key: str) -> bytes:
    return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _object_exists(bucket: str, key: str) -> bool:
    try:
        s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def _resolve_bridge_key(bucket: str, prefix: str, rel_key: str) -> str:
    key = f"{prefix}{rel_key}"
    if _object_exists(bucket, key):
        return key

    base = key.rstrip("/")
    for suffix in [".csv", ".CSV", ".parquet", ".geojson", ".json"]:
        if _object_exists(bucket, f"{base}{suffix}"):
            return f"{base}{suffix}"

    parent = str(Path(base).parent).replace("\\", "/")
    name = Path(base).name
    resp = s3_client().list_objects_v2(Bucket=bucket, Prefix=f"{parent}/")
    matches = [
        obj["Key"]
        for obj in resp.get("Contents", [])
        if Path(obj["Key"]).name.startswith(name)
    ]
    if not matches:
        raise FileNotFoundError(f"No bridge file found at s3://{bucket}/{key}")
    if len(matches) > 1:
        log.warning("Multiple bridge file candidates found; using %s", matches[0])
    return matches[0]


def _read_table_from_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _read_s3_bytes(bucket, key)
    lower = key.lower()
    if lower.endswith(".parquet"):
        return pq.read_table(io.BytesIO(body)).to_pandas()
    if lower.endswith(".geojson") or lower.endswith(".json"):
        return gpd.read_file(io.BytesIO(body))
    return pd.read_csv(io.BytesIO(body))


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip("/")
    return f"{prefix}/" if prefix else ""


def _choose_column(df: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {kind} column {requested!r} not found")
        return requested

    norm_to_actual = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        c = norm_to_actual.get(_normalize_col(cand))
        if c is not None:
            return c

    lowered = {c: _normalize_col(c) for c in df.columns}
    if kind == "latitude":
        hits = [c for c, n in lowered.items() if "lat" in n and "delta" not in n]
    elif kind == "longitude":
        hits = [c for c, n in lowered.items() if ("lon" in n or "long" in n) and "along" not in n]
    elif kind == "scour":
        hits = [c for c, n in lowered.items() if "113" in n or "scour" in n]
    elif kind == "bridge id":
        hits = [
            c for c, n in lowered.items()
            if n in {"assetname", "bridgeid", "structureid", "structurenumber", "strucnum", "nbi"}
        ]
    else:
        hits = []

    if hits:
        return hits[0]
    raise ValueError(f"Could not infer {kind} column. Use the matching CLI override.")


def _coerce_decimal_degrees(s: pd.Series, is_lon: bool) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    limit = 180 if is_lon else 90

    # Some NBI exports store DMS-like integers. Keep ordinary decimal degrees as-is.
    too_large = vals.abs() > limit
    if too_large.any():
        abs_vals = vals.abs()
        deg = (abs_vals // 1_000_000).astype("float64")
        mins = ((abs_vals - deg * 1_000_000) // 10_000).astype("float64")
        secs = (abs_vals - deg * 1_000_000 - mins * 10_000) / 100
        converted = deg + mins / 60 + secs / 3600
        vals = vals.where(~too_large, converted)

    if is_lon:
        vals = vals.where(vals < 0, -vals)
    return vals


def load_bridges(
    bucket: str,
    key: str,
    lat_col: str | None,
    lon_col: str | None,
    scour_col: str | None,
    id_col: str | None,
) -> gpd.GeoDataFrame:
    df = _read_table_from_s3(bucket, key)
    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name in df.columns and not df.geometry.isna().all():
        gdf = df.to_crs(WGS84)
    else:
        lat = _choose_column(
            df, lat_col,
            ["(B.L.05 ) Latitude", "B.L.05 Latitude", "latitude", "lat", "bridge_lat", "y"],
            "latitude",
        )
        lon = _choose_column(
            df, lon_col,
            ["(B.L.06 ) Longitude", "B.L.06 Longitude", "longitude", "lon", "long", "bridge_lon", "x"],
            "longitude",
        )
        df[lat] = _coerce_decimal_degrees(df[lat], is_lon=False)
        df[lon] = _coerce_decimal_degrees(df[lon], is_lon=True)
        df = df[df[lat].between(37.0, 42.5) & df[lon].between(-89.0, -84.0)].copy()
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=WGS84)

    scour = _choose_column(
        gdf, scour_col,
        ["(113) Scour Critical Bridges", "nbi_113", "item_113", "scour_critical", "scour"],
        "scour",
    )
    bridge_id = id_col
    if bridge_id is None:
        try:
            bridge_id = _choose_column(
                gdf, None,
                ["Asset Name", "bridge_id", "structure_id", "structure_number"],
                "bridge id",
            )
        except ValueError:
            bridge_id = "_bridge_index"
            gdf[bridge_id] = range(len(gdf))

    gdf = gdf.reset_index(drop=True)
    gdf["bridge_id"] = gdf[bridge_id].astype(str)
    gdf["nbi_113_raw"] = gdf[scour]
    gdf["nbi_113"] = pd.to_numeric(gdf[scour].astype(str).str.extract(r"(-?\d+)")[0], errors="coerce")
    gdf["over_waterway"] = gdf["nbi_113"].between(0, 8, inclusive="both")
    gdf["scour_critical"] = gdf["nbi_113"].le(3) & gdf["over_waterway"]
    return gdf


def read_active_usgs_stations(bucket: str, prefix: str) -> gpd.GeoDataFrame:
    station_df = _read_table_from_s3(bucket, f"{prefix}{STATION_KEY}")
    station_df["site_no"] = station_df["site_no"].astype(str)

    obj = s3_client().get_object(Bucket=bucket, Key=f"{prefix}{STREAMFLOW_KEY}")
    body = obj["Body"].read()
    try:
        table = pq.read_table(
            io.BytesIO(body),
            columns=["site_no", "datetime_utc"],
            filters=[("datetime_utc", ">=", pd.Timestamp("2026-01-01", tz="UTC"))],
        )
    except Exception:
        table = pq.read_table(io.BytesIO(body), columns=["site_no", "datetime_utc"])
    flow = table.to_pandas()
    flow["datetime_utc"] = pd.to_datetime(flow["datetime_utc"], utc=True, errors="coerce")
    flow = flow[flow["datetime_utc"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    flow["site_no"] = flow["site_no"].astype(str)
    active_sites = set(flow["site_no"].unique())
    log.info("USGS streamflow stations with 2026 records: %d", len(active_sites))

    active = station_df[station_df["site_no"].isin(active_sites)].copy()
    gdf = gpd.GeoDataFrame(
        active,
        geometry=gpd.points_from_xy(active["dec_long_va"], active["dec_lat_va"]),
        crs=WGS84,
    )
    return gdf


def assign_usgs_to_nearest_bridge(bridges: gpd.GeoDataFrame, stations: gpd.GeoDataFrame) -> set[str]:
    if stations.empty or bridges.empty:
        return set()

    b = bridges[["bridge_id", "geometry"]].to_crs(STATE_PLANE_EAST_FT)
    s = stations[["site_no", "geometry"]].to_crs(STATE_PLANE_EAST_FT)
    nearest = gpd.sjoin_nearest(s, b, how="left", distance_col="nearest_bridge_dist_ft")
    covered = set(nearest["bridge_id"].dropna().astype(str))
    log.info("USGS active stations cover %d unique nearest bridges", len(covered))
    return covered


def _download_s3_uri(uri: str, out_dir: Path) -> Path:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local = out_dir / Path(key).name
    s3_client().download_file(bucket, key, str(local))
    return local


def _read_flowlines(path_or_uri: str) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as td:
        path = Path(path_or_uri)
        source: str | Path = path
        if path_or_uri.startswith("s3://"):
            source = _download_s3_uri(path_or_uri, Path(td))
        gdf = gpd.read_file(source)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    if "comid" not in gdf.columns:
        for c in gdf.columns:
            if _normalize_col(c) in {"comid", "featureid", "feature_id", "link"}:
                gdf = gdf.rename(columns={c: "comid"})
                break
    if "comid" not in gdf.columns:
        raise ValueError("NWM flowline file must contain a COMID/feature_id/link column")
    return gdf[["comid", "geometry"]].copy()


def cover_by_flowline_file(
    bridges: gpd.GeoDataFrame,
    flowline_path: str,
    threshold_ft: float,
) -> pd.DataFrame:
    streams = _read_flowlines(flowline_path)
    bridges_ft = bridges[["bridge_id", "geometry"]].to_crs(STATE_PLANE_EAST_FT)
    streams_ft = streams.to_crs(STATE_PLANE_EAST_FT)

    joined = gpd.sjoin_nearest(
        bridges_ft,
        streams_ft[["comid", "geometry"]],
        how="left",
        max_distance=threshold_ft,
        distance_col="nwm_dist_ft",
    )
    joined = joined.drop_duplicates("bridge_id")
    out = joined[["bridge_id", "comid", "nwm_dist_ft"]].copy()
    out["nwm_covered"] = out["comid"].notna()
    return out


def _fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any] | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            log.debug("NLDI %s returned %s", url, r.status_code)
        except Exception as exc:  # noqa: BLE001
            log.debug("NLDI request failed: %s", exc)
        time.sleep(min(20, 1.5 * attempt))
    return None


def _nearest_comid_for_point(lon: float, lat: float, timeout: int, retries: int) -> int | None:
    coords = quote(f"POINT({lon:.8f} {lat:.8f})", safe="")
    js = _fetch_json(f"{NLDI_BASE}/comid/position?coords={coords}", timeout, retries)
    if not js:
        return None
    feats = js.get("features", [])
    if not feats:
        return None
    props = feats[0].get("properties", {})
    for key in ["comid", "COMID", "identifier", "nhdplus_comid"]:
        if key in props and props[key] is not None:
            return int(str(props[key]).replace("comid-", ""))
    return None


def _flowline_for_comid(comid: int, timeout: int, retries: int):
    js = _fetch_json(f"{NLDI_BASE}/comid/{comid}", timeout, retries)
    if not js:
        return None
    feats = js.get("features", [])
    if not feats:
        return None
    geom = feats[0].get("geometry")
    return shape(geom) if geom else None


def cover_by_nldi(
    bridges: gpd.GeoDataFrame,
    threshold_ft: float,
    max_workers: int,
    timeout: int,
    retries: int,
) -> pd.DataFrame:
    """Fallback NWM coverage using NLDI nearest-COMID lookup and line geometry."""
    points = bridges[["bridge_id", "geometry"]].to_crs(WGS84).copy()
    rows = list(points.itertuples(index=False))

    log.info("Querying nearest NLDI COMID for %d bridges...", len(rows))
    comid_by_bridge: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_nearest_comid_for_point, row.geometry.x, row.geometry.y, timeout, retries): row.bridge_id
            for row in rows
        }
        for i, fut in enumerate(as_completed(futs), 1):
            bridge_id = futs[fut]
            try:
                comid_by_bridge[str(bridge_id)] = fut.result()
            except Exception:  # noqa: BLE001
                comid_by_bridge[str(bridge_id)] = None
            if i % 500 == 0:
                log.info("  nearest COMID lookup: %d / %d", i, len(rows))

    unique_comids = sorted({c for c in comid_by_bridge.values() if c is not None})
    log.info("Fetching %d unique NLDI flowline geometries...", len(unique_comids))
    geoms: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_flowline_for_comid, c, timeout, retries): c for c in unique_comids}
        for i, fut in enumerate(as_completed(futs), 1):
            comid = futs[fut]
            try:
                geom = fut.result()
                if geom is not None:
                    geoms[comid] = geom
            except Exception:  # noqa: BLE001
                pass
            if i % 250 == 0:
                log.info("  flowline fetch: %d / %d", i, len(unique_comids))

    stream_gdf = gpd.GeoDataFrame(
        [{"comid": c, "geometry": g} for c, g in geoms.items()],
        crs=WGS84,
    ).to_crs(STATE_PLANE_EAST_FT)
    geom_by_comid = dict(zip(stream_gdf["comid"].astype(int), stream_gdf.geometry))
    points_ft = points.to_crs(STATE_PLANE_EAST_FT)

    out_rows = []
    for row in points_ft.itertuples(index=False):
        comid = comid_by_bridge.get(str(row.bridge_id))
        line = geom_by_comid.get(int(comid)) if comid is not None else None
        dist = float(row.geometry.distance(line)) if line is not None else math.nan
        out_rows.append({
            "bridge_id": str(row.bridge_id),
            "comid": int(comid) if comid is not None else None,
            "nwm_dist_ft": dist,
            "nwm_covered": bool(pd.notna(dist) and dist <= threshold_ft),
        })
    return pd.DataFrame(out_rows)


def write_summary_csv(summary: dict[str, int], bucket: str, key: str) -> None:
    df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    write_bytes_to_s3(df.to_csv(index=False).encode("utf-8"), bucket, key)


def _fmt(n: int) -> str:
    return f"{int(n):,}"


def build_figure(summary: dict[str, int], threshold_ft: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    title = (
        "Indiana Bridge Coverage - USGS Stations, NWM Streams & Scour Critical Status\n"
        f"{int(threshold_ft)} ft proximity threshold  |  Indiana State Plane East (EPSG:2965)"
    )
    ax.set_title(title, fontsize=15, pad=16)

    outer = patches.FancyBboxPatch(
        (0.35, 0.55), 9.55, 8.55,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=2.2, edgecolor="#4c4c4c", facecolor="#f7f7f7",
    )
    water = patches.FancyBboxPatch(
        (1.25, 1.55), 7.8, 6.9,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=2.2, edgecolor="#2458ad", facecolor="#eaf3ff",
        linestyle="--", alpha=0.85,
    )
    nwm = patches.Ellipse(
        (5.0, 5.05), 5.1, 4.75,
        linewidth=2.2, edgecolor="#4ec36b", facecolor="#91df9f", alpha=0.48,
    )
    scour = patches.Ellipse(
        (5.2, 4.45), 3.45, 3.25,
        linewidth=2.2, edgecolor="#ef565a", facecolor="#f4a0a1", alpha=0.50,
    )
    usgs = patches.Ellipse(
        (4.08, 5.78), 1.85, 1.68,
        linewidth=2.2, edgecolor="#4696f6", facecolor="#9ac7ff", alpha=0.55,
    )

    for patch in [outer, water, nwm, scour, usgs]:
        ax.add_patch(patch)

    ax.text(5.1, 8.78, f"All bridges  (n = {_fmt(summary['all_bridges'])})",
            ha="center", va="top", fontsize=16, fontweight="bold", color="#2b2b2b")
    ax.text(5.1, 8.0, f"Bridges over waterways  (n = {_fmt(summary['waterway_bridges'])})",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#2458ad")

    ax.text(5.0, 7.0, "NWM streams", ha="center", va="center", fontsize=13,
            color="#4ec36b", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    ax.text(4.14, 6.42, "USGS\nstations", ha="center", va="center", fontsize=10.5,
            color="#4696f6", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))
    ax.text(5.2, 2.45, "Scour critical", ha="center", va="center", fontsize=12.5,
            color="#e4575b", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))

    ax.text(6.55, 5.63, _fmt(summary["nwm_only_waterway"]), fontsize=15, fontweight="bold")
    ax.text(5.8, 4.05, _fmt(summary["nwm_scour_not_usgs_waterway"]), fontsize=15, fontweight="bold")
    ax.text(4.55, 4.98, _fmt(summary["nwm_scour_usgs_waterway"]), fontsize=13, fontweight="bold")
    ax.text(3.38, 6.33, _fmt(summary["nwm_usgs_not_scour_waterway"]), fontsize=14, fontweight="bold")

    ax.text(1.95, 2.45, f"No coverage\nn = {_fmt(summary['no_coverage_waterway'])}",
            ha="center", va="center", fontsize=11, color="#666666", style="italic")
    ax.text(0.55, 0.88,
            f"Not over waterways (NBI 113 outside 0-8)\nn = {_fmt(summary['not_waterway'])}",
            ha="left", va="bottom", fontsize=11, color="#555555", style="italic")

    legend_x = 10.45
    ax.add_patch(patches.FancyBboxPatch(
        (legend_x, 7.0), 3.15, 2.55,
        boxstyle="round,pad=0.08,rounding_size=0.04",
        linewidth=1.0, edgecolor="#cccccc", facecolor="white",
    ))
    ax.text(legend_x + 1.58, 9.35, "Set definitions", ha="center", fontsize=11.5)

    legend_items = [
        ("#91df9f", "#4ec36b",
         f"Within {int(threshold_ft)} ft of NWM stream\n(n = {_fmt(summary['nwm_waterway'])} bridges over waterways)"),
        ("#f4a0a1", "#ef565a",
         f"Scour critical (rating 0-3)\n(n = {_fmt(summary['scour_critical_waterway'])} bridges over waterways)"),
        ("#9ac7ff", "#4696f6",
         f"Nearest bridge to active USGS station\n(n = {_fmt(summary['usgs_waterway'])} bridges over waterways)"),
    ]
    y = 8.88
    for fc, ec, label in legend_items:
        ax.add_patch(patches.Rectangle((legend_x + 0.15, y - 0.12), 0.22, 0.11,
                                       facecolor=fc, edgecolor=ec, alpha=0.8))
        ax.text(legend_x + 0.5, y - 0.03, label, ha="left", va="center", fontsize=10.3)
        y -= 0.43

    ax.text(
        0.5, 0.32,
        "USGS coverage assigns each active 2026 streamflow station to its nearest bridge. "
        "NWM coverage uses true flowline distance.",
        fontsize=9.5, color="#777777", style="italic",
    )
    fig.tight_layout()
    return fig


def summarize(flags: pd.DataFrame) -> dict[str, int]:
    water = flags["over_waterway"]
    scour = flags["scour_critical"]
    nwm = flags["nwm_covered"]
    usgs = flags["usgs_covered"]

    return {
        "all_bridges": len(flags),
        "waterway_bridges": int(water.sum()),
        "not_waterway": int((~water).sum()),
        "scour_critical_waterway": int((water & scour).sum()),
        "nwm_waterway": int((water & nwm).sum()),
        "usgs_waterway": int((water & usgs).sum()),
        "no_coverage_waterway": int((water & ~nwm & ~usgs).sum()),
        "nwm_only_waterway": int((water & nwm & ~scour & ~usgs).sum()),
        "usgs_only_waterway": int((water & usgs & ~nwm & ~scour).sum()),
        "nwm_usgs_not_scour_waterway": int((water & nwm & usgs & ~scour).sum()),
        "nwm_scour_not_usgs_waterway": int((water & nwm & scour & ~usgs).sum()),
        "nwm_scour_usgs_waterway": int((water & nwm & scour & usgs).sum()),
        "scour_not_nwm_not_usgs_waterway": int((water & scour & ~nwm & ~usgs).sum()),
        "usgs_scour_not_nwm_waterway": int((water & usgs & scour & ~nwm).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bucket", help="Override config aws.output_bucket")
    ap.add_argument("--prefix", help="Override config aws.output_prefix, e.g. v1 or v1/")
    ap.add_argument("--bridge-key", default=BRIDGE_KEY_DEFAULT,
                    help="S3 key relative to output_prefix for bridge CSV")
    ap.add_argument("--lat-col")
    ap.add_argument("--lon-col")
    ap.add_argument("--scour-col")
    ap.add_argument("--id-col")
    ap.add_argument("--threshold-ft", type=float, default=200.0)
    ap.add_argument("--nwm-flowlines",
                    help="Optional local or s3:// NWM/NHDPlus flowline file with COMID geometry")
    ap.add_argument("--nldi-workers", type=int, default=8)
    ap.add_argument("--nldi-timeout", type=int, default=30)
    ap.add_argument("--nldi-retries", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bucket = args.bucket or cfg["aws"]["output_bucket"]
    prefix = _normalize_prefix(args.prefix or cfg["aws"]["output_prefix"])

    bridge_key = _resolve_bridge_key(bucket, prefix, args.bridge_key)
    log.info("Reading bridges from s3://%s/%s", bucket, bridge_key)
    bridges = load_bridges(bucket, bridge_key, args.lat_col, args.lon_col, args.scour_col, args.id_col)
    log.info("Loaded %d bridges; %d over waterways; %d scour critical",
             len(bridges), int(bridges["over_waterway"].sum()), int(bridges["scour_critical"].sum()))

    stations = read_active_usgs_stations(bucket, prefix)
    usgs_bridge_ids = assign_usgs_to_nearest_bridge(bridges, stations)

    if args.nwm_flowlines:
        log.info("Computing NWM coverage from supplied flowlines: %s", args.nwm_flowlines)
        nwm_cov = cover_by_flowline_file(bridges, args.nwm_flowlines, args.threshold_ft)
    else:
        log.info("No --nwm-flowlines supplied; using NLDI nearest-COMID fallback")
        nwm_cov = cover_by_nldi(
            bridges,
            threshold_ft=args.threshold_ft,
            max_workers=args.nldi_workers,
            timeout=args.nldi_timeout,
            retries=args.nldi_retries,
        )

    flags = bridges.drop(columns=["geometry"]).copy()
    flags["bridge_id"] = flags["bridge_id"].astype(str)
    flags["usgs_covered"] = flags["bridge_id"].isin(usgs_bridge_ids)
    flags = flags.merge(nwm_cov, on="bridge_id", how="left")
    flags["nwm_covered"] = flags["nwm_covered"].fillna(False).astype(bool)

    summary = summarize(flags)
    summary["active_usgs_stations_2026"] = int(stations["site_no"].nunique())
    summary["usgs_nearest_bridges_all"] = int(len(usgs_bridge_ids))
    for key, value in summary.items():
        log.info("%-35s %s", key, f"{value:,}")

    write_parquet_to_s3(flags, bucket, f"{prefix}{FLAGS_KEY}")
    write_summary_csv(summary, bucket, f"{prefix}{SUMMARY_KEY}")

    fig = build_figure(summary, args.threshold_ft)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}")

    log.info("Wrote s3://%s/%s%s", bucket, prefix, FLAGS_KEY)
    log.info("Wrote s3://%s/%s%s", bucket, prefix, SUMMARY_KEY)
    log.info("Wrote s3://%s/%s%s", bucket, prefix, FIG_KEY)


if __name__ == "__main__":
    main()
