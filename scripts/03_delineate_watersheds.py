"""03_delineate_watersheds.py

Delineate the contributing watershed for every Indiana streamflow gauge
using the USGS StreamStats web service. Each gauge is delineated by
posting (lat, lon, rcode='IN') to `watershed.geojson`. The returned
GeoJSON polygon is stored in S3.

USGS service rules we honor:
  - max 4 concurrent requests
  - sticky server affinity per workspace (we capture the workspaceID
    returned by the service and stay on the same server for any
    follow-up flow-stat call in step 04)

Writes:
    s3://<bucket>/<prefix>watersheds/per_gauge/{site_no}.geojson
    s3://<bucket>/<prefix>watersheds/workspace_index.parquet

Migration note: USGS announced sunset of `streamstatsservices` for
2026-01-30. The replacement is `ss-delineate`; swap `BASE_URL` and the
response parsing in `delineate_one()` to migrate.
"""
from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import (
    RetryPolicy,
    load_config,
    s3_client,
    s3_object_exists,
    with_retries,
    write_parquet_to_s3,
)

log = logging.getLogger("03_delineate")

# Pin to one StreamStats server so workspaceID stays valid for the flow-stat call.
# `prodweba` and `prodwebb` are identical; we pick A and stay there.
SERVER_HOST = "prodweba.streamstats.usgs.gov"


class StreamStatsWatershedNotFound(Exception):
    """Raised when StreamStats has no delineation for a gauge point."""


def delineate_one(lat: float, lon: float, rcode: str, timeout: int) -> dict:
    """One watershed delineation. Returns parsed JSON (with workspaceID + GeoJSON)."""
    url = f"https://{SERVER_HOST}/streamstatsservices/watershed.geojson"
    params = {
        "rcode": rcode,
        "xlocation": f"{lon:.6f}",
        "ylocation": f"{lat:.6f}",
        "crs": 4326,
        "includeparameters": "false",   # keep delineation lean; basin chars come in step 04
        "includeflowtypes": "false",
        "includefeatures": "true",      # we want the polygon
        "simplify": "false",
    }
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code == 404:
        raise StreamStatsWatershedNotFound(
            f"StreamStats returned 404 for lon={lon:.6f}, lat={lat:.6f}"
        )
    r.raise_for_status()
    return r.json()


def extract_polygon_geojson(payload: dict) -> Optional[dict]:
    """Pull the watershed polygon out of the StreamStats payload as a GeoJSON Feature."""
    fc = payload.get("featurecollection") or []
    for entry in fc:
        if entry.get("name") == "globalwatershed":
            return entry.get("feature")
    return None


def process_site(row: pd.Series, cfg: dict) -> tuple[str, str, str]:
    site_no = str(row["site_no"])
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    key = f"{prefix}watersheds/per_gauge/{site_no}.geojson"

    if s3_object_exists(bucket, key):
        return site_no, "skipped", ""

    def _call():
        return delineate_one(
            lat=float(row["dec_lat_va"]),
            lon=float(row["dec_long_va"]),
            rcode=cfg["streamstats"]["rcode"],
            timeout=cfg["streamstats"]["request_timeout_sec"],
        )

    try:
        payload = with_retries(
            _call,
            RetryPolicy(max_attempts=cfg["streamstats"]["retries"], base_delay=5.0, max_delay=120.0),
            exceptions=(requests.RequestException,),
        )
    except StreamStatsWatershedNotFound as e:
        log.warning("Site %s skipped: %s", site_no, e)
        return site_no, "not_found", ""
    workspace_id = payload.get("workspaceID", "")
    feature = extract_polygon_geojson(payload)
    if feature is None:
        return site_no, "no_polygon", workspace_id

    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(feature).encode(),
        ContentType="application/geo+json",
    )
    return site_no, "ok", workspace_id


def read_station_inventory(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}stations/indiana_streamflow_sites.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    inv = read_station_inventory(bucket, prefix)
    inv["dec_lat_va"] = pd.to_numeric(inv["dec_lat_va"], errors="coerce")
    inv["dec_long_va"] = pd.to_numeric(inv["dec_long_va"], errors="coerce")
    inv = inv.dropna(subset=["dec_lat_va", "dec_long_va"]).reset_index(drop=True)
    log.info("Delineating watersheds for %d gauges", len(inv))

    workspace_records = []
    max_workers = cfg["streamstats"]["max_concurrent"]  # USGS limit
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_site, row, cfg): str(row["site_no"]) for _, row in inv.iterrows()}
        done = 0
        for fut in as_completed(futs):
            site = futs[fut]
            try:
                site_no, status, ws = fut.result()
                workspace_records.append({"site_no": site_no, "status": status, "workspace_id": ws})
                done += 1
                if done % 25 == 0:
                    log.info("[%d/%d] last: %s -> %s", done, len(inv), site_no, status)
            except Exception as e:
                workspace_records.append({"site_no": site, "status": f"error: {e}", "workspace_id": ""})
                log.error("Site %s failed: %s", site, e)

    # Persist workspace_id index — required by step 04 to compute flow stats
    # against the same StreamStats server that produced the delineation.
    ws_df = pd.DataFrame(workspace_records)
    write_parquet_to_s3(ws_df, bucket, f"{prefix}watersheds/workspace_index.parquet")
    status_counts = ws_df["status"].value_counts().to_dict()
    not_found = status_counts.get("not_found", 0)
    if len(ws_df) and not_found / len(ws_df) > 0.25:
        log.warning(
            "High StreamStats not_found rate: %d/%d. If this affects most gauges, "
            "the legacy streamstatsservices endpoint may need migration to ss-delineate.",
            not_found,
            len(ws_df),
        )
    log.info("Done. Status counts: %s", status_counts)


if __name__ == "__main__":
    main()
