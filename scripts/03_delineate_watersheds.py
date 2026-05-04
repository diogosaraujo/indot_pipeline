"""03_delineate_watersheds.py

Delineate the contributing watershed for every Indiana streamflow gauge
using the USGS NLDI (Network Linked Data Index) basin endpoint.

The NLDI basin endpoint accepts a USGS site number directly and returns
a GeoJSON FeatureCollection containing the upstream drainage-area polygon.
It replaced the legacy StreamStats delineation workflow after the
`streamstatsservices` API was deprecated on 2026-01-30.

NLDI endpoint used:
    GET https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site_no}/basin

Writes:
    s3://<bucket>/<prefix>watersheds/per_gauge/{site_no}.geojson
    s3://<bucket>/<prefix>watersheds/workspace_index.parquet
    (workspace_id column is empty — retained for schema compatibility with step 04)
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

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite"


class WatershedNotFound(Exception):
    """Raised when NLDI has no basin for the given site."""


def delineate_one(site_no: str, timeout: int) -> dict:
    """Fetch upstream basin polygon from NLDI. Returns a GeoJSON FeatureCollection."""
    url = f"{NLDI_BASE}/USGS-{site_no}/basin"
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        raise WatershedNotFound(f"NLDI returned 404 for site {site_no}")
    r.raise_for_status()
    return r.json()


def extract_polygon_geojson(fc: dict) -> Optional[dict]:
    """Return the first Feature from the NLDI FeatureCollection, or None."""
    features = fc.get("features") or []
    return features[0] if features else None


def process_site(row: pd.Series, cfg: dict) -> tuple[str, str, str]:
    site_no = str(row["site_no"])
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    key = f"{prefix}watersheds/per_gauge/{site_no}.geojson"

    if s3_object_exists(bucket, key):
        return site_no, "skipped", ""

    def _call():
        return delineate_one(site_no, cfg["streamstats"]["request_timeout_sec"])

    try:
        fc = with_retries(
            _call,
            RetryPolicy(
                max_attempts=cfg["streamstats"]["retries"],
                base_delay=5.0,
                max_delay=120.0,
            ),
            exceptions=(requests.RequestException,),
        )
    except WatershedNotFound as e:
        log.warning("Site %s skipped: %s", site_no, e)
        return site_no, "not_found", ""

    feature = extract_polygon_geojson(fc)
    if feature is None:
        return site_no, "no_polygon", ""

    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(feature).encode(),
        ContentType="application/geo+json",
    )
    return site_no, "ok", ""


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
    log.info("Delineating watersheds for %d gauges via NLDI", len(inv))

    workspace_records = []
    max_workers = cfg["streamstats"]["max_concurrent"]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(process_site, row, cfg): str(row["site_no"])
            for _, row in inv.iterrows()
        }
        done = 0
        for fut in as_completed(futs):
            site = futs[fut]
            try:
                site_no, status, ws = fut.result()
                workspace_records.append(
                    {"site_no": site_no, "status": status, "workspace_id": ws}
                )
                done += 1
                if done % 25 == 0:
                    log.info("[%d/%d] last: %s -> %s", done, len(inv), site_no, status)
            except Exception as e:
                workspace_records.append(
                    {"site_no": site, "status": f"error: {e}", "workspace_id": ""}
                )
                log.error("Site %s failed: %s", site, e)

    ws_df = pd.DataFrame(workspace_records)
    write_parquet_to_s3(ws_df, bucket, f"{prefix}watersheds/workspace_index.parquet")
    status_counts = ws_df["status"].value_counts().to_dict()
    not_found = status_counts.get("not_found", 0)
    if len(ws_df) and not_found / len(ws_df) > 0.5:
        log.warning(
            "High NLDI not_found rate: %d/%d. Non-standard site IDs (groundwater, "
            "coastal) are expected to be missing — stream gages should delineate.",
            not_found,
            len(ws_df),
        )
    log.info("Done. Status counts: %s", status_counts)


if __name__ == "__main__":
    main()
