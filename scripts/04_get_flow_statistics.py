"""04_get_flow_statistics.py

Pull published flood-frequency flows (Q2..Q500) for each Indiana streamflow
gauge from the USGS Gage Statistics Service. For gauges where the Gage
Statistics service has no entry (typically newer or short-record gauges),
fall back to the StreamStats regression-equation flow stats keyed by the
workspace_id captured in step 03.

Both pathways return the same return-period set when available:
  PK2, PK5, PK10, PK25, PK50, PK100, PK200, PK500
(PKn = peak flow with n-year recurrence interval, in cfs.)

Writes:
    s3://<bucket>/<prefix>flow_stats/per_gauge_flow_stats.parquet

Output schema (one row per gauge):
    site_no, source, Q2, Q5, Q10, Q25, Q50, Q100, Q200, Q500,
    drainage_area_mi2, regression_region
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import RetryPolicy, load_config, s3_client, with_retries, write_parquet_to_s3

log = logging.getLogger("04_flowstats")

# Gage Statistics Service: published flood-frequency for actual gages
GAGE_STATS_BASE = "https://streamstats.usgs.gov/gagestatsservices"
# Use the same StreamStats server we used for delineation
SS_HOST = "prodweba.streamstats.usgs.gov"

# Map StreamStats statistic codes / gage-stats codes to friendly column names.
# These are the codes the StreamStats stat catalog uses in IN.
RETURN_PERIOD_MAP = {
    "PK2":   "Q2",
    "PK5":   "Q5",
    "PK10":  "Q10",
    "PK25":  "Q25",
    "PK50":  "Q50",
    "PK100": "Q100",
    "PK200": "Q200",
    "PK500": "Q500",
}


def fetch_gage_stats(site_no: str, timeout: int) -> Optional[list[dict]]:
    """GET /Statistics?stationIDOrCode={site_no}&statisticGroupID=2 (peak-flow group)."""
    url = f"{GAGE_STATS_BASE}/Statistics"
    params = {"stationIDOrCode": site_no, "statisticGroupID": 2}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    payload = r.json()
    return payload if isinstance(payload, list) else payload.get("statistics")


def parse_gage_stats(stats: list[dict]) -> dict:
    """Reduce the gage-stats array to a flat dict of Qn values in cfs."""
    out = {v: None for v in RETURN_PERIOD_MAP.values()}
    for s in stats or []:
        code = (s.get("statisticCode") or s.get("code") or "").upper()
        # Try multiple shapes used across versions of the service
        if code in RETURN_PERIOD_MAP:
            value = s.get("value")
            if value is None:
                value = s.get("statisticValue")
            try:
                out[RETURN_PERIOD_MAP[code]] = float(value) if value is not None else None
            except (TypeError, ValueError):
                pass
    return out


def fetch_streamstats_regression(workspace_id: str, rcode: str, timeout: int) -> Optional[dict]:
    """Fall back to ungaged regression equations using the saved workspace ID."""
    if not workspace_id:
        return None
    url = f"https://{SS_HOST}/streamstatsservices/flowstatistics.json"
    params = {
        "rcode": rcode,
        "workspaceID": workspace_id,
        "includeflowtypes": "true",
    }
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except ValueError:
        return None
    out = {v: None for v in RETURN_PERIOD_MAP.values()}
    drainage_area = None
    region = None
    for region_block in payload or []:
        region = region_block.get("RegressionRegions", [{}])[0].get("Name", region)
        for r_eq in region_block.get("RegressionRegions", []):
            for stat in r_eq.get("Results", []):
                code = (stat.get("code") or "").upper()
                if code in RETURN_PERIOD_MAP:
                    val = stat.get("Value")
                    try:
                        out[RETURN_PERIOD_MAP[code]] = float(val) if val is not None else None
                    except (TypeError, ValueError):
                        pass
    return {"flows": out, "drainage_area_mi2": drainage_area, "regression_region": region}


def process_site(site_no: str, workspace_id: str, cfg: dict) -> dict:
    timeout = cfg["streamstats"]["request_timeout_sec"]
    rcode = cfg["streamstats"]["rcode"]
    record = {"site_no": site_no, "source": None, "drainage_area_mi2": None,
              "regression_region": None}
    record.update({v: None for v in RETURN_PERIOD_MAP.values()})

    # 1) Try Gage Statistics Service first (published, gauge-derived).
    try:
        stats = with_retries(
            lambda: fetch_gage_stats(site_no, timeout),
            RetryPolicy(max_attempts=4, base_delay=4.0),
            exceptions=(requests.RequestException,),
        )
    except Exception as e:
        log.warning("Gage stats failed for %s: %s", site_no, e)
        stats = None

    if stats:
        flows = parse_gage_stats(stats)
        if any(v is not None for v in flows.values()):
            record["source"] = "gage_stats"
            record.update(flows)
            return record

    # Regression fallback via streamstatsservices is disabled —
    # that endpoint was deprecated 2026-01-30 and no longer responds.
    # Step 03 now uses NLDI which does not produce workspace IDs.
    return record


def read_workspace_index(bucket: str, prefix: str) -> pd.DataFrame:
    obj = s3_client().get_object(
        Bucket=bucket, Key=f"{prefix}watersheds/workspace_index.parquet"
    )
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    ws = read_workspace_index(bucket, prefix)

    records = []
    max_workers = cfg["streamstats"]["max_concurrent"]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(process_site, str(row["site_no"]), row.get("workspace_id", ""), cfg): row["site_no"]
            for _, row in ws.iterrows()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                log.error("Failed: %s", e)
                continue
            records.append(rec)
            if i % 25 == 0:
                log.info("[%d/%d] %s source=%s",
                         i, len(futs), rec["site_no"], rec["source"])

    df = pd.DataFrame.from_records(records)
    write_parquet_to_s3(df, bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    log.info("Done. Source counts: %s", df["source"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
