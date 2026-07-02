"""station_funnel.py

Reports the station funnel — how many USGS streamflow gauges we started with and
how many survived each exclusion to reach the fixed-Tc comparison (08c) — with
the real counts read from S3.  Also breaks down WHY stations were dropped.

Stages / exclusions:
    0. Inventory (01)        all Indiana streamflow gauges (param 00060)
    1. Has IV streamflow (02)
    2. flow_stats universe   stations evaluated for a flood-frequency fit
    3. LP3 fit (04b)         lp3_peak_series vs insufficient, with reasons:
                               - no_peak_record   (no USGS annual peaks)
                               - regulated_excluded (peak_cd 5/6 → Rule B)
                               - insufficient_record (< 10 clean annual peaks)
    4. Clustered (03c)       LP3-fitted AND complete basin characteristics
    5. Used in 08c           clustered AND valid Q AND Kirpich Tc AND Atlas14
                             AND streamflow AND a non-empty flow∩MRMS window

Usage:
    python scripts/station_funnel.py
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq

from utils import load_config

# Reuse 08's loaders so keys / logic match the real 08c run exactly.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TC_KEY = "analysis/event_confusion_matrix_tc.parquet"
LP3_SUMMARY_KEY = "analysis/lp3_frequency_curves/lp3_summary.csv"


def _pq(bucket, prefix, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=prefix + key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def _csv(bucket, prefix, key, **kw):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=prefix + key)
    return pd.read_csv(io.BytesIO(o["Body"].read()), **kw)


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    def hdr(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

    # 0. Inventory ------------------------------------------------------------
    inv = _pq(bucket, prefix, "stations/indiana_streamflow_sites.parquet",
              ["site_no"])
    inv_sites = set(inv["site_no"].astype(str))
    hdr("STATION FUNNEL")
    print(f"0. Inventory (01)                 : {len(inv_sites):5d}  Indiana streamflow gauges")

    # 1. Has IV streamflow ----------------------------------------------------
    try:
        sf = _pq(bucket, prefix, "streamflow/instantaneous/all_gauges_long.parquet",
                 ["site_no"])
        iv_sites = set(sf["site_no"].astype(str))
        print(f"1. Has IV streamflow (02)         : {len(iv_sites):5d}")
    except Exception as e:
        print(f"1. Has IV streamflow (02)         :   n/a  ({e})")

    # 2/3. flow_stats + LP3 outcome ------------------------------------------
    fs = _pq(bucket, prefix, "flow_stats/per_gauge_flow_stats.parquet")
    fs["site_no"] = fs["site_no"].astype(str)
    print(f"2. flow_stats universe            : {len(fs):5d}  stations evaluated (04b)")
    if "source" in fs.columns:
        hdr("3. LP3 fit (04b) — source")
        print(fs["source"].value_counts(dropna=False).to_string())
    fitted = set(fs.loc[fs.get("source") == "lp3_peak_series", "site_no"])
    print(f"\n   LP3-fitted (lp3_peak_series)   : {len(fitted):5d}")

    try:
        lp3 = _csv(bucket, prefix, LP3_SUMMARY_KEY, dtype={"site_no": str})
        print("\n   04b status breakdown (lp3_summary.csv):")
        print(lp3["status"].value_counts(dropna=False).to_string())
    except Exception as e:
        print(f"\n   lp3_summary.csv not read: {e}")

    # 4. Clustered ------------------------------------------------------------
    clusters = m.load_clusters(bucket, prefix)          # dict site_no → cluster
    clustered = set(clusters)
    hdr("4. Clustered (03c)")
    print(f"   Clustered stations             : {len(clustered):5d}")
    cl_series = pd.Series(list(clusters.values()))
    print("   per cluster:")
    print(cl_series.value_counts().sort_index().to_string())

    # 5. Used in 08c (reconstruct the exact intersection + window) ------------
    atlas14 = m.load_atlas14(bucket, prefix)
    tc_by_site = _pq(bucket, prefix, "watersheds/basin_characteristics.parquet",
                     ["site_no", "tc_hr"])
    tc_by_site["site_no"] = tc_by_site["site_no"].astype(str)
    tc_sites = set(tc_by_site.dropna(subset=["tc_hr"])["site_no"])
    q_cols = [f"Q{rp}" for rp in m.FLOW_RPS if f"Q{rp}" in fs.columns]
    has_q = set(fs.loc[fs[q_cols].notna().any(axis=1), "site_no"])
    a14_sites = set(atlas14["site_no"].astype(str))
    sf_sites = set(m.load_streamflow(bucket, prefix)["site_no"].astype(str))

    stations_all = clustered & has_q & tc_sites & a14_sites & sf_sites
    hdr("5. Used in 08c")
    print(f"   clustered                      : {len(clustered):5d}")
    print(f"   ∩ valid Q                      : {len(clustered & has_q):5d}")
    print(f"   ∩ Kirpich Tc                   : {len(clustered & has_q & tc_sites):5d}")
    print(f"   ∩ Atlas14                      : {len(clustered & has_q & tc_sites & a14_sites):5d}")
    print(f"   ∩ streamflow                   : {len(stations_all):5d}")

    # what 08c actually wrote
    try:
        used = _pq(bucket, prefix, TC_KEY, ["site_no", "flow_rp_yr"])
        used["site_no"] = used["site_no"].astype(str)
        used_sites = set(used["site_no"])
        print(f"\n   ACTUALLY USED (08c output)     : {len(used_sites):5d}  distinct stations")
        print("   by flood target:")
        for rp in m.FLOW_RPS:
            print(f"     Q{rp:<4d}: {used[used['flow_rp_yr']==rp]['site_no'].nunique():5d} stations")
        window_dropped = stations_all - used_sites
        print(f"\n   dropped by empty flow∩MRMS win : {len(window_dropped):5d}")
    except Exception as e:
        print(f"\n   08c output not read (run 08c first): {e}")

    # Clustered-but-unused reasons
    hdr("Clustered but NOT used — reason")
    dropped = clustered - stations_all
    print(f"   no valid Q                     : {len(clustered - has_q):5d}")
    print(f"   no Kirpich Tc                  : {len((clustered & has_q) - tc_sites):5d}")
    print(f"   no Atlas14                     : {len((clustered & has_q & tc_sites) - a14_sites):5d}")
    print(f"   no streamflow                  : {len((clustered & has_q & tc_sites & a14_sites) - sf_sites):5d}")
    print(f"   total clustered-but-unused     : {len(dropped):5d}  (before MRMS-window)")


if __name__ == "__main__":
    main()
