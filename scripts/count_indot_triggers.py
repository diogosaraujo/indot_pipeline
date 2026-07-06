"""count_indot_triggers.py

Totals for the CURRENT INDOT trigger (08d output, fixed 24 h >= 2.5 in):
  • number of Q10 / Q50 / Q100 flood events (n_flow_events)
  • inspections triggered = alarms = TP + FP
      TP = correct inspection, FP = needless inspection, FN = flood with no alarm
  • inspections per year (network) = Σ (TP+FP)_i / T_i,  T_i = n_common_hours_i/8766

Usage:
    python scripts/count_indot_triggers.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

from utils import load_config

INDOT_KEY = "analysis/event_confusion_matrix_indot.parquet"
FLOW_RPS = [10, 50, 100]
HOURS_PER_YEAR = 8766.0


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{prefix}{INDOT_KEY}")
    df = pq.read_table(io.BytesIO(obj["Body"].read()),
                       columns=["site_no", "flow_rp_yr", "tp", "fp", "fn",
                                "n_flow_events", "n_common_hours"]).to_pandas()
    df["site_no"] = df["site_no"].astype(str)
    df["alarms"] = df["tp"] + df["fp"]
    df["station_years"] = df["n_common_hours"] / HOURS_PER_YEAR
    print(f"Stations in output: {df['site_no'].nunique()}   "
          f"(fixed 24 h >= 2.5 in, nearest ISD/GHCNh)\n")

    print(f"{'target':>6} {'events':>7} {'inspections':>12} {'  TP':>6} {'  FP':>6} "
          f"{'insp/yr(net)':>13}")
    for rp in FLOW_RPS:
        sub = df[df["flow_rp_yr"] == rp]
        if sub.empty:
            print(f"  Q{rp:<4d}  (no rows)")
            continue
        events = int(sub.drop_duplicates("site_no")["n_flow_events"].sum())
        tp, fp = int(sub["tp"].sum()), int(sub["fp"].sum())
        insp = tp + fp
        insp_yr = float((sub["alarms"] / sub["station_years"]).sum())
        print(f"  Q{rp:<4d} {events:>7d} {insp:>12d} {tp:>6d} {fp:>6d} {insp_yr:>13.1f}")

    print("\ninspections = TP + FP (alarms fired);  FN floods raise no alarm.")


if __name__ == "__main__":
    main()
