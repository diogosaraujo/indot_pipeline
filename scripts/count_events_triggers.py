"""count_events_triggers.py

Totals across the stations in the 08c fixed-Tc output:
  • number of Q10 / Q50 / Q100 flood events (n_flow_events, threshold-independent)
  • number of inspections triggered = alarms = TP + FP, per precip ARI threshold
    (TP = correct inspection, FP = needless inspection; FN = flood with no alarm)

Usage:
    python scripts/count_events_triggers.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

from utils import load_config

TC_KEY = "analysis/event_confusion_matrix_tc.parquet"
SOURCE = "nearest"
FLOW_RPS = [10, 50, 100]


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{prefix}{TC_KEY}")
    df = pq.read_table(io.BytesIO(obj["Body"].read()),
                       columns=["site_no", "source", "precip_rp_yr", "flow_rp_yr",
                                "tp", "fp", "fn", "n_flow_events"]).to_pandas()
    df = df[df["source"] == SOURCE].copy()
    df["site_no"] = df["site_no"].astype(str)
    print(f"Stations in output: {df['site_no'].nunique()}\n")

    # ── Flood events per target (n_flow_events is the same across precip ARI) ──
    ev = (df.drop_duplicates(["site_no", "flow_rp_yr"])
            .groupby("flow_rp_yr")["n_flow_events"].sum())
    print("Total flood events across all stations:")
    for rp in FLOW_RPS:
        print(f"  Q{rp:<4d}: {int(ev.get(rp, 0)):5d} events")

    # ── Inspections triggered = TP + FP, per precip ARI × flow target ──────────
    df["alarms"] = df["tp"] + df["fp"]
    print("\nInspections triggered (alarms = TP + FP), by precip ARI threshold:")
    piv = df.pivot_table(index="precip_rp_yr", columns="flow_rp_yr",
                         values="alarms", aggfunc="sum").reindex(columns=FLOW_RPS)
    piv.columns = [f"Q{c}" for c in piv.columns]
    print(piv.astype(int).to_string())

    # ── Hits vs needless (TP / FP split) for context ──────────────────────────
    for name in ("tp", "fp"):
        p = df.pivot_table(index="precip_rp_yr", columns="flow_rp_yr",
                           values=name, aggfunc="sum").reindex(columns=FLOW_RPS)
        p.columns = [f"Q{c}" for c in p.columns]
        label = "correct inspections (TP)" if name == "tp" else "needless inspections (FP)"
        print(f"\n{label}, by precip ARI:")
        print(p.astype(int).to_string())


if __name__ == "__main__":
    main()
