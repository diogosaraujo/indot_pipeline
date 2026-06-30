"""best_trigger_per_cluster.py

For the MRMS source, find the best-CSI (duration, precip_rp) trigger per basin
cluster for each flow target (Q10/Q50/Q100), with the pooled POD/FAR/CSI and the
number of flood events behind it.

Two things it answers:
  • how Q50/Q100 events are distributed across clusters (the large-basin bias);
  • whether the best trigger DURATION is stable across return periods within a
    cluster (it should be — duration ~ basin response time, not flood magnitude),
    which justifies reusing the Q10 per-cluster durations for Q50/Q100.

Usage:
    python scripts/best_trigger_per_cluster.py
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
SOURCE = "nearest"          # MRMS
FLOW_RPS = [10, 50, 100]
EPS = 1e-9

DURATION_LABELS = {
    1: "1h", 2: "2h", 3: "3h", 6: "6h", 12: "12h", 24: "1d", 48: "2d", 72: "3d",
    96: "4d", 120: "5d", 168: "7d", 240: "10d", 480: "20d", 720: "30d",
    1080: "45d", 1440: "60d",
}


def main() -> None:
    o = boto3.client("s3").get_object(
        Bucket=BUCKET, Key=PREFIX + "analysis/event_confusion_matrix.parquet")
    df = pq.read_table(io.BytesIO(o["Body"].read())).to_pandas()
    df = df[df["source"] == SOURCE]
    if "cluster" not in df.columns:
        print("No cluster column."); return

    clusters = sorted(int(c) for c in df["cluster"].dropna().unique())
    print(f"Source: {SOURCE} (MRMS)   clusters: {clusters}\n")

    for flow_rp in FLOW_RPS:
        print(f"================  Q{flow_rp} target  ================")
        print(f"{'cluster':>7} {'n_gauges':>8} {'n_events':>8}  "
              f"{'best trigger':>14}  {'CSI':>5} {'POD':>5} {'FAR':>5}")
        for c in clusters:
            sub = df[(df["flow_rp_yr"] == flow_rp) & (df["cluster"] == c)]
            if sub.empty:
                continue
            n_gauges = sub["site_no"].nunique()
            n_events = int(sub.groupby("site_no")["n_flow_events"].first().sum())

            pooled = sub.groupby(["duration_hr", "precip_rp_yr"])[["tp", "fp", "fn"]].sum()
            pooled["CSI"] = pooled["tp"] / (pooled["tp"] + pooled["fp"] + pooled["fn"] + EPS)
            pooled["POD"] = pooled["tp"] / (pooled["tp"] + pooled["fn"] + EPS)
            pooled["FAR"] = pooled["fp"] / (pooled["tp"] + pooled["fp"] + EPS)
            (dur, prp) = pooled["CSI"].idxmax()
            r = pooled.loc[(dur, prp)]
            trig = f"{DURATION_LABELS.get(dur, dur)}/P{int(prp)}"
            print(f"{c:>7} {n_gauges:>8} {n_events:>8}  {trig:>14}  "
                  f"{r['CSI']:>5.2f} {r['POD']:>5.2f} {r['FAR']:>5.2f}")
        print()


if __name__ == "__main__":
    main()
