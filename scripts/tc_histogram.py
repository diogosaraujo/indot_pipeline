"""tc_histogram.py

Distribution of the Kirpich time of concentration (Tc) across the streamflow
stations used in the 08c fixed-Tc comparison.  Tc is read straight from the 08c
output (one value per station).

Writes:
    s3://<bucket>/<prefix>analysis/figures/tc_histogram.{png,svg}

Usage:
    python scripts/tc_histogram.py
"""
from __future__ import annotations

import io

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from utils import load_config, write_bytes_to_s3

TC_KEY  = "analysis/event_confusion_matrix_tc.parquet"
FIG_KEY = "analysis/figures/tc_histogram"


def main() -> None:
    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{prefix}{TC_KEY}")
    df = pq.read_table(io.BytesIO(obj["Body"].read()),
                       columns=["site_no", "tc_hr"]).to_pandas()
    tc = df.drop_duplicates("site_no")["tc_hr"].astype(float).dropna()
    n = len(tc)
    med = float(tc.median())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tc, bins=np.arange(0, tc.max() + 6, 6), color="#4292c6", edgecolor="white")
    ax.axvline(med, color="#b30000", ls="--", lw=1.5)
    ax.text(med, ax.get_ylim()[1] * 0.92, f" median = {med:.0f} h",
            color="#b30000", fontsize=10, va="top")
    ax.set_xlabel("Kirpich time of concentration, Tc (hours)", fontsize=11)
    ax.set_ylabel("Number of stations", fontsize=11)
    ax.set_title(f"Tc distribution — {n} streamflow stations", fontsize=12)
    fig.tight_layout()

    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=150, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{FIG_KEY}.{ext}")
        print(f"Wrote s3://{bucket}/{prefix}{FIG_KEY}.{ext}")
    plt.close(fig)
    print(f"n={n}  median={med:.1f} h  min={tc.min():.1f}  max={tc.max():.1f}")


if __name__ == "__main__":
    main()
