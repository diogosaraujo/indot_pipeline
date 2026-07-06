"""atlas14_map.py

Map of a NOAA Atlas 14 precipitation-frequency depth over Indiana (default:
50-yr, 24-h).  Atlas 14 here is point data at the gauge locations, so the depths
are interpolated onto a grid (linear, masked to the data hull) with the station
points overlaid.

Writes:
    s3://<bucket>/<prefix>analysis/figures/atlas14_p{rp}_{dur}h_map.{png,svg}

Usage:
    python scripts/atlas14_map.py                 # 50-yr, 24-h
    python scripts/atlas14_map.py --rp 100 --duration 24
"""
from __future__ import annotations

import argparse
import io

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import griddata

from utils import load_config, write_bytes_to_s3

ATLAS_KEY = "atlas14/precipitation_frequency.parquet"
INV_KEY   = "stations/indiana_streamflow_sites.parquet"


def _pq(bucket, key, columns=None):
    o = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(o["Body"].read()), columns=columns).to_pandas()


def _dur_label(h: int) -> str:
    return f"{h // 24}-day" if h >= 24 and h % 24 == 0 else f"{h}-h"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rp", type=int, default=50, help="return period (yr)")
    ap.add_argument("--duration", type=int, default=24, help="duration (hours)")
    args = ap.parse_args()

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]

    a = _pq(bucket, f"{prefix}{ATLAS_KEY}",
            ["site_no", "duration_hr", "return_period_yr", "depth_in"])
    a = a[(a["duration_hr"] == args.duration) & (a["return_period_yr"] == args.rp)]
    a["site_no"] = a["site_no"].astype(str)

    inv = _pq(bucket, f"{prefix}{INV_KEY}", ["site_no", "dec_lat_va", "dec_long_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    df = a.merge(inv, on="site_no").dropna(subset=["dec_lat_va", "dec_long_va", "depth_in"])
    if df.empty:
        raise SystemExit(f"No Atlas14 rows for {args.rp}-yr / {args.duration}-h.")
    lon, lat, dep = df["dec_long_va"].to_numpy(float), df["dec_lat_va"].to_numpy(float), df["depth_in"].to_numpy(float)
    print(f"{len(df)} stations | depth {dep.min():.2f}–{dep.max():.2f} in")

    pad = 0.15
    gx = np.linspace(lon.min() - pad, lon.max() + pad, 300)
    gy = np.linspace(lat.min() - pad, lat.max() + pad, 300)
    GX, GY = np.meshgrid(gx, gy)
    Z = griddata((lon, lat), dep, (GX, GY), method="linear")

    fig, ax = plt.subplots(figsize=(6, 6.5))
    cf = ax.contourf(GX, GY, Z, levels=14, cmap="YlGnBu")
    ax.scatter(lon, lat, c=dep, cmap="YlGnBu", s=16, edgecolor="0.25", linewidth=0.3, zorder=3)
    ax.set_aspect(1.0 / np.cos(np.radians(float(lat.mean()))))
    ax.set_xlim(gx.min(), gx.max()); ax.set_ylim(gy.min(), gy.max())
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"NOAA Atlas 14 — {args.rp}-yr, {_dur_label(args.duration)} "
                 f"precipitation depth, Indiana", fontsize=12)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Depth (in)")
    fig.tight_layout()

    stem = f"analysis/figures/atlas14_p{args.rp}_{args.duration}h_map"
    for ext in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=170, bbox_inches="tight")
        write_bytes_to_s3(buf.getvalue(), bucket, f"{prefix}{stem}.{ext}")
        print(f"Wrote s3://{bucket}/{prefix}{stem}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
