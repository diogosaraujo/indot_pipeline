"""Shared helpers for the one-time precompute (run on EC2 with the full env).

These scripts reuse the numbered pipeline modules (03b, 04c, 07) by loading them
the same way scripts/visualize_bridge_event.py does, and write per-bridge
threshold tables under s3://<bucket>/<prefix>monitor/precompute/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
MONITOR = REPO / "monitor"

for p in (str(MONITOR), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from monitor_common import config  # noqa: E402
from monitor_common.s3io import read_parquet, write_parquet  # noqa: E402


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pre_key(name: str) -> str:
    _, prefix = config.bucket_prefix()
    return f"{prefix}monitor/precompute/{name}"


def load_over_water_bridges() -> "pd.DataFrame":
    """The over-water bridge set with id / lat / lon / flags / comid."""
    import pandas as pd
    b, p = config.bucket_prefix()
    df = read_parquet(b, f"{p}analysis/bridge_coverage/bridge_coverage_flags.parquet")
    df["bridge_id"] = df["bridge_id"].astype(str) if "bridge_id" in df.columns \
        else df[config.ASSET_COL].astype(str)
    df = df[df[config.WATERWAY_COL].astype(bool)].copy()
    df["lat"] = pd.to_numeric(df[config.LAT_COL], errors="coerce")
    df["lon"] = pd.to_numeric(df[config.LON_COL], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
    else:
        df["comid"] = pd.array([pd.NA] * len(df), dtype="Int64")
    keep = ["bridge_id", config.ASSET_COL, "lat", "lon", "comid",
            config.WATERWAY_COL, config.SCOUR_COL]
    return df[[c for c in keep if c in df.columns]]
