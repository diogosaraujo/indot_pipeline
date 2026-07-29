"""Load and normalize the precomputed bridge monitor config (warm-cached)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .s3io import read_parquet

log = logging.getLogger("monitor.catalog")

_CFG: pd.DataFrame | None = None


def load(force: bool = False) -> pd.DataFrame:
    global _CFG
    if _CFG is not None and not force:
        return _CFG
    k = config.keys()
    df = read_parquet(k["bucket"], k["config"])
    df["bridge_id"] = df["bridge_id"].astype(str)
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
    df["tc_dur_hr"] = (pd.to_numeric(df["tc_dur_hr"], errors="coerce")
                       .fillna(1).astype(int).clip(lower=1))
    df[config.SCOUR_COL] = df[config.SCOUR_COL].astype(bool)
    for rp in config.SEVERITY_RPS:
        for c in (f"P{rp}", f"Q{rp}_cfs"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    _CFG = df
    log.info("Loaded monitor config: %d bridges (%d scour-critical, %d with COMID)",
             len(df), int(df[config.SCOUR_COL].sum()), int(df["comid"].notna().sum()))
    return df


def bridge_row(cfg: pd.DataFrame, bridge_id: str) -> dict:
    r = cfg.loc[cfg["bridge_id"] == str(bridge_id)]
    if r.empty:
        raise KeyError(f"bridge_id {bridge_id} not in monitor config")
    r = r.iloc[0]
    out = {
        "bridge_id": str(bridge_id),
        "asset": str(r[config.ASSET_COL]) if config.ASSET_COL in r else str(bridge_id),
        "lat": float(r["lat"]), "lon": float(r["lon"]),
        "scour": bool(r[config.SCOUR_COL]),
        "comid": None if pd.isna(r["comid"]) else int(r["comid"]),
        "tc_dur_hr": int(r["tc_dur_hr"]),
    }
    for rp in config.SEVERITY_RPS:
        out[f"P{rp}"] = float(r[f"P{rp}"]) if f"P{rp}" in r and pd.notna(r[f"P{rp}"]) else np.nan
        out[f"Q{rp}_cfs"] = float(r[f"Q{rp}_cfs"]) if f"Q{rp}_cfs" in r and pd.notna(r[f"Q{rp}_cfs"]) else np.nan
    return out
