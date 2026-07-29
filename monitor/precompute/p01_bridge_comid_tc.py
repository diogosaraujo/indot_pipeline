"""p01 — per over-water bridge: crossing COMID + Kirpich Tc.

For each over-water bridge:
  * COMID  — the flags-parquet `comid` if present, else NLDI comid/position.
  * Tc     — Kirpich(1940) on the upstream main stem navigated from the COMID
             (NLDI UM flowlines) with a 10-85 channel slope from USGS 3DEP,
             exactly as scripts/03b_basin_characteristics.py does for gauges.

Heavy (network-bound): each bridge = 1 NLDI navigation + 2 3DEP elevation calls
(+ 1 NLDI comid/position when the COMID is missing). Expect this to run for a
few hours for all ~16.9k over-water bridges; it is fully checkpointed/resumable.

Writes  s3://<bucket>/<prefix>monitor/precompute/bridge_comid_tc.parquet
    bridge_id, Asset Name, lat, lon, comid, over_waterway, scour_critical,
    stream_length_mi, slope_ft_mi, tc_hr, tc_dur_hr
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from common import config, load_over_water_bridges, load_script, pre_key
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p01")

m03b = load_script("basin_char_03b", "03b_basin_characteristics.py")
NLDI = "https://api.water.usgs.gov/nldi/linked-data"
OUT = pre_key("bridge_comid_tc.parquet")


def resolve_comid(lat: float, lon: float, timeout: int) -> int | None:
    r = requests.get(f"{NLDI}/comid/position", params={"coords": f"POINT({lon} {lat})"},
                     timeout=timeout)
    r.raise_for_status()
    feats = r.json().get("features", [])
    return int(feats[0]["properties"]["comid"]) if feats else None


def channel_geometry_from_comid(comid: int, timeout: int):
    """(stream_length_mi, slope_ft_mi) from the UM main stem navigated off a COMID."""
    url = f"{NLDI}/comid/{comid}/navigation/UM/flowlines"
    r = requests.get(url, params={"distance": 2000}, timeout=timeout)
    r.raise_for_status()
    coords: list = []
    for feat in r.json().get("features", []):
        coords.extend(feat["geometry"]["coordinates"])
    if len(coords) < 2:
        return None, None
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + m03b._haversine_mi(*coords[i - 1], *coords[i]))
    total = cum[-1]
    if total <= 0:
        return None, None

    def at(frac):
        target = frac * total
        for i in range(1, len(cum)):
            if cum[i] >= target:
                return coords[i]
        return coords[-1]

    e10 = m03b._elevation_ft(*at(0.10), timeout)
    e85 = m03b._elevation_ft(*at(0.85), timeout)
    slope = max(abs(e85 - e10) / (0.75 * total), 0.1)
    return total, slope


def process(row: dict, timeout: int) -> dict:
    from utils import RetryPolicy, with_retries
    out = dict(row)
    out.setdefault("stream_length_mi", None)
    out.setdefault("slope_ft_mi", None)
    out.setdefault("tc_hr", None)
    comid = row.get("comid")
    if comid is None or (isinstance(comid, float) and np.isnan(comid)) or pd.isna(comid):
        try:
            comid = with_retries(lambda: resolve_comid(row["lat"], row["lon"], timeout),
                                 RetryPolicy(max_attempts=3, base_delay=3.0),
                                 exceptions=(requests.RequestException,))
        except Exception as e:  # noqa: BLE001
            log.debug("%s: comid resolve failed: %s", row["bridge_id"], e)
            comid = None
    out["comid"] = None if comid is None else int(comid)

    if out["comid"] is not None:
        try:
            length_mi, slope = with_retries(
                lambda: channel_geometry_from_comid(out["comid"], timeout),
                RetryPolicy(max_attempts=3, base_delay=3.0),
                exceptions=(requests.RequestException,))
            if length_mi and slope:
                out["stream_length_mi"] = round(length_mi, 4)
                out["slope_ft_mi"] = round(slope, 3)
                out["tc_hr"] = round(m03b.kirpich_tc_hr(length_mi, slope), 3)
        except Exception as e:  # noqa: BLE001
            log.debug("%s: channel geometry failed: %s", row["bridge_id"], e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N (testing)")
    args = ap.parse_args()

    bridges = load_over_water_bridges()
    if args.limit:
        bridges = bridges.head(args.limit)
    log.info("Over-water bridges: %d", len(bridges))

    done: dict[str, dict] = {}
    b, _ = config.bucket_prefix()
    try:
        prev = read_parquet(b, OUT)
        prev["bridge_id"] = prev["bridge_id"].astype(str)
        done = {r["bridge_id"]: r for r in prev.to_dict("records")
                if pd.notna(r.get("tc_hr")) or pd.notna(r.get("comid"))}
        log.info("Resuming: %d bridges already computed", len(done))
    except Exception:
        pass

    todo = [r for r in bridges.to_dict("records") if r["bridge_id"] not in done]
    log.info("To process: %d", len(todo))

    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r, args.timeout): r["bridge_id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                log.error("%s: %s", futs[fut], e)
            if i % 250 == 0:
                log.info("[%d/%d] checkpointing (%d total)", i, len(todo), len(results))
                write_parquet(_finalize(pd.DataFrame(results)), b, OUT)

    out = _finalize(pd.DataFrame(results))
    write_parquet(out, b, OUT)
    log.info("Wrote %s: %d bridges, %d with Tc, %d with COMID",
             OUT, len(out), int(out["tc_hr"].notna().sum()), int(out["comid"].notna().sum()))


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("bridge_id").copy()
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
    df["tc_dur_hr"] = (pd.to_numeric(df["tc_hr"], errors="coerce")
                       .round().fillna(1).astype(int).clip(lower=1))
    return df.sort_values("bridge_id").reset_index(drop=True)


if __name__ == "__main__":
    main()
