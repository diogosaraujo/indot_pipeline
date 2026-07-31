"""p01 — per over-water bridge: crossing COMID + Kirpich Tc.

  * COMID  — the flags-parquet `comid` if present, else NLDI comid/position
             (deduplicated per ~110 m cell for the bridges missing one).
  * Tc     — Kirpich(1940) on the upstream main stem navigated from the COMID
             (NLDI UM flowlines) with a 10-85 channel slope from USGS 3DEP,
             exactly as scripts/03b_basin_characteristics.py. Because every
             bridge on the same reach shares the same upstream main stem, Tc is
             computed ONCE PER UNIQUE COMID and mapped back to its bridges.

NLDI (api.water.usgs.gov) rate-limits aggressively (HTTP 429). All NLDI calls go
through a single global rate limiter (`--nldi-spacing`, default 0.5 s ≈ 2 req/s)
with Retry-After-aware backoff. Everything is checkpointed/resumable:
    monitor/precompute/comid_by_cell.parquet   cell -> comid          (resolution cache)
    monitor/precompute/tc_by_comid.parquet     comid -> length/slope/tc (geometry cache)
    monitor/precompute/bridge_comid_tc.parquet FINAL, per bridge

Tune down if you still see 429s:  --workers 2 --nldi-spacing 1.0
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from common import config, load_over_water_bridges, load_script, pre_key
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p01")
logging.getLogger("urllib3").setLevel(logging.ERROR)

m03b = load_script("basin_char_03b", "03b_basin_characteristics.py")
NLDI = "https://api.water.usgs.gov/nldi/linked-data"

CELL_CACHE = pre_key("comid_by_cell.parquet")
TC_CACHE = pre_key("tc_by_comid.parquet")
OUT = pre_key("bridge_comid_tc.parquet")


class RateLimiter:
    """Global request pacer: hands each caller a future time slot spaced by
    `spacing` seconds, then sleeps outside the lock so threads don't pile up."""
    def __init__(self, spacing: float):
        self.spacing = spacing
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            slot = max(time.monotonic(), self._next)
            self._next = slot + self.spacing
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def nldi_get(url: str, params: dict, limiter: RateLimiter, timeout: int, attempts: int = 6):
    """GET with global pacing + Retry-After-aware 429 backoff. None on give-up."""
    for a in range(1, attempts + 1):
        limiter.wait()
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep(min(60.0, 3.0 * 2 ** (a - 1)))
            continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            time.sleep(min(60.0, float(ra) if ra and ra.isdigit() else 3.0 * 2 ** (a - 1)))
            continue
        if not r.ok:
            return None
        return r
    return None


def resolve_comid(lat: float, lon: float, lim: RateLimiter, timeout: int):
    r = nldi_get(f"{NLDI}/comid/position", {"coords": f"POINT({lon} {lat})"}, lim, timeout)
    if r is None:
        return None
    feats = r.json().get("features", [])
    return int(feats[0]["properties"]["comid"]) if feats else None


class TransientError(Exception):
    """A request failed (429/network/EPQS) — retry on the next run, don't cache."""


def geometry_for_comid(comid: int, nldi: RateLimiter, epqs: RateLimiter, timeout: int):
    """(stream_length_mi, slope_ft_mi) from the UM main stem off `comid`.

    Raises TransientError on a request failure (so the caller re-tries next run);
    returns (None, None) only for a DEFINITIVE no-main-stem (headwater) result,
    which is cached so it is not retried.
    """
    r = nldi_get(f"{NLDI}/comid/{comid}/navigation/UM/flowlines", {"distance": 2000}, nldi, timeout)
    if r is None:
        raise TransientError("flowlines request failed")
    coords: list = []
    for feat in r.json().get("features", []):
        coords.extend(feat["geometry"]["coordinates"])
    if len(coords) < 2:
        return None, None          # definitive: headwater / no upstream main stem
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

    try:
        epqs.wait(); e10 = m03b._elevation_ft(*at(0.10), timeout)
        epqs.wait(); e85 = m03b._elevation_ft(*at(0.85), timeout)
    except Exception as e:  # noqa: BLE001
        raise TransientError(f"3DEP elevation failed: {e}") from e
    return total, max(abs(e85 - e10) / (0.75 * total), 0.1)


# ── Phase 1: COMID per bridge (dedup missing ones by cell) ───────────────────

def resolve_all_comids(bridges: pd.DataFrame, lim: RateLimiter, timeout: int,
                       workers: int, b: str) -> pd.Series:
    bridges = bridges.copy()
    bridges["cell"] = bridges["lat"].round(3).astype(str) + "_" + bridges["lon"].round(3).astype(str)
    have = pd.to_numeric(bridges["comid"], errors="coerce")
    missing = bridges[have.isna()]
    log.info("COMID: %d from flags, %d bridges need resolution (%d unique cells)",
             int(have.notna().sum()), len(missing), missing["cell"].nunique())

    cell_map: dict[str, int] = {}
    try:
        cache = read_parquet(b, CELL_CACHE)
        cell_map = {str(r.cell): int(r.comid) for r in cache.itertuples() if pd.notna(r.comid)}
        log.info("  resumed %d cached cell->comid", len(cell_map))
    except Exception:
        pass

    todo = (missing.groupby("cell")[["lat", "lon"]].first().reset_index())
    todo = todo[~todo["cell"].isin(cell_map)]
    if len(todo):
        rows = list(cell_map.items())
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(resolve_comid, r.lat, r.lon, lim, timeout): r.cell
                    for r in todo.itertuples()}
            for i, fut in enumerate(as_completed(futs), 1):
                c = fut.result()
                if c is not None:
                    cell_map[futs[fut]] = c
                if i % 200 == 0:
                    log.info("  [cells %d/%d] checkpoint", i, len(todo))
                    write_parquet(pd.DataFrame([{"cell": k, "comid": v} for k, v in cell_map.items()]),
                                  b, CELL_CACHE)
        write_parquet(pd.DataFrame([{"cell": k, "comid": v} for k, v in cell_map.items()]),
                      b, CELL_CACHE)

    resolved = bridges["cell"].map(cell_map)
    out = have.where(have.notna(), resolved)
    return pd.to_numeric(out, errors="coerce").astype("Int64").values


# ── Phase 2: Tc per unique COMID (cached) ────────────────────────────────────

def compute_tc(comids: list[int], nldi: RateLimiter, epqs: RateLimiter,
               timeout: int, workers: int, b: str) -> pd.DataFrame:
    done: dict[int, dict] = {}
    try:
        cache = read_parquet(b, TC_CACHE)
        done = {int(r["comid"]): r for r in cache.to_dict("records")}
        log.info("Tc: resumed %d cached COMIDs", len(done))
    except Exception:
        pass
    todo = [c for c in comids if c not in done]
    log.info("Tc: %d unique COMIDs, %d to compute", len(comids), len(todo))

    results = list(done.values())

    def work(c):
        length, slope = geometry_for_comid(c, nldi, epqs, timeout)
        rec = {"comid": int(c), "stream_length_mi": None, "slope_ft_mi": None, "tc_hr": None}
        if length and slope:
            rec.update(stream_length_mi=round(length, 4), slope_ft_mi=round(slope, 3),
                       tc_hr=round(m03b.kirpich_tc_hr(length, slope), 3))
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                log.debug("COMID %s geometry failed: %s", futs[fut], e)
            if i % 250 == 0:
                n_tc = sum(1 for r in results if pd.notna(r.get("tc_hr")))
                log.info("  [comid %d/%d] checkpoint (%d with Tc)", i, len(todo), n_tc)
                write_parquet(pd.DataFrame(results).drop_duplicates("comid"), b, TC_CACHE)

    df = pd.DataFrame(results).drop_duplicates("comid")
    write_parquet(df, b, TC_CACHE)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--nldi-spacing", type=float, default=0.5,
                    help="seconds between NLDI calls, global (raise if you see 429s)")
    ap.add_argument("--epqs-spacing", type=float, default=0.1,
                    help="seconds between USGS 3DEP elevation calls, global")
    ap.add_argument("--limit", type=int, default=None, help="first N bridges (testing)")
    args = ap.parse_args()

    b, _ = config.bucket_prefix()
    nldi = RateLimiter(args.nldi_spacing)
    epqs = RateLimiter(args.epqs_spacing)

    bridges = load_over_water_bridges()
    if args.limit:
        bridges = bridges.head(args.limit)
    log.info("Over-water bridges: %d", len(bridges))

    bridges["comid"] = resolve_all_comids(bridges, nldi, args.timeout, args.workers, b)
    uniq = sorted({int(c) for c in bridges["comid"].dropna().unique()})
    tc = compute_tc(uniq, nldi, epqs, args.timeout, args.workers, b)

    out = bridges.merge(tc, on="comid", how="left")
    out["tc_dur_hr"] = (pd.to_numeric(out["tc_hr"], errors="coerce")
                        .round().fillna(1).astype(int).clip(lower=1))
    out = out.drop(columns=[c for c in ("cell",) if c in out.columns])
    write_parquet(out.sort_values("bridge_id").reset_index(drop=True), b, OUT)
    log.info("Wrote %s: %d bridges, %d with COMID, %d with Tc (%d unique reaches)",
             OUT, len(out), int(out["comid"].notna().sum()),
             int(out["tc_hr"].notna().sum()), len(uniq))


if __name__ == "__main__":
    main()
