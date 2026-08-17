"""e02 — NHDPlus flowline geometry for every bridge COMID, flattened for plotting.

The NWM panels draw the river network itself, not dots at reach outlets, so the
map shows the flooding rivers against the quiet ones. That needs a polyline per
COMID. scripts/nwm_comid_geojson.py already does this for the ~106 gauge-matched
reaches; this is the same idea over the ~6.9 k bridge reaches.

Stored flattened (comid, part_id, lon, lat) rather than as GeoJSON so the
consumers need no geopandas/shapely — same rationale as monitor p07.

NLDI rate-limits hard (HTTP 429), so every call goes through one global spacing
limiter, exactly as monitor/precompute/p01 learned to do. Resumable: COMIDs
already stored are skipped, and definitive failures are recorded so a rerun
doesn't retry them forever.

Writes  episode/flowlines.parquet      comid, part_id, lon, lat
        episode/flowlines_status.parquet  comid, ok, npts

Usage:
    python episode/e02_flowlines.py                    # ~1 h for 6.9 k reaches
    python episode/e02_flowlines.py --spacing 1.0 --workers 2      # if 429s
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

from common import bucket, ep_key, load_config
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e02")
logging.getLogger("urllib3").setLevel(logging.ERROR)

NLDI = "https://api.water.usgs.gov/nldi/linked-data/comid"
OUT = ep_key("flowlines.parquet")
STATUS = ep_key("flowlines_status.parquet")


class RateLimiter:
    """One global spacing gate — NLDI 429s hard under unthrottled fan-out."""

    def __init__(self, spacing: float):
        self.spacing = spacing
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            slot = max(time.monotonic(), self._next)
            self._next = slot + self.spacing
        d = slot - time.monotonic()
        if d > 0:
            time.sleep(d)

    def back_off(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


def fetch_one(comid: int, lim: RateLimiter, timeout: int, tries: int = 3):
    """Flowline vertices for one COMID, as a list of (lon, lat) per part."""
    for attempt in range(tries):
        lim.wait()
        try:
            r = requests.get(f"{NLDI}/{int(comid)}", timeout=timeout)
        except requests.RequestException:
            continue
        if r.status_code == 429:
            lim.back_off(float(r.headers.get("Retry-After", 5)))
            continue
        if r.status_code == 404:
            return []                       # definitive: no geometry for this comid
        if not r.ok:
            continue
        try:
            feats = r.json().get("features", [])
        except ValueError:
            continue
        parts = []
        for f in feats:
            g = f.get("geometry") or {}
            if g.get("type") == "LineString":
                parts.append(g["coordinates"])
            elif g.get("type") == "MultiLineString":
                parts.extend(g["coordinates"])
        return parts
    raise RuntimeError("retries exhausted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.5, help="seconds between NLDI calls")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N comids")
    args = ap.parse_args()

    cfg = load_config()
    comids = sorted(cfg["comid"].dropna().astype(np.int64).unique().tolist())
    if args.limit:
        comids = comids[:args.limit]

    done_rows, done_ids = [], set()
    try:
        prev = read_parquet(bucket(), OUT)
        done_rows.append(prev)
        done_ids |= set(prev["comid"].unique().tolist())
        log.info("Resuming: %d COMIDs already stored", len(done_ids))
    except Exception:  # noqa: BLE001
        pass
    try:                                      # don't retry definitive 404s forever
        st = read_parquet(bucket(), STATUS)
        done_ids |= set(st.loc[~st["ok"], "comid"].tolist())
    except Exception:  # noqa: BLE001
        st = None

    todo = [c for c in comids if c not in done_ids]
    log.info("Fetching %d of %d COMIDs at %.2fs spacing (~%.0f min)",
             len(todo), len(comids), args.spacing,
             len(todo) * args.spacing / max(args.workers, 1) / 60)

    lim = RateLimiter(args.spacing)
    rows, status = [], []

    def flush():
        if not rows:
            return
        df = pd.concat(done_rows + [pd.DataFrame(rows)], ignore_index=True)
        write_parquet(df.drop_duplicates(["comid", "part_id", "lon", "lat"]), bucket(), OUT)
        write_parquet(pd.DataFrame(status), bucket(), STATUS)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, c, lim, args.timeout): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                parts = fut.result()
            except Exception:  # noqa: BLE001
                status.append(dict(comid=c, ok=False, npts=0)); continue
            npts = 0
            for pid, coords in enumerate(parts):
                for lon, lat in coords:
                    rows.append(dict(comid=c, part_id=pid, lon=float(lon), lat=float(lat)))
                    npts += 1
            status.append(dict(comid=c, ok=bool(parts), npts=npts))
            if i % 500 == 0:
                nok = sum(1 for s in status if s["ok"])
                log.info("[%d/%d] %d ok, %d vertices", i, len(todo), nok, len(rows))
                flush()

    flush()
    nok = sum(1 for s in status if s["ok"])
    log.info("Done: %d/%d COMIDs have geometry, %d vertices total",
             nok, len(todo), len(rows))
    if len(todo) - nok:
        log.warning("%d COMIDs have no flowline — they will be absent from the "
                    "river network (not silently drawn at zero).", len(todo) - nok)


if __name__ == "__main__":
    main()
