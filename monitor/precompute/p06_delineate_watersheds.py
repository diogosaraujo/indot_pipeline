"""p06 — delineate upstream watersheds per bridge COMID + SIZE the areal cost.

For each unique over-water bridge COMID, fetch the NLDI upstream-basin polygon,
compute its geodesic area, and estimate the MRMS (~1 km) pixel count it covers.

The headline output is the **total estimated pixel count across all basins**
(the "membership nnz"): that single number decides the watershed-mean runtime
architecture — upstream flow-accumulation (cheap, Lambda) vs a per-bridge
membership matrix (may exceed Lambda's 10 GB -> Fargate) — and therefore the
firm cost of the whole areal track.

Heavy on NLDI (basin polygons are large, esp. continental basins), so all calls
go through one global rate limiter; checkpointed/resumable.

Reads   monitor/precompute/bridge_comid_tc.parquet
Writes  monitor/precompute/watersheds/<comid>.geojson          (basin polygons)
        monitor/precompute/bridge_watershed_area.parquet        (comid, area_mi2, est_pixels, ok)
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from common import config, pre_key
from monitor_common.s3io import read_parquet, write_bytes, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p06")
logging.getLogger("urllib3").setLevel(logging.ERROR)

NLDI = "https://api.water.usgs.gov/nldi/linked-data"
IN = pre_key("bridge_comid_tc.parquet")
OUT = pre_key("bridge_watershed_area.parquet")
POLY_PREFIX = pre_key("watersheds/")            # + {comid}.geojson

# MRMS MultiSensor grid is ~0.01deg (~1 km). At Indiana latitude a cell is
# ~0.95 km^2 ~ 0.366 mi^2, so ~2.73 pixels per mi^2. Approximate (basins that
# extend south/east span other latitudes) but fine for a cost estimate.
MRMS_PIXELS_PER_MI2 = 2.73
_M2_PER_MI2 = 2_589_988.1


class RateLimiter:
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


def nldi_get(url: str, params: dict, lim: RateLimiter, timeout: int, attempts: int = 6):
    for a in range(1, attempts + 1):
        lim.wait()
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


def basin_area_mi2(feats: list):
    from pyproj import Geod
    from shapely.geometry import shape
    from shapely.ops import unary_union
    geoms = [shape(f["geometry"]) for f in feats if f.get("geometry")]
    if not geoms:
        return None
    g = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    m2 = abs(Geod(ellps="WGS84").geometry_area_perimeter(g)[0])
    return m2 / _M2_PER_MI2 if m2 > 0 else None


def measure(comid: int, lim: RateLimiter, timeout: int, store: bool) -> dict:
    r = nldi_get(f"{NLDI}/comid/{comid}/basin", {}, lim, timeout)
    if r is None:
        return {"comid": comid, "area_mi2": None, "est_pixels": None, "ok": False}
    try:
        area = basin_area_mi2(r.json().get("features", []))
    except Exception:  # noqa: BLE001
        area = None
    if area is None:
        return {"comid": comid, "area_mi2": None, "est_pixels": None, "ok": False}
    if store:
        write_bytes(r.content, config.bucket_prefix()[0], f"{POLY_PREFIX}{comid}.geojson",
                    content_type="application/geo+json")
    return {"comid": comid, "area_mi2": round(area, 3),
            "est_pixels": int(round(area * MRMS_PIXELS_PER_MI2)), "ok": True}


def report(df: pd.DataFrame) -> None:
    ok = df[df["area_mi2"].notna()]
    failed = df[df["area_mi2"].isna()]
    tot_px = int(ok["est_pixels"].sum()) if len(ok) else 0
    log.info("══════════════════ SIZING ══════════════════")
    log.info("Delineated %d/%d COMIDs", len(ok), len(df))
    if len(failed):
        log.warning("*** %d COMIDs UNSIZED — the totals below are a LOWER BOUND (not complete). ***",
                    len(failed))
        log.warning("*** Re-run p06 to retry them (resume keeps the sized ones). Unsized: %s ***",
                    list(int(c) for c in failed["comid"])[:40])
    if not len(ok):
        return
    log.info("Summed drainage (with nesting): %s mi2", f"{ok['area_mi2'].sum():,.0f}")
    log.info("TOTAL est MRMS pixels (membership nnz) = %s  (%.1f M)", f"{tot_px:,}", tot_px / 1e6)
    gb = tot_px * 12 / 1e9
    verdict = "EXCEEDS Lambda 10 GB -> Fargate / flow-accumulation required" if gb > 10 else "fits in Lambda"
    log.info("Naive per-bridge membership matrix ~ %.1f GB (nnz x 12 B)  ->  %s", gb, verdict)
    log.info("Basin-size distribution (area, count, pixel share):")
    for thr in [100, 1000, 5000, 20000, 100000]:
        big = ok[ok["area_mi2"] > thr]
        share = 100 * big["est_pixels"].sum() / max(tot_px, 1)
        log.info("   > %7d mi2 : %5d basins, %7.1f M pixels (%4.0f%% of total)",
                 thr, len(big), big["est_pixels"].sum() / 1e6, share)
    log.info("Largest 5 basins:")
    for r in ok.nlargest(5, "area_mi2").itertuples():
        log.info("   COMID %-10d  %s mi2  ~%s pixels",
                 r.comid, f"{r.area_mi2:,.0f}", f"{int(r.est_pixels):,}")
    if len(failed):
        log.warning("*** %d basins still UNSIZED — true totals are HIGHER than shown above. ***",
                    len(failed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120, help="basin polygons can be large/slow")
    ap.add_argument("--nldi-spacing", type=float, default=0.5)
    ap.add_argument("--no-store-polygons", action="store_true",
                    help="only measure area/pixels; don't save polygons (re-fetch needed for p07)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    b, _ = config.bucket_prefix()
    lim = RateLimiter(args.nldi_spacing)

    src = read_parquet(b, IN)
    src["comid"] = pd.to_numeric(src["comid"], errors="coerce").astype("Int64")
    comids = sorted({int(c) for c in src["comid"].dropna().unique()})
    if args.limit:
        comids = comids[:args.limit]
    log.info("Unique over-water COMIDs to delineate: %d", len(comids))

    done: dict[int, dict] = {}
    try:
        prev = read_parquet(b, OUT)
        done = {int(r["comid"]): r for r in prev.to_dict("records") if pd.notna(r.get("area_mi2"))}
        log.info("Resuming: %d already delineated", len(done))
    except Exception:
        pass

    store = not args.no_store_polygons
    todo = [c for c in comids if c not in done]
    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(measure, c, lim, args.timeout, store): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception:  # noqa: BLE001
                results.append({"comid": futs[fut], "area_mi2": None, "est_pixels": None, "ok": False})
            if i % 250 == 0:
                df = pd.DataFrame(results).drop_duplicates("comid")
                write_parquet(df, b, OUT)
                nok = int(df["area_mi2"].notna().sum())
                mpx = pd.to_numeric(df["est_pixels"], errors="coerce").fillna(0).sum() / 1e6
                log.info("[%d/%d] checkpoint (%d delineated, %.1f M pixels so far)",
                         i, len(todo), nok, mpx)

    # Retry any failures once more (2x timeout) so a transient blip on a large
    # basin can't silently drop it from the sizing (never null-and-forget).
    failed = [r["comid"] for r in results if not r.get("ok", False)]
    if failed:
        log.info("Retrying %d unsized basins at %ds timeout...", len(failed), args.timeout * 2)
        keep = [r for r in results if r.get("ok", False)]
        with ThreadPoolExecutor(max_workers=max(1, args.workers // 2)) as ex:
            futs = {ex.submit(measure, c, lim, args.timeout * 2, store): c for c in failed}
            for fut in as_completed(futs):
                try:
                    keep.append(fut.result())
                except Exception:  # noqa: BLE001
                    keep.append({"comid": futs[fut], "area_mi2": None, "est_pixels": None, "ok": False})
        results = keep

    df = pd.DataFrame(results).drop_duplicates("comid").sort_values("comid").reset_index(drop=True)
    write_parquet(df, b, OUT)
    log.info("Wrote %s", OUT)
    report(df)


if __name__ == "__main__":
    main()
