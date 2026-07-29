"""p02 — Atlas-14 precipitation-frequency depths per over-water bridge.

Reuses scripts/07_extract_atlas14.py (NOAA PFDS web service). Atlas-14 depends
only on lat/lon, so bridges are deduplicated to a ~110 m grid (round to 3 dp)
before fetching, then the depths are mapped back to every bridge in the cell.

Writes  s3://<bucket>/<prefix>monitor/precompute/bridge_atlas14.parquet
    bridge_id, duration_hr, return_period_yr, depth_in
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import config, load_over_water_bridges, load_script, pre_key
from monitor_common.s3io import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p02")

m07 = load_script("atlas14_07", "07_extract_atlas14.py")
OUT = pre_key("bridge_atlas14.parquet")


def fetch_cell(cell_id: str, lat: float, lon: float) -> pd.DataFrame | None:
    from utils import RetryPolicy, with_retries
    try:
        text = with_retries(lambda: m07.fetch_atlas14(lat, lon),
                            RetryPolicy(max_attempts=4, base_delay=5.0))
        time.sleep(0.3)                          # be gentle to PFDS
        if not text:
            return None
        df = m07.parse_atlas14(cell_id, text)    # cols: site_no, duration_hr, return_period_yr, depth_in
        return df
    except Exception as e:  # noqa: BLE001
        log.debug("cell %s fetch failed: %s", cell_id, e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bridges = load_over_water_bridges()
    if args.limit:
        bridges = bridges.head(args.limit)
    bridges["cell"] = (bridges["lat"].round(3).astype(str) + "_" +
                       bridges["lon"].round(3).astype(str))
    cells = (bridges.groupby("cell")[["lat", "lon"]].first().reset_index())
    log.info("Over-water bridges: %d  ->  %d unique Atlas-14 cells", len(bridges), len(cells))

    b, _ = config.bucket_prefix()
    have_cells: set[str] = set()
    prev = None
    try:
        prev = read_parquet(b, OUT)
        have_cells = set(prev["cell"].astype(str)) if "cell" in prev.columns else set()
        log.info("Resuming: %d cells already fetched", len(have_cells))
    except Exception:
        pass

    todo = cells[~cells["cell"].isin(have_cells)]
    frames = [prev] if prev is not None else []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_cell, r.cell, r.lat, r.lon): r.cell
                for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()
            if df is not None and not df.empty:
                df["cell"] = futs[fut]
                frames.append(df)
            if i % 200 == 0:
                log.info("[%d/%d] checkpointing", i, len(todo))
                write_parquet(pd.concat(frames, ignore_index=True), b, OUT)

    cell_ddf = pd.concat(frames, ignore_index=True).drop_duplicates(["cell", "duration_hr", "return_period_yr"])
    write_parquet(cell_ddf, b, OUT)
    log.info("Wrote %s: %d rows across %d cells",
             OUT, len(cell_ddf), cell_ddf["cell"].nunique())


if __name__ == "__main__":
    main()
