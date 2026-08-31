"""p02b — one-time repair of the duration labels in bridge_atlas14.parquet.

scripts/07_extract_atlas14.py aligned PFDS's unlabelled `quantiles` array against
a 20-entry label list, but Volume 2 (Ohio River Basin, which contains Indiana)
publishes only 19 durations — it has no 5-day row. The old code absorbed the
mismatch with `LABELS[-n:]`, dropping "5-min" off the front and shifting every
remaining label one step longer. Result: each stored duration_hr held the NEXT
SHORTER duration's depth, and every precipitation threshold derived from it was
8-25% too low, worst at short Tc.

NOTHING WAS LOST — only mislabelled. So this is a pure relabel, not a refetch:

    stored   1 h  ->  0.5 h (30-min)      stored  24 h  ->  12 h
    stored   2 h  ->    1 h               stored  48 h  ->  24 h
    stored   3 h  ->    2 h               stored  72 h  ->  48 h
    stored   6 h  ->    3 h               stored  96 h  ->  72 h
    stored  12 h  ->    6 h               stored 120 h  ->  96 h
    stored >= 168 h are already correct — the shift spans 1 h..120 h only.

The 0.5 h row is dropped: the trigger never accumulates under an hour, and
keeping it would make duration_hr non-integer for one anchor nobody reads.

Verified at scale against monitor/assets/atlas14_grid_24h.npz (p10), which was
itself checked against live PFDS to 0.04%. Before the relabel the point table
sits 15.1% below the raster at 24 h; after, 0.05%.

Writes the corrected table back over bridge_atlas14.parquet, keeping the
original at bridge_atlas14.pre_relabel.parquet. Re-run p04 afterwards to rebuild
the monitor config from it.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import config, pre_key
from monitor_common.s3io import read_bytes, read_parquet, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p02b")

SRC = pre_key("bridge_atlas14.parquet")
BACKUP = pre_key("bridge_atlas14.pre_relabel.parquet")

# stored duration_hr -> the TRUE duration whose depth it actually holds
REMAP = {1: 0.5, 2: 1, 3: 2, 6: 3, 12: 6, 24: 12,
         48: 24, 72: 48, 96: 72, 120: 96}
DROP_BELOW_HR = 1.0


def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    i = np.searchsorted(src, dst).clip(1, len(src) - 1)
    return np.where(np.abs(dst - src[i - 1]) <= np.abs(src[i] - dst), i - 1, i)


def _check_against_raster(df: pd.DataFrame, label: str) -> float | None:
    """Median |dev| of the table's 24-h depths vs the gridded Atlas-14 raster."""
    import io
    b = config.bucket_prefix()[0]
    try:
        z = np.load(io.BytesIO(read_bytes(b, config.keys()["atlas14_grid"])))
    except Exception as e:  # noqa: BLE001
        log.warning("  raster unavailable (%s) — skipping the %s check", e, label)
        return None
    ari, depth, lats, lons = z["ari"], z["depth"], z["lats"], z["lons"]

    s = df[(df["duration_hr"] == 24) & (df["return_period_yr"] == 100)].copy()
    s = s.drop_duplicates("cell")
    ll = s["cell"].astype(str).str.split("_", expand=True)
    s["lat"] = pd.to_numeric(ll[0], errors="coerce")
    s["lon"] = pd.to_numeric(ll[1], errors="coerce")
    s = s.dropna(subset=["lat", "lon", "depth_in"])
    if s.empty:
        return None
    ri = (len(lats) - 1) - _nearest(lats[::-1], s["lat"].to_numpy())
    ci = _nearest(lons, s["lon"].to_numpy())
    gv = depth[list(ari).index(100)][ri, ci]
    pv = s["depth_in"].to_numpy()
    ok = np.isfinite(gv) & np.isfinite(pv) & (pv > 0)
    dev = np.abs(gv[ok] - pv[ok]) / pv[ok] * 100
    log.info("  %-8s 24-h/100-yr vs raster: median %.2f%%  p95 %.2f%%  (n=%d)",
             label, float(np.median(dev)), float(np.percentile(dev, 95)), int(ok.sum()))
    return float(np.median(dev))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    b = config.bucket_prefix()[0]
    df = read_parquet(b, SRC)
    log.info("Read %s: %d rows, durations %s", SRC, len(df),
             sorted(df["duration_hr"].unique())[:8])

    before = _check_against_raster(df, "BEFORE")

    d = df.copy()
    d["duration_hr"] = d["duration_hr"].map(lambda x: REMAP.get(int(x), int(x)))
    n_drop = int((d["duration_hr"] < DROP_BELOW_HR).sum())
    d = d[d["duration_hr"] >= DROP_BELOW_HR].copy()
    d["duration_hr"] = d["duration_hr"].astype(int)
    log.info("Relabelled; dropped %d sub-hourly rows. New durations: %s",
             n_drop, sorted(d["duration_hr"].unique())[:8])

    after = _check_against_raster(d, "AFTER")
    if after is not None and after > 1.0:
        raise SystemExit(f"relabelled table still {after:.2f}% off the raster — not writing")
    if before is not None and after is not None:
        log.info("  agreement improved %.2f%% -> %.2f%%", before, after)

    need = [1, 2, 3, 6, 12, 24]
    have = set(d["duration_hr"].unique())
    missing = [x for x in need if x not in have]
    if missing:
        raise SystemExit(f"trigger anchors missing after relabel: {missing}")
    log.info("  all trigger anchors present: %s", need)

    if args.dry_run:
        log.info("dry run — nothing written")
        return
    write_parquet(df, b, BACKUP)
    log.info("Backed up original -> s3://%s/%s", b, BACKUP)
    write_parquet(d.reset_index(drop=True), b, SRC)
    log.info("Wrote s3://%s/%s  (%d rows)", b, SRC, len(d))
    log.info("NOW RE-RUN p04 to rebuild bridge_monitor_config.parquet")


if __name__ == "__main__":
    main()
