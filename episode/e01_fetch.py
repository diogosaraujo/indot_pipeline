"""e01 — re-fetch MRMS + NWM for the episode from the NOAA public buckets.

The monitor prunes its own state at 48 h, so Aug 12-14 is long gone from
monitor/state/. NOAA still has it (MRMS ~30 d, NWM ~40 d), so we pull it back.

Two size decisions make this cheap enough to keep:
  * MRMS CONUS is 3500x7000 float32 = 98 MB/hr. We SUBSET to Indiana on ingest
    (~340x410 = 0.5 MB) before storing — 120 hours then costs ~65 MB instead of
    12 GB.
  * NWM channel_rt is 2.7 M reaches. We keep only the ~6.9 k bridge COMIDs, for
    both products, which is a few hundred KB per hour.

Writes  episode/mrms/{YYYYMMDDHH}.npz        arr, lats, lons  (inches)
        episode/nwm/{YYYYMMDDHH}.parquet     comid, q_ol_cms, v_ol_ms,
                                             q_aa_cms, v_aa_ms

Resumable: an hour already present in S3 is skipped, so a killed run just
continues. Run on EC2 (us-east-1) — the NOAA reads are free and in-region.

Usage:
    python episode/e01_fetch.py                 # all DAYS
    python episode/e01_fetch.py --days 2026-08-14 --workers 6
"""
from __future__ import annotations

import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import (DAYS, IN_BBOX, bucket, ep_key, hour_range, load_config,
                    mrms_key, nwm_key)
from monitor_common import config, mrms, nwm
from monitor_common.s3io import list_keys, read_bytes, write_bytes, write_parquet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e01")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)


def fetch_mrms_hour(ts: pd.Timestamp, have: set[str]) -> str:
    key = mrms_key(ts)
    if key in have:
        return "skip"
    got = mrms.read_grid(ts)
    if got is None:
        return "miss"
    arr, glat_desc, glon_asc = got
    la0, la1 = IN_BBOX["lat"]; lo0, lo1 = IN_BBOX["lon"]
    rows = np.where((glat_desc >= la0) & (glat_desc <= la1))[0]
    cols = np.where((glon_asc >= lo0) & (glon_asc <= lo1))[0]
    sub = np.asarray(arr[np.ix_(rows, cols)], dtype=np.float32)
    sub = np.where(np.isfinite(sub), sub, 0.0).astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=sub, lats=glat_desc[rows].astype(np.float32),
                        lons=glon_asc[cols].astype(np.float32))
    write_bytes(buf.getvalue(), bucket(), key, content_type="application/octet-stream")
    return "ok"


def fetch_nwm_hour(ts: pd.Timestamp, comids: np.ndarray, have: set[str]) -> str:
    key = nwm_key(ts)
    if key in have:
        return "skip"
    ol = nwm.read_comids(ts, config.NWM_PRODUCT_TRIGGER, comids)
    aa = nwm.read_comids(ts, config.NWM_PRODUCT_DISPLAY, comids)
    if ol is None and aa is None:
        return "miss"
    base = pd.DataFrame(index=pd.Index(comids, name="comid"))
    if ol is not None:
        base["q_ol_cms"] = ol["streamflow_cms"]; base["v_ol_ms"] = ol["velocity_ms"]
    if aa is not None:
        base["q_aa_cms"] = aa["streamflow_cms"]; base["v_aa_ms"] = aa["velocity_ms"]
    write_parquet(base.reset_index(), bucket(), key)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel hours; NWM pulls 2x12.6 MB per hour")
    ap.add_argument("--skip-mrms", action="store_true")
    ap.add_argument("--skip-nwm", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    comids = cfg["comid"].dropna().astype(np.int64).unique()
    log.info("Episode fetch: %d day(s), %d bridge COMIDs", len(args.days), len(comids))

    hours = [h for d in args.days for h in hour_range(d)]
    have_m = set(list_keys(bucket(), ep_key("mrms/")))
    have_n = set(list_keys(bucket(), ep_key("nwm/")))
    log.info("Already stored: %d MRMS, %d NWM hours", len(have_m), len(have_n))

    for label, fn, skip in (("MRMS", lambda t: fetch_mrms_hour(t, have_m), args.skip_mrms),
                            ("NWM", lambda t: fetch_nwm_hour(t, comids, have_n), args.skip_nwm)):
        if skip:
            log.info("%s: skipped by flag", label); continue
        tally = {"ok": 0, "skip": 0, "miss": 0, "err": 0}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, t): t for t in hours}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    tally[fut.result()] += 1
                except Exception as e:  # noqa: BLE001
                    tally["err"] += 1
                    log.warning("%s %s failed: %s", label, futs[fut], e)
                if i % 24 == 0:
                    log.info("%s %d/%d  %s", label, i, len(hours), tally)
        log.info("%s done: %s", label, tally)
        if tally["miss"]:
            log.warning("%s: %d hour(s) missing at the source — figures for those "
                        "hours will show a gap rather than silently interpolating.",
                        label, tally["miss"])


if __name__ == "__main__":
    main()
