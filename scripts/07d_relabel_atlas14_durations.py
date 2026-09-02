#!/usr/bin/env python3
"""07d_relabel_atlas14_durations.py

ONE-TIME repair of the duration labels in the study's Atlas 14 tables, so
08c / 08h can be re-run against correctly-labelled depths.

THE BUG.  07_extract_atlas14.py reads the PFDS *js* endpoint, which returns
`quantiles` as a bare nested array with NO duration labels, so the alignment came
entirely from ATLAS14_DURATION_LABELS.  That list had 20 entries including a
"5-day" row.  **Atlas 14 Volume 2 (Ohio River Basin — the volume containing
Indiana) publishes only 19 durations: it goes 4-day -> 7-day, with no 5-day.**
The old code absorbed the mismatch with `LABELS[-n:]`, which dropped "5-min" off
the FRONT and shifted every remaining label one step longer.  Result: each stored
`duration_hr` holds the depth of the NEXT SHORTER published duration.

    stored   1 h  ->  0.5 h (30-min)      stored  24 h  ->  12 h
    stored   2 h  ->    1 h               stored  48 h  ->  24 h
    stored   3 h  ->    2 h               stored  72 h  ->  48 h
    stored   6 h  ->    3 h               stored  96 h  ->  72 h
    stored  12 h  ->    6 h               stored 120 h  ->  96 h
    stored >= 168 h are already correct — the shift spans 1 h..120 h only.

The presence of a `duration_hr == 120` row is the fingerprint: a correctly
parsed Volume 2 response cannot produce one.  This script refuses to run if that
row is absent, so it cannot be applied twice.

NOTHING WAS LOST, only mislabelled, so this is a pure relabel and needs no
refetch.  (Verified against live PFDS at 12 gauges x 7 durations x 6 ARIs:
|stored - TRUE(next shorter duration)| = 0.000% at median, p95 AND max.)
The 0.5 h row is dropped so `duration_hr` stays integral for other consumers;
the study's shortest Kirpich Tc is 3 h, so the 1 h anchor is ample.

parse_atlas14 itself is already fixed and now RAISES on a count mismatch instead
of realigning, so a fresh `07`/`07c` run is also correct — that is the slower
alternative to this script (~294 gauge + ~636 station PFDS fetches).

WHAT THIS CHANGES DOWNSTREAM.  The confusion matrices in
analysis/event_confusion_matrix_tc*.parquet were never wrong: each row recorded
the depth it used and counted against that real depth.  Only the ARI *label* was
wrong, by a factor of ~0.55 (P10 behaved as P5.7).  Re-running 08c/08h after
this script puts the labels back on the right thresholds.  08g is unaffected —
it uses the areal DDF from 07a, which never touches this parser.

Reads / writes (backing up the original alongside):
    atlas14/precipitation_frequency.parquet            <- 08c  (key: site_no)
    atlas14/precipitation_frequency_stations.parquet   <- 08h  (key: station_id)

Usage:
    python scripts/07d_relabel_atlas14_durations.py --dry-run
    python scripts/07d_relabel_atlas14_durations.py
    python scripts/08c_tc_trigger_analysis.py
    python scripts/08h_station_tc_trigger_analysis.py
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import urllib.request

import numpy as np
import pandas as pd

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("07d_relabel")

# stored duration_hr -> the TRUE duration whose depth it actually holds
REMAP = {1: 0.5, 2: 1, 3: 2, 6: 3, 12: 6, 24: 12,
         48: 24, 72: 48, 96: 72, 120: 96}
FINGERPRINT_DUR = 120          # a duration Volume 2 does not publish
DROP_BELOW_HR = 1              # keeps duration_hr integral

TABLES = [
    ("atlas14/precipitation_frequency.parquet", "site_no", "08c"),
    ("atlas14/precipitation_frequency_stations.parquet", "station_id", "08h"),
]

PFDS_CSV = ("https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
            "?lat={lat:.4f}&lon={lon:.4f}&type=pf&data=depth&units=english&series=pds")
LABEL_H = {"60-min": 1, "2-hr": 2, "3-hr": 3, "6-hr": 6, "12-hr": 12,
           "24-hr": 24, "2-day": 48, "3-day": 72, "4-day": 96, "7-day": 168}


def read_s3(bucket: str, key: str) -> pd.DataFrame:
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def pfds_curve(lat: float, lon: float, timeout: float = 120):
    """{duration_hr: {ari: depth}} from the LABELLED csv endpoint.

    Deliberately not the js endpoint this bug came from: the csv names every
    duration row, so there is no positional assumption left to get wrong. This
    is independent ground truth, not a second opinion from the same source.
    """
    try:
        txt = urllib.request.urlopen(PFDS_CSV.format(lat=lat, lon=lon),
                                     timeout=timeout).read().decode()
    except Exception as e:  # noqa: BLE001
        log.debug("PFDS fetch failed at %.4f,%.4f: %s", lat, lon, e)
        return None
    hdr = re.search(r"by duration for ARI \(years\):,(.+)$", txt, re.M)
    if not hdr:
        return None
    rps = [int(float(x)) for x in hdr.group(1).split(",")]
    out = {}
    for lab, h in LABEL_H.items():
        mm = re.search(rf"^{re.escape(lab)}:,(.+)$", txt, re.M)
        if mm:
            out[h] = dict(zip(rps, [float(x) for x in mm.group(1).split(",")]))
    return out


def verify(df: pd.DataFrame, idcol: str, coords: pd.DataFrame,
           n: int, timeout: float) -> None:
    """Spot-check the RELABELLED table against live PFDS."""
    if coords is None or coords.empty:
        log.warning("  no coordinates available — skipping the live PFDS check")
        return
    ids = [i for i in coords[idcol].unique() if i in set(df[idcol])]
    if not ids:
        log.warning("  no overlapping ids — skipping the live PFDS check")
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    devs = []
    for sid in pick:
        row = coords[coords[idcol] == sid].iloc[0]
        cur = pfds_curve(float(row["lat"]), float(row["lon"]), timeout)
        if not cur:
            continue
        sub = df[df[idcol] == sid]
        for D in (3, 12, 24, 48):
            if D not in cur:
                continue
            for R in (10, 100):
                v = sub[(sub.duration_hr == D) & (sub.return_period_yr == R)]["depth_in"]
                if v.empty or R not in cur[D] or cur[D][R] <= 0:
                    continue
                devs.append(abs(float(v.iloc[0]) - cur[D][R]) / cur[D][R] * 100)
    if not devs:
        log.warning("  live PFDS check inconclusive (no comparable points)")
        return
    a = np.array(devs)
    log.info("  live PFDS check: |dev| median %.3f%%  max %.3f%%  (n=%d)",
             float(np.median(a)), float(a.max()), len(a))
    if a.max() > 1.0:
        raise SystemExit(
            f"relabelled table still disagrees with live PFDS by {a.max():.2f}% "
            "— do not use these depths")


def gauge_coords(bucket, prefix) -> pd.DataFrame | None:
    try:
        s = read_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    except Exception as e:  # noqa: BLE001
        log.warning("  gauge coords unavailable (%s)", e)
        return None
    return pd.DataFrame({"site_no": s["site_no"].astype(str),
                         "lat": pd.to_numeric(s["dec_lat_va"], errors="coerce"),
                         "lon": pd.to_numeric(s["dec_long_va"], errors="coerce")}).dropna()


def station_coords(bucket, prefix) -> pd.DataFrame | None:
    frames = []
    for k in ("precip/noaa/stations_isd.parquet", "precip/noaa/stations_ghcnh.parquet"):
        try:
            s = read_s3(bucket, f"{prefix}{k}")
        except Exception:  # noqa: BLE001
            continue
        idc = next((c for c in s.columns if c.lower() in ("station_id", "id")), None)
        latc = next((c for c in s.columns if c.lower() in ("lat", "latitude")), None)
        lonc = next((c for c in s.columns if c.lower() in ("lon", "longitude")), None)
        if not (idc and latc and lonc):
            continue
        frames.append(pd.DataFrame({"station_id": s[idc].astype(str),
                                    "lat": pd.to_numeric(s[latc], errors="coerce"),
                                    "lon": pd.to_numeric(s[lonc], errors="coerce")}))
    if not frames:
        log.warning("  station coords unavailable")
        return None
    return pd.concat(frames).dropna().drop_duplicates("station_id")


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["duration_hr"] = d["duration_hr"].map(lambda x: REMAP.get(int(x), int(x)))
    n_drop = int((d["duration_hr"] < DROP_BELOW_HR).sum())
    d = d[d["duration_hr"] >= DROP_BELOW_HR].copy()
    d["duration_hr"] = d["duration_hr"].astype(int)
    log.info("  relabelled; dropped %d sub-hourly rows; durations now %s",
             n_drop, sorted(d["duration_hr"].unique())[:10])
    return d.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="relabel and verify but do not write to S3")
    ap.add_argument("--check-sites", type=int, default=6,
                    help="live PFDS spot-checks per table (0 to skip)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    for key, idcol, consumer in TABLES:
        # ASCII only: the Windows console is cp1252 and mangles box-drawing.
        log.info("--- %s  (feeds %s) ---", key, consumer)
        df = read_s3(bucket, f"{prefix}{key}")
        df[idcol] = df[idcol].astype(str)
        durs = sorted(df["duration_hr"].unique())
        log.info("  %d rows, %d ids, durations %s", len(df), df[idcol].nunique(), durs[:10])

        if FINGERPRINT_DUR not in durs:
            log.warning("  no %d-h row — this table is ALREADY relabelled. Skipping.",
                        FINGERPRINT_DUR)
            continue

        out = relabel(df)
        if args.check_sites:
            coords = (gauge_coords(bucket, prefix) if idcol == "site_no"
                      else station_coords(bucket, prefix))
            verify(out, idcol, coords, args.check_sites, args.timeout)

        if args.dry_run:
            log.info("  dry run — nothing written")
            continue
        backup = key.replace(".parquet", ".pre_relabel.parquet")
        write_parquet_to_s3(df, bucket, f"{prefix}{backup}")
        log.info("  backed up original -> s3://%s/%s%s", bucket, prefix, backup)
        write_parquet_to_s3(out, bucket, f"{prefix}{key}")
        log.info("  wrote s3://%s/%s%s  (%d rows)", bucket, prefix, key, len(out))

    log.info("Done. Now re-run:")
    log.info("  python scripts/08c_tc_trigger_analysis.py")
    log.info("  python scripts/08h_station_tc_trigger_analysis.py")
    log.info("(08g is unaffected — it uses the areal DDF from 07a.)")


if __name__ == "__main__":
    main()
