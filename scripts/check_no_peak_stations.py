"""check_no_peak_stations.py

Identify the Indiana streamflow gages that have NO published USGS annual-peak
record (the filter that discarded ~21 stations in the funnel) and audit the
instantaneous-value (IV) record we actually hold for them: start date, end date,
span in years, observation count, and number of usable water years.

The point is to answer "which of the no-peak stations could we recover by
deriving an annual-maximum series from IV?"  A station is realistically
recoverable for at-site LP3 / Bulletin 17C only if it has enough record — the
pipeline's working threshold is >= 10 water years (see 04b and
compare_iv_vs_peak.py).

"Usable water year" here matches 04b / compare_iv_vs_peak.py: a (site, water
year) with >= 100 IV readings.  That is stricter than raw distinct water years,
so both counts are reported.

Definitions
    inventory   : all Indiana streamflow gages (stations/indiana_streamflow_sites)
    peak_sites  : gages present in streamflow/annual_peaks/usgs_annual_peaks
    no_peak     : inventory - peak_sites   <-- the discarded stations
    recoverable : no_peak gages that also have IV data with >= 10 usable water yrs

Output
    prints the no-peak roster + IV record audit, and writes
    s3://<bucket>/<prefix>analysis/no_peak_record_audit.csv

Usage:
    python scripts/check_no_peak_stations.py
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("no_peak_audit")

INV_KEY   = "stations/indiana_streamflow_sites.parquet"
PEAK_KEY  = "streamflow/annual_peaks/usgs_annual_peaks.parquet"
IV_KEY    = "streamflow/instantaneous/all_gauges_long.parquet"
OUT_KEY   = "analysis/no_peak_record_audit.csv"

MIN_OBS_PER_YEAR = 100   # a water year "counts" only with >= 100 IV readings
MIN_WATER_YEARS  = 10    # LP3 recoverability threshold used across the pipeline


def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    # ── 1. Who has no annual-peak record? ────────────────────────────────────
    inv = _read_parquet_s3(bucket, f"{prefix}{INV_KEY}", columns=["site_no"])
    inv_sites = set(inv["site_no"].astype(str))

    peaks = _read_parquet_s3(bucket, f"{prefix}{PEAK_KEY}", columns=["site_no"])
    peak_sites = set(peaks["site_no"].astype(str))

    no_peak = sorted(inv_sites - peak_sites)
    log.info("Inventory streamflow gages          : %d", len(inv_sites))
    log.info("Gages WITH annual-peak record       : %d", len(inv_sites & peak_sites))
    log.info("Gages WITHOUT annual-peak record    : %d  <-- discarded set", len(no_peak))
    if not no_peak:
        log.info("No stations lack a peak record — nothing to audit.")
        return

    # ── 2. Audit the IV record for the discarded set ─────────────────────────
    iv = _read_parquet_s3(bucket, f"{prefix}{IV_KEY}",
                          columns=["site_no", "datetime", "value_cfs"])
    iv["site_no"]   = iv["site_no"].astype(str)
    iv = iv[iv["site_no"].isin(no_peak)]                 # only the discarded set
    iv["datetime"]  = pd.to_datetime(iv["datetime"], utc=True)
    iv["value_cfs"] = pd.to_numeric(iv["value_cfs"], errors="coerce")
    iv = iv[iv["value_cfs"] > 0]

    # Water year: Oct-Sep
    iv["water_year"] = iv["datetime"].dt.year + (iv["datetime"].dt.month >= 10).astype(int)

    # Usable water years = those with >= MIN_OBS_PER_YEAR readings
    wy_obs = iv.groupby(["site_no", "water_year"]).size().rename("n_obs").reset_index()
    usable = wy_obs[wy_obs["n_obs"] >= MIN_OBS_PER_YEAR]
    n_wy_usable = usable.groupby("site_no").size().rename("n_wy_usable")

    g = iv.groupby("site_no")
    rec = pd.DataFrame({
        "start":       g["datetime"].min().dt.date,
        "end":         g["datetime"].max().dt.date,
        "span_yr":     ((g["datetime"].max() - g["datetime"].min()).dt.days / 365.25).round(1),
        "n_obs":       g["value_cfs"].count(),
        "n_wy_raw":    g["water_year"].nunique(),
    })
    rec = rec.join(n_wy_usable)
    rec["n_wy_usable"] = rec["n_wy_usable"].fillna(0).astype(int)
    rec["recoverable"] = rec["n_wy_usable"] >= MIN_WATER_YEARS

    # Stations that have NO IV at all → not in rec; list them explicitly.
    has_iv = set(rec.index)
    no_iv  = [s for s in no_peak if s not in has_iv]

    # Build a full roster row for every discarded station (IV or not)
    roster = (rec.reset_index()
                 .rename(columns={"index": "site_no"})
                 .sort_values("n_wy_usable", ascending=False))
    if no_iv:
        pad = pd.DataFrame({"site_no": no_iv})
        for col in ["start", "end", "span_yr", "n_obs", "n_wy_raw", "n_wy_usable"]:
            pad[col] = pd.NA
        pad["recoverable"] = False
        roster = pd.concat([roster, pad], ignore_index=True)

    # ── 3. Report ────────────────────────────────────────────────────────────
    n_recover = int(rec["recoverable"].sum())
    log.info("")
    log.info("Discarded (no-peak) stations WITH IV data     : %d / %d", len(has_iv), len(no_peak))
    log.info("Discarded stations with NO IV data at all     : %d", len(no_iv))
    log.info("Recoverable (IV has >= %d usable water years) : %d", MIN_WATER_YEARS, n_recover)
    if len(has_iv):
        log.info("\nIV usable-water-year distribution (no-peak set):\n%s",
                 rec["n_wy_usable"].describe().round(1).to_string())
    log.info("\nFull no-peak roster:\n%s", roster.to_string(index=False))
    if no_iv:
        log.info("\nNo-peak AND no-IV (unrecoverable): %s", ", ".join(no_iv))

    # ── 4. Save ──────────────────────────────────────────────────────────────
    out = f"{prefix}{OUT_KEY}"
    s3_client().put_object(Bucket=bucket, Key=out,
                           Body=roster.to_csv(index=False).encode(),
                           ContentType="text/csv")
    log.info("\nFull table -> s3://%s/%s", bucket, out)


if __name__ == "__main__":
    main()
