"""04a_get_annual_peak_flow.py

Download the USGS annual PEAK-FLOW series for every Indiana streamflow site and
save it to the bucket.  The annual peak (``peak_va``) is the maximum
INSTANTANEOUS discharge of each water year — the same physical quantity as the
annual maximum of our instantaneous-value (IV) record, and the authoritative
basis for Bulletin 17C flood-frequency analysis.  The peak record typically
extends decades before the telemetry-era IV record.

Source: NWIS peak-flow service (RDB)
    https://nwis.waterdata.usgs.gov/nwis/peak?site_no=...&format=rdb

Peak qualification codes (``peak_cd``) we explicitly preserve — these matter for
B17C and for your "is it really an instantaneous peak?" question:
    1  Discharge is a Maximum Daily Average   <-- NOT an instantaneous peak
    2  Discharge is an Estimate
    3  Discharge affected by Dam Failure
    4  Discharge less than indicated value
    5  Affected to unknown degree by Regulation or Diversion
    6  Discharge affected by Regulation or Diversion
    7  Discharge is an Historic Peak
    8  Discharge actually greater than indicated value
    9  Discharge due to Snowmelt, Hurricane, Ice-Jam, etc.
    A  Year of occurrence unknown / not exact
    B  Month or day of occurrence unknown / not exact
    C  Affected by Urbanization, Mining, Channelization, etc.
    D  Base discharge changed during this year
    E  Only Annual Maximum Peak available for this year

Output
──────
    s3://<bucket>/<prefix>streamflow/annual_peaks/usgs_annual_peaks.parquet
        long format: site_no, peak_dt, water_year, peak_va, peak_cd, gage_ht

Usage:
    python scripts/04a_get_annual_peak_flow.py
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
import requests

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("04a_peaks")

PEAK_URL    = "https://nwis.waterdata.usgs.gov/nwis/peak"
MAX_WORKERS = 8
TIMEOUT     = 30
MAX_RETRIES = 3
OUT_KEY     = "streamflow/annual_peaks/usgs_annual_peaks.parquet"


# ── Date handling ─────────────────────────────────────────────────────────────

def _parse_peak_date(s: str) -> tuple[pd.Timestamp, int | None]:
    """Parse a USGS peak_dt, tolerating partial dates (month/day == '00').

    USGS encodes unknown month or day as '00' (e.g. '1867-02-00', '1923-00-00').
    Returns (timestamp_or_NaT, water_year).
    """
    if not s or pd.isna(s):
        return pd.NaT, None
    parts = str(s).split("-")
    if len(parts) != 3:
        return pd.NaT, None
    try:
        year  = int(parts[0])
        month = int(parts[1]) if parts[1] != "00" else 0
        day   = int(parts[2]) if parts[2] != "00" else 0
    except ValueError:
        return pd.NaT, None

    # Water year (Oct–Sep).  If month is unknown, fall back to calendar year.
    if month == 0:
        wy = year
        ts = pd.NaT
    else:
        wy = year + 1 if month >= 10 else year
        try:
            ts = pd.Timestamp(year=year, month=month, day=day if day else 1)
        except ValueError:
            ts = pd.NaT
    return ts, wy


# ── Peak service fetch ────────────────────────────────────────────────────────

def fetch_peaks(site_no: str) -> pd.DataFrame:
    """Return a DataFrame of annual peaks for one site (may be empty)."""
    params = {"site_no": site_no, "agency_cd": "USGS", "format": "rdb"}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(PEAK_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return _parse_rdb(site_no, r.text)
        except requests.RequestException as e:
            last_err = e
    log.warning("  %s: failed after %d attempts (%s)", site_no, MAX_RETRIES, last_err)
    return pd.DataFrame()


def _parse_rdb(site_no: str, text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()                       # header + format only, no data
    header = lines[0].split("\t")
    rows   = [ln.split("\t") for ln in lines[2:]]   # skip the '5s/10d' format row
    tbl    = pd.DataFrame(rows, columns=header)
    if "peak_dt" not in tbl.columns or "peak_va" not in tbl.columns:
        return pd.DataFrame()

    tbl["peak_va"] = pd.to_numeric(tbl["peak_va"].str.strip().replace("", None),
                                   errors="coerce")
    tbl = tbl[tbl["peak_va"].notna()]               # drop gage-height-only rows
    if tbl.empty:
        return pd.DataFrame()

    parsed = tbl["peak_dt"].apply(_parse_peak_date)
    tbl["peak_dt_parsed"] = [p[0] for p in parsed]
    tbl["water_year"]     = [p[1] for p in parsed]

    out = pd.DataFrame({
        "site_no":    str(site_no),
        "peak_dt":    tbl["peak_dt_parsed"].values,
        "water_year": tbl["water_year"].values,
        "peak_va":    tbl["peak_va"].values,
        "peak_cd":    (tbl["peak_cd"].str.strip().values
                       if "peak_cd" in tbl.columns else None),
        "gage_ht":    (pd.to_numeric(tbl["gage_ht"].str.strip().replace("", None),
                                     errors="coerce").values
                       if "gage_ht" in tbl.columns else None),
    })
    # Keep one peak per water year (USGS annual series); if duplicates, keep max.
    out = (out.sort_values("peak_va", ascending=False)
              .drop_duplicates(subset=["water_year"], keep="first")
              .sort_values("water_year")
              .reset_index(drop=True))
    return out


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    inv = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet",
                           columns=["site_no"])
    sites = sorted(inv["site_no"].astype(str).unique())
    log.info("Downloading annual peaks for %d sites...", len(sites))

    frames: list[pd.DataFrame] = []
    n_empty = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_peaks, s): s for s in sites}
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()
            if df.empty:
                n_empty += 1
            else:
                frames.append(df)
            if i % 25 == 0:
                log.info("  %d/%d", i, len(sites))

    if not frames:
        log.error("No peak data retrieved — aborting.")
        return

    peaks = pd.concat(frames, ignore_index=True)
    write_parquet_to_s3(peaks, bucket, f"{prefix}{OUT_KEY}")
    log.info("Saved %d peaks across %d sites -> s3://%s/%s%s",
             len(peaks), peaks["site_no"].nunique(), bucket, prefix, OUT_KEY)

    # ── Summary ───────────────────────────────────────────────────────────────
    per_site = peaks.groupby("site_no").agg(
        n_peaks=("peak_va", "size"),
        wy_start=("water_year", "min"),
        wy_end=("water_year", "max"),
    )
    log.info("Sites with no published peak record: %d", n_empty)
    log.info("Annual-peak count distribution:\n%s",
             per_site["n_peaks"].describe().round(1).to_string())
    log.info("Sites with >= 10 annual peaks: %d / %d",
             int((per_site["n_peaks"] >= 10).sum()), len(per_site))

    # Qualification-code prevalence (directly relevant to "is it instantaneous?")
    cd = peaks["peak_cd"].fillna("").astype(str)
    n_daily_avg = int(cd.str.contains("1").sum())
    n_historic  = int(cd.str.contains("7").sum())
    n_regulated = int(cd.str.contains("5|6").sum())
    n_estimated = int(cd.str.contains("2").sum())
    log.info("Peak qualification flags — daily-average(1): %d | historic(7): %d | "
             "regulated(5/6): %d | estimated(2): %d",
             n_daily_avg, n_historic, n_regulated, n_estimated)
    if n_daily_avg:
        log.info("  NOTE: %d peaks are flagged as Maximum Daily Average (code 1) — "
                 "these are NOT instantaneous and should be filtered before LP3.",
                 n_daily_avg)


if __name__ == "__main__":
    main()
