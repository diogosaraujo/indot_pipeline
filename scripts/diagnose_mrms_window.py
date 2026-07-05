"""diagnose_mrms_window.py

Why does 08c's flow∩MRMS window drop stations (158 → 106)?

Replays the EXACT universe + window logic of 08c_tc_trigger_analysis (reusing
08's loaders so keys/logic match), classifies every station the window step
excludes, and — for the ones missing MRMS — decides whether they are RECOVERABLE.

MRMS coverage is meant to span 2002 → present:
    script 05   NOAA MRMS PDS      2020-10-14 → present
    script 05b  ISU MRMS + Stage IV 2002-01-01 → 2020-10-13  (backfill)
Both extractors read the *active* gauge list (indiana_streamflow_sites_active),
which script 01 filters to end_date >= 2020-10-14.  So a gauge that stopped
before 2020-10-14 is excluded from MRMS extraction even though Stage IV covers
2002-2020 — that is why such gauges show up as no_mrms.

Classification of dropped stations:
    no_mrms / recoverable   — no MRMS series, but streamflow overlaps the Stage IV
                              era (flow_end >= 2002-01-01).  Would be restored by
                              running 05b over the FULL inventory, not just active.
    no_mrms / out_of_era    — no MRMS series AND streamflow ends before 2002.
                              Genuinely nothing to compare; correctly excluded.
    no_overlap              — has an MRMS series but spans don't intersect.
    no_flow                 — no streamflow rows (shouldn't happen; a gate).

Prints a per-station table + summary, and writes exports/mrms_window_dropped.csv.

Usage:
    python scripts/diagnose_mrms_window.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from utils import load_config

STAGE4_START = pd.Timestamp("2002-01-01", tz="UTC")   # earliest historical MRMS (Stage IV)

# Reuse 08's loaders so keys / logic match the real 08c run exactly.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

OUT_CSV = Path("exports/mrms_window_dropped.csv")


def _spans(df: pd.DataFrame) -> pd.DataFrame:
    """site_no → (start, end) from a long table with datetime_utc."""
    g = df.groupby("site_no")["datetime_utc"]
    return pd.DataFrame({"start": g.min(), "end": g.max()})


def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]
    product_key = cfg["mrms"]["products"][0]["key"]

    # ── Universe: valid Q ∩ Kirpich Tc ∩ Atlas14 ∩ streamflow (08c criteria) ──
    flow_stats = m.load_flow_stats(bucket, prefix)
    atlas14    = m.load_atlas14(bucket, prefix)
    streamflow = m.load_streamflow(bucket, prefix)

    tc = m._read_parquet_s3(
        bucket, f"{prefix}watersheds/basin_characteristics.parquet",
        ["site_no", "tc_hr"])
    tc["site_no"] = tc["site_no"].astype(str)
    tc_sites = set(tc.dropna(subset=["tc_hr"])["site_no"])

    q_cols = [f"Q{rp}" for rp in m.FLOW_RPS if f"Q{rp}" in flow_stats.columns]
    has_q  = set(flow_stats.loc[flow_stats[q_cols].notna().any(axis=1), "site_no"])

    universe = sorted(
        has_q & tc_sites & set(atlas14["site_no"]) & set(streamflow["site_no"]))
    print(f"Universe (valid Q ∩ Tc ∩ Atlas14 ∩ streamflow): {len(universe)}")

    # ── Which of these are in the 'active' list the MRMS extractors use? ──────
    try:
        active = m._read_parquet_s3(
            bucket, f"{prefix}stations/indiana_streamflow_sites_active.parquet",
            ["site_no"])
        active_sites = set(active["site_no"].astype(str))
    except Exception as e:                                    # noqa: BLE001
        print(f"Could not read active-stations list: {e}")
        active_sites = set()

    # ── Spans: streamflow and MRMS nearest-pixel ─────────────────────────────
    flow_span = _spans(streamflow)
    try:
        mrms = m.load_mrms_nearest(bucket, prefix, product_key)
        mrms_span = _spans(mrms)
        mrms_cov = (mrms["datetime_utc"].min(), mrms["datetime_utc"].max())
    except Exception as e:                                    # noqa: BLE001
        print(f"Could not load MRMS nearest: {e}")
        mrms_span = pd.DataFrame(columns=["start", "end"])
        mrms_cov = (pd.NaT, pd.NaT)

    print(f"MRMS nearest_pixel coverage in S3: {mrms_cov[0]} → {mrms_cov[1]}")
    print(f"Stage IV era floor: {STAGE4_START.date()}\n")

    # ── Classify each universe station exactly as 08c does ───────────────────
    rows: list[dict] = []
    for s in universe:
        fs_ = flow_span["start"].get(s, pd.NaT)
        fe_ = flow_span["end"].get(s, pd.NaT)
        ms_ = mrms_span["start"].get(s, pd.NaT)
        me_ = mrms_span["end"].get(s, pd.NaT)

        if pd.isna(ms_) or pd.isna(me_):
            # No MRMS series. Recoverable if the gauge has streamflow in the
            # Stage IV era (flow_end >= 2002) — 05b just never extracted it
            # because it was absent from the active list.
            if not pd.isna(fe_) and fe_ >= STAGE4_START:
                reason = "no_mrms / recoverable"
            else:
                reason = "no_mrms / out_of_era"
        elif pd.isna(fs_) or pd.isna(fe_):
            reason = "no_flow"
        else:
            ws, we = max(fs_, ms_), min(fe_, me_)
            reason = "kept" if ws < we else "no_overlap"

        rows.append({
            "site_no":     s,
            "reason":      reason,
            "in_active":   s in active_sites,
            "flow_start":  fs_, "flow_end":  fe_,
            "mrms_start":  ms_, "mrms_end":  me_,
        })

    res = pd.DataFrame(rows)
    dropped = res[res["reason"] != "kept"].copy()

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\nKept (valid flow∩MRMS window): {(res['reason'] == 'kept').sum()}")
    print(f"Dropped:                        {len(dropped)}\n")
    print("Drop reasons:")
    print(res["reason"].value_counts().to_string())

    if not dropped.empty:
        print("\nDropped stations:")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(dropped.sort_values(["reason", "site_no"]).to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dropped.sort_values(["reason", "site_no"]).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
