"""diagnose_mrms_window.py

Why does 08c's flow∩MRMS window drop stations (158 → 106)?

Replays the EXACT universe + window logic of 08c_tc_trigger_analysis (reusing
08's loaders so keys/logic match), then classifies every station that the window
step excludes:

    no_mrms     — station has NO nearest-pixel MRMS series at all.
                  A pipeline gap (05_extract_mrms_nearest never wrote it);
                  recoverable — re-running the extraction would restore it.
    no_flow     — station has no streamflow rows (shouldn't happen; here for
                  completeness since streamflow is a universe gate).
    no_overlap  — has both series but their time spans don't intersect the MRMS
                  era. An honest exclusion — nothing to compare.

Prints a per-station table + summary, and writes exports/mrms_window_dropped.csv.

Usage:
    python scripts/diagnose_mrms_window.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from utils import load_config

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

    # ── Spans: streamflow and MRMS nearest-pixel ─────────────────────────────
    flow_span = _spans(streamflow)
    try:
        mrms = m.load_mrms_nearest(bucket, prefix, product_key)
        mrms_span = _spans(mrms)
    except Exception as e:                                    # noqa: BLE001
        print(f"Could not load MRMS nearest: {e}")
        mrms_span = pd.DataFrame(columns=["start", "end"])

    # ── Classify each universe station exactly as 08c does ───────────────────
    rows: list[dict] = []
    for s in universe:
        fs_ = flow_span["start"].get(s, pd.NaT)
        fe_ = flow_span["end"].get(s, pd.NaT)
        ms_ = mrms_span["start"].get(s, pd.NaT)
        me_ = mrms_span["end"].get(s, pd.NaT)

        if pd.isna(ms_) or pd.isna(me_):
            reason = "no_mrms"
        elif pd.isna(fs_) or pd.isna(fe_):
            reason = "no_flow"
        else:
            ws, we = max(fs_, ms_), min(fe_, me_)
            reason = "kept" if ws < we else "no_overlap"

        rows.append({
            "site_no":     s,
            "reason":      reason,
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
