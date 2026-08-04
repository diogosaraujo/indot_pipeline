"""p05 — validate bridge_monitor_config.parquet before launching the monitor.

Runs five families of checks and prints a PASS/WARN report:

  1. INTERNAL   monotonic P10<=P50<=P100 & Q10<=Q50<=Q100; identical Q per COMID;
                identical P per (cell,Tc); NaN/negative accounting.
  2. Q vs 04c   for COMIDs shared with the 106 NWM gauge reaches
                (flow_stats/nwm_per_gauge_flow_stats.parquet) — should match ~exactly.
  3. STUDY      bridge COMID 10357264 vs its cached LP3 (flow_stats/bridge_nwm_lp3/).
  4. P re-fetch (optional, --refetch N) re-pull NOAA PFDS for N bridge cells and
                compare to stored depths; also check depth increases with duration.
  5. PHYSICAL   distribution percentiles + maps of Q10 / P10 + Tc histogram.

Usage:
    python monitor/precompute/p05_validate.py                 # checks 1-3,5
    python monitor/precompute/p05_validate.py --refetch 20    # also re-fetch PFDS
    python monitor/precompute/p05_validate.py --no-plots
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import config, load_script, pre_key
from monitor_common.s3io import read_parquet, write_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("precompute.p05")

RPS = config.SEVERITY_RPS
WARN: list[str] = []


def warn(msg: str) -> None:
    WARN.append(msg)
    log.warning("WARN  %s", msg)


def ok(msg: str) -> None:
    log.info("ok    %s", msg)


# ── 1. Internal consistency ──────────────────────────────────────────────────

def check_internal(cfg: pd.DataFrame) -> None:
    log.info("── 1. internal consistency ───────────────────────────────")
    n = len(cfg)

    for fam, cols in (("P", [f"P{r}" for r in RPS]), ("Q", [f"Q{r}_cfs" for r in RPS])):
        sub = cfg[cols].to_numpy(float)
        rows = np.isfinite(sub).all(axis=1)
        bad = rows & ~((np.diff(sub, axis=1) >= -1e-6).all(axis=1))
        (ok if bad.sum() == 0 else warn)(
            f"{fam} monotonic (P10<=P50<=P100): {int(bad.sum())} violations of {int(rows.sum())} complete rows")

    # identical Q for all bridges sharing a COMID (merge integrity)
    q = cfg.dropna(subset=["comid"])
    gnun = q.groupby("comid")[[f"Q{r}_cfs" for r in RPS]].nunique(dropna=False)
    bad_comid = (gnun > 1).any(axis=1).sum()
    (ok if bad_comid == 0 else warn)(f"identical Q per COMID: {int(bad_comid)} COMIDs with inconsistent Q")

    # identical P within (cell, tc_dur)
    tmp = cfg.copy()
    tmp["cell"] = tmp["lat"].round(3).astype(str) + "_" + tmp["lon"].round(3).astype(str)
    gnun = tmp.groupby(["cell", "tc_dur_hr"])[[f"P{r}" for r in RPS]].nunique(dropna=False)
    bad_cell = (gnun > 1).any(axis=1).sum()
    (ok if bad_cell == 0 else warn)(f"identical P per (cell,Tc): {int(bad_cell)} groups with inconsistent P")

    # coverage / null accounting
    for c in [f"P{r}" for r in RPS] + [f"Q{r}_cfs" for r in RPS]:
        nnull = int(cfg[c].isna().sum())
        neg = int((cfg[c] < 0).sum())
        line = f"{c}: {n - nnull}/{n} present ({100*(n-nnull)/n:.1f}%), {neg} negative"
        (warn if neg else ok)(line)

    # tc_dur range
    td = cfg["tc_dur_hr"]
    (ok if td.between(1, 72).all() else warn)(
        f"tc_dur_hr range [{int(td.min())}, {int(td.max())}] h (expect 1-72)")


# ── 2. Q vs 04c gauge flow-stats ─────────────────────────────────────────────

def check_q_vs_04c(cfg: pd.DataFrame) -> None:
    log.info("── 2. Q vs 04c NWM gauge flow-stats ──────────────────────")
    b, p = config.bucket_prefix()
    try:
        g = read_parquet(b, f"{p}flow_stats/nwm_per_gauge_flow_stats.parquet")
    except Exception as e:  # noqa: BLE001
        warn(f"could not read 04c gauge flow-stats ({e}) — skipping the strongest cross-check")
        return
    g["comid"] = pd.to_numeric(g["comid"], errors="coerce").astype("Int64")
    bridges = cfg.dropna(subset=["comid"]).drop_duplicates("comid")[
        ["comid", "Q10_cfs", "Q50_cfs", "Q100_cfs"]]
    m = bridges.merge(g[["comid", "Q10", "Q50", "Q100"]], on="comid", how="inner")
    if m.empty:
        warn("no COMIDs shared between bridges and the 106 gauge reaches — "
             "relying on the study-bridge check instead")
        return
    worst = 0.0
    for r in RPS:
        a = m[f"Q{r}_cfs"].to_numpy(float)
        bb = m[f"Q{r}"].to_numpy(float)
        pct = np.abs(a - bb) / np.where(bb != 0, np.abs(bb), np.nan) * 100
        worst = max(worst, float(np.nanmax(pct)))
        log.info("    Q%d: %d shared COMIDs, median |Δ|=%.3f%%, max=%.3f%%",
                 r, len(m), float(np.nanmedian(pct)), float(np.nanmax(pct)))
    (ok if worst < 1.0 else warn)(
        f"Q vs 04c on {len(m)} shared reaches: worst deviation {worst:.3f}% (expect <1%)")


# ── 3. Study bridge ──────────────────────────────────────────────────────────

def check_study_bridge(cfg: pd.DataFrame, comid: int = 10357264) -> None:
    log.info("── 3. study bridge COMID %d ──────────────────────────────", comid)
    b, p = config.bucket_prefix()
    row = cfg[cfg["comid"] == comid]
    if row.empty:
        warn(f"study COMID {comid} not among the bridges — skipping")
        return
    try:
        cache = read_parquet(b, f"{p}flow_stats/bridge_nwm_lp3/{comid}.parquet")
    except Exception as e:  # noqa: BLE001
        warn(f"no cached study LP3 for COMID {comid} ({e}) — skipping")
        return
    ref = {int(r.return_period_yr): float(r.q_cfs) for r in cache.itertuples()}
    r0 = row.iloc[0]
    worst = 0.0
    for rp in RPS:
        got, exp = float(r0[f"Q{rp}_cfs"]), ref.get(rp, np.nan)
        pct = abs(got - exp) / exp * 100 if exp else np.nan
        worst = max(worst, pct if np.isfinite(pct) else 0.0)
        log.info("    Q%d: config=%.0f cached=%.0f  Δ=%.3f%%", rp, got, exp, pct)
    (ok if worst < 1.0 else warn)(f"study bridge worst deviation {worst:.3f}% (expect <1%)")


# ── 4. P re-fetch from PFDS (optional) ───────────────────────────────────────

def check_p_refetch(cfg: pd.DataFrame, n: int) -> None:
    log.info("── 4. Atlas-14 re-fetch (%d cells) ───────────────────────", n)
    b, p = config.bucket_prefix()
    ddf = read_parquet(b, pre_key("bridge_atlas14.parquet"))
    ddf["cell"] = ddf["cell"].astype(str)

    # depth increases with duration (per cell, RP10) — pure DDF sanity, no network
    bad = 0
    for cell, g in ddf[ddf["return_period_yr"] == 10].groupby("cell"):
        g = g.sort_values("duration_hr")
        if (np.diff(g["depth_in"].to_numpy(float)) < -1e-6).any():
            bad += 1
    (ok if bad == 0 else warn)(f"depth increases with duration: {bad} cells violate")

    if n <= 0:
        return
    m07 = load_script("atlas14_07", "07_extract_atlas14.py")
    tmp = cfg.copy()
    tmp["cell"] = tmp["lat"].round(3).astype(str) + "_" + tmp["lon"].round(3).astype(str)
    sample = tmp.drop_duplicates("cell").sample(min(n, tmp["cell"].nunique()), random_state=1)
    diffs = []
    for r in sample.itertuples():
        try:
            text = m07.fetch_atlas14(r.lat, r.lon)
            fresh = m07.parse_atlas14(r.cell, text)
        except Exception as e:  # noqa: BLE001
            log.debug("refetch %s failed: %s", r.cell, e)
            continue
        stored = ddf[ddf["cell"] == r.cell]
        j = fresh.merge(stored, on=["duration_hr", "return_period_yr"], suffixes=("_new", "_old"))
        if j.empty:
            continue
        pct = np.abs(j["depth_in_new"] - j["depth_in_old"]) / j["depth_in_old"] * 100
        diffs.append(float(np.nanmax(pct)))
    if diffs:
        (ok if max(diffs) < 0.5 else warn)(
            f"PFDS re-fetch on {len(diffs)} cells: worst deviation {max(diffs):.3f}% (expect ~0)")
    else:
        warn("PFDS re-fetch returned nothing to compare")


# ── 5. Distributions + maps ──────────────────────────────────────────────────

def summarize_and_plot(cfg: pd.DataFrame, plots: bool) -> None:
    log.info("── 5. distributions ──────────────────────────────────────")
    qs = [0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for c in [f"P{r}" for r in RPS] + [f"Q{r}_cfs" for r in RPS] + ["tc_dur_hr"]:
        v = pd.to_numeric(cfg[c], errors="coerce").dropna()
        pcts = ", ".join(f"{int(q*100)}%={v.quantile(q):,.2f}" for q in qs)
        log.info("    %-9s  %s", c, pcts)

    # eyeball the biggest / smallest reaches
    top = cfg.dropna(subset=["Q10_cfs"]).nlargest(8, "Q10_cfs")[["Asset Name", "comid", "Q10_cfs", "Q100_cfs"]]
    log.info("    largest Q10 reaches (sanity: should be major rivers):\n%s", top.to_string(index=False))

    if not plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b, _ = config.bucket_prefix()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    d = cfg.dropna(subset=["Q10_cfs"])
    sc = axes[0].scatter(d["lon"], d["lat"], c=np.log10(d["Q10_cfs"].clip(lower=1)),
                         s=4, cmap="viridis")
    axes[0].set_title("log10 Q10 (cfs) — big rivers should stand out"); fig.colorbar(sc, ax=axes[0])
    dp = cfg.dropna(subset=["P10"])
    sc2 = axes[1].scatter(dp["lon"], dp["lat"], c=dp["P10"], s=4, cmap="YlGnBu")
    axes[1].set_title("P10 at Tc (in)"); fig.colorbar(sc2, ax=axes[1])
    axes[2].hist(cfg["tc_dur_hr"], bins=range(1, int(cfg["tc_dur_hr"].max()) + 2), color="#555")
    axes[2].set_title("Tc accumulation duration (h)"); axes[2].set_xlabel("h")
    for a in axes[:2]:
        a.set_xlabel("lon"); a.set_ylabel("lat"); a.set_aspect(1.3)
    fig.tight_layout()
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    key = f"{config.bucket_prefix()[1]}monitor/validation/coverage_maps.png"
    write_bytes(buf.getvalue(), b, key, content_type="image/png")
    log.info("    maps -> s3://%s/%s", b, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch", type=int, default=0, help="re-fetch N Atlas-14 cells from PFDS")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    b, _ = config.bucket_prefix()
    cfg = read_parquet(b, config.keys()["config"])
    cfg["comid"] = pd.to_numeric(cfg["comid"], errors="coerce").astype("Int64")
    log.info("Loaded config: %d bridges", len(cfg))

    check_internal(cfg)
    check_q_vs_04c(cfg)
    check_study_bridge(cfg)
    check_p_refetch(cfg, args.refetch)
    summarize_and_plot(cfg, not args.no_plots)

    log.info("══════════════════════════════════════════════════════════")
    if WARN:
        log.warning("VALIDATION FINISHED WITH %d WARNING(S):", len(WARN))
        for w in WARN:
            log.warning("   • %s", w)
    else:
        log.info("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
