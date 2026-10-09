"""p08 — diagnostic audit of the NWM-Retrospective LP3 fits.  READ-ONLY.

Why this exists
───────────────
p03 keeps only Q10/Q50/Q100 and throws away everything needed to judge the fit:
`b04b.fit_lp3` already returns mean_log / std_log / skew / skew_at_site /
n_censored / fitting_method, and `b04b.compute_gof` already computes PPCC,
RMSE(log) and a KS p-value.  None of it is persisted, so the only QC the fleet
has ever had is monotonicity plus the Q100/Q10 < 1.05 "degenerate" filter in
p04/p09 — a test on the three output numbers, never on the fit or on the sample
behind it.  COMID 13437963 (ratio 1.542, Q100 ~76x too low) passes that filter.

This script re-fits every COMID with the SAME machinery p03 uses, and persists
the parameters, the goodness-of-fit statistics, the full return-period ladder,
the annual-maximum series itself, and a battery of under/over-estimation
detectors.

It writes two NEW keys and MODIFIES NOTHING:
    monitor/precompute/lp3_audit.parquet          one row per COMID
    monitor/precompute/lp3_annual_maxima.parquet  comid, water_year, peak_cfs

bridge_comid_lp3.parquet and bridge_monitor_config.parquet are never written.
No threshold in p04/p09 is changed.  Decide first, then act.

Detectors
─────────
The two error directions are NOT symmetric.  Underestimation announces itself
operationally (chronic firing); overestimation is silent (the bridge simply
never alerts — the Carmel case).  So overestimation can only be found by
testing each fit against the record it was fitted from.

D1  exceedance-count audit   years exceeding Q_T vs the expected N/T, scored
                             against Binomial(N, 1/T).  A direct statement of
                             what "return period" means.  Both directions.
D2  record-max ratios        a 45-yr record's largest annual max is ~a 45-yr
                             event, so expect Q50 ~ max and Q100 > max.
                             Q100 < max is near-certain underestimation.
D2b empirical quantile       Q_T,fit vs the Cunnane plotting-position estimate
                             from the same sample, computed ONLY where the
                             record actually supports that return period.
D3  negative-skew ceiling    LP3 with log-skew g < 0 is BOUNDED ABOVE at
                             10^(mean_log - 2*std_log/g).  A fit whose ceiling
                             sits near the record max cannot produce a large
                             flood at ANY return period — the suspected
                             mechanism behind 13437963.
D4  goodness of fit          PPCC / KS / RMSE.  Direction-agnostic: if the LP3
                             shape does not describe the sample, no quantile
                             from it is trustworthy.
D5  parameter pathology      std_log -> 0, Grubbs-Beck over-censoring, EMA
                             non-convergence, near-constant annual maxima.

Reproduction check
──────────────────
The fit is reproduced with p03's exact latitude fallback so the audit describes
the thresholds actually in production.  dev_vs_stored_pct reports the worst
per-RP deviation against the stored bridge_comid_lp3.parquet; it should be ~0.
A non-zero value means the audit is not describing the live fits — investigate
before trusting any verdict.

Usage
─────
    python p08_lp3_audit.py                      # whole fleet (hours, EC2)
    python p08_lp3_audit.py --comids 13437963 3398606 --no-resume
    python p08_lp3_audit.py --limit 500
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats

from common import config, load_script, pre_key
from monitor_common.s3io import read_parquet, write_parquet

import p03_retro_lp3 as p03

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p08")

m04c = p03.m04c
b04b = p03.b04b

IN    = pre_key("bridge_comid_tc.parquet")
STORED = pre_key("bridge_comid_lp3.parquet")
OUT_AUDIT = pre_key("lp3_audit.parquet")
OUT_AMS   = pre_key("lp3_annual_maxima.parquet")

# Full ladder, not just the three the trigger uses.  Storing these also fixes
# the 3-point ARI extrapolation that reported 13437963 as a ~12,900-yr event.
RPS = b04b.ALL_RETURN_PERIODS                 # [2, 5, 10, 25, 50, 100, 200, 500]
TRIGGER_RPS = (10, 50, 100)                   # the ones that actually fire

# ── Detector thresholds ──────────────────────────────────────────────────────
# Deliberately explicit and conservative.  Tune here, not inline.
BINOM_P        = 0.01   # binomial tail below which an exceedance count is "wrong"
CEILING_MARGIN = 2.0    # negative-skew ceiling must clear this x the record max
EMP_LOW        = 0.5    # Q_fit / Q_empirical below this -> underestimation
EMP_HIGH       = 2.0    # ...above this -> overestimation
R100_MAX_HIGH  = 5.0    # Q100 more than this x the record max -> wild extrapolation
PPCC_MIN       = 0.95   # probability-plot correlation below this -> poor fit
KS_MIN         = 0.05   # KS p-value below this -> reject the LP3 shape
CENSOR_MAX     = 0.30   # Grubbs-Beck censoring above this fraction -> biased sample
STD_LOG_MIN    = 0.05   # log-space spread below this -> collapsed curve
TIES_MAX       = 0.50   # fraction of repeated annual maxima above this -> not a real series


def empirical_quantile(ann_sorted: np.ndarray, rp: float) -> float:
    """Cunnane (a=0.4) plotting-position estimate of the rp-year flood, in cfs.

    Returns NaN when the record is too short to bracket that return period —
    an empirical 100-yr from 45 years would be extrapolation, not evidence.
    """
    n = len(ann_sorted)
    p_ne = (np.arange(1, n + 1) - 0.4) / (n + 0.2)      # non-exceedance, ascending
    target = 1.0 - 1.0 / rp
    if target > p_ne[-1] or target < p_ne[0]:
        return float("nan")
    return float(10.0 ** np.interp(target, p_ne, np.log10(ann_sorted)))


def lp3_upper_bound(mean_log: float, std_log: float, skew: float) -> float:
    """Pearson-III upper bound in cfs when the log-skew is negative, else NaN.

    Pearson III with shape a = 4/g^2, scale b = sigma*g/2 has origin
    xi = mu - 2*sigma/g.  For g < 0 that origin lies ABOVE the mean and is a
    hard upper bound on log10(Q): the distribution cannot exceed it at any
    return period.
    """
    if not np.isfinite(skew) or skew >= 0 or std_log <= 0:
        return float("nan")
    return float(10.0 ** (mean_log - 2.0 * std_log / skew))


def audit_one(comid: int, retro_site: pd.DataFrame, lat: float) -> tuple[dict, pd.DataFrame] | None:
    """Re-fit one COMID and score it.  Returns (audit_row, annual_maxima_frame)."""
    ann = m04c.annual_max_series(retro_site)       # water-year maxima, cfs
    n = len(ann)
    if n < b04b.MIN_YEARS:
        return None

    vals = ann.values.astype(float)
    log_q = np.log10(vals)
    params = b04b.fit_lp3(log_q, lat)
    gof = b04b.compute_gof(log_q, params)

    mean_log, std_log = params["mean_log"], params["std_log"]
    skew = params["skew"]
    q = {rp: float(b04b.lp3_quantile(rp, mean_log, std_log, skew)) for rp in RPS}

    ann_sorted = np.sort(vals)
    max_ann = float(ann_sorted[-1])
    n_distinct = int(len(np.unique(vals)))

    row: dict = {
        "comid": int(comid), "lat_used": float(lat),
        # ── record ──
        "n_years": n,
        "wy_start": int(ann.index.min()), "wy_end": int(ann.index.max()),
        "max_ann_cfs": max_ann, "min_ann_cfs": float(ann_sorted[0]),
        "median_ann_cfs": float(np.median(vals)),
        "n_distinct": n_distinct, "ties_frac": 1.0 - n_distinct / n,
        # ── fitted parameters (the whole point — p03 drops all of these) ──
        "mean_log": mean_log, "std_log": std_log,
        "skew": skew, "skew_at_site": params["skew_at_site"],
        "skew_regional": params["skew_regional"],
        "weight_at_site": params["weight_at_site"],
        "n_censored": params["n_censored"],
        "censor_frac": params["n_censored"] / n,
        "fitting_method": params["fitting_method"],
        **gof,
    }
    row.update({f"Q{rp}_cfs": q[rp] for rp in RPS})

    # ── D1 exceedance-count audit ────────────────────────────────────────────
    # Caveat: Q_T was fitted FROM this sample, so the counts are not strictly
    # independent of the fit and the binomial p-value is a consistency
    # diagnostic, not an exact hypothesis test.  It still ranks reaches well.
    for rp in TRIGGER_RPS:
        k = int(np.sum(vals > q[rp]))
        p = 1.0 / rp
        row[f"exc{rp}_obs"] = k
        row[f"exc{rp}_exp"] = n * p
        row[f"exc{rp}_p_hi"] = float(stats.binom.sf(k - 1, n, p))   # too many -> under
        row[f"exc{rp}_p_lo"] = float(stats.binom.cdf(k, n, p))      # too few  -> over

    # ── D2 record-max ratios ─────────────────────────────────────────────────
    row["r50_max"] = q[50] / max_ann if max_ann > 0 else np.nan
    row["r100_max"] = q[100] / max_ann if max_ann > 0 else np.nan

    # ── D2b empirical quantiles, only where the record supports them ─────────
    for rp in (10, 50):
        qe = empirical_quantile(ann_sorted, rp)
        row[f"Q{rp}_emp_cfs"] = qe
        row[f"r{rp}_emp"] = q[rp] / qe if np.isfinite(qe) and qe > 0 else np.nan

    # ── D3 negative-skew ceiling ─────────────────────────────────────────────
    bound = lp3_upper_bound(mean_log, std_log, skew)
    row["bounded_above"] = bool(np.isfinite(bound))
    row["q_bound_cfs"] = bound
    row["bound_over_max"] = bound / max_ann if np.isfinite(bound) and max_ann > 0 else np.nan

    # ── verdict ──────────────────────────────────────────────────────────────
    flags: list[str] = []
    under, over = [], []

    if np.isfinite(row["r100_max"]) and row["r100_max"] < 1.0:
        under.append("Q100_below_record_max")
    if row["exc10_p_hi"] < BINOM_P:
        under.append("too_many_Q10_exceedances")
    if np.isfinite(row["bound_over_max"]) and row["bound_over_max"] < CEILING_MARGIN:
        under.append("negative_skew_ceiling")
    if np.isfinite(row.get("r10_emp", np.nan)) and row["r10_emp"] < EMP_LOW:
        under.append("Q10_below_empirical")

    if n >= 30 and row["exc10_obs"] == 0:
        over.append("never_reached_own_Q10")
    if np.isfinite(row.get("r10_emp", np.nan)) and row["r10_emp"] > EMP_HIGH:
        over.append("Q10_above_empirical")
    if np.isfinite(row["r100_max"]) and row["r100_max"] > R100_MAX_HIGH:
        over.append("Q100_far_above_record")

    # A fit with no spread is not a curve at all, so it outranks every other
    # label: calling a flat series "OVER" because it never exceeded its own Q10
    # describes the artefact, not the error.
    degenerate = []
    if std_log < STD_LOG_MIN:
        degenerate.append("collapsed_std_log")
    if row["ties_frac"] > TIES_MAX:
        degenerate.append("repeated_annual_maxima")

    unstable = []
    if np.isfinite(gof["ppcc"]) and gof["ppcc"] < PPCC_MIN:
        unstable.append("low_ppcc")
    if np.isfinite(gof["ks_pval"]) and gof["ks_pval"] < KS_MIN:
        unstable.append("ks_reject")
    if row["censor_frac"] > CENSOR_MAX:
        unstable.append("over_censored")
    if "unstable" in str(params["fitting_method"]).lower():
        unstable.append("ema_unstable")

    flags = degenerate + under + over + unstable
    row["flags"] = ",".join(flags)
    row["n_flags"] = len(flags)
    row["verdict"] = ("DEGENERATE" if degenerate else
                      "UNDER" if under else
                      "OVER" if over else
                      "UNSTABLE" if unstable else "OK")

    # Fold-error for ranking: how many times wrong, >= 1 by construction.
    sev = [1.0]
    if np.isfinite(row["r100_max"]) and row["r100_max"] < 1.0:
        sev.append(1.0 / row["r100_max"])
    r10e = row.get("r10_emp", np.nan)
    if np.isfinite(r10e) and r10e > 0:
        sev.append(max(r10e, 1.0 / r10e))
    if np.isfinite(row["bound_over_max"]) and row["bound_over_max"] < CEILING_MARGIN:
        sev.append(CEILING_MARGIN / max(row["bound_over_max"], 1e-9))
    row["severity"] = float(max(sev))

    ams = pd.DataFrame({"comid": int(comid),
                        "water_year": ann.index.astype(int),
                        "peak_cfs": vals})
    return row, ams


def _finalize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.drop_duplicates("comid")
              .sort_values(["severity", "n_flags"], ascending=False)
              .reset_index(drop=True))


def attach_stored_deviation(audit: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Worst per-RP % deviation of the re-fit against the stored p03 table.

    This is the audit's own credibility check: it should be ~0.  Anything
    larger means p08 is not reproducing the fits that are actually live.
    """
    if audit.empty:
        return audit
    try:
        stored = read_parquet(bucket, STORED)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read stored LP3 (%s) — skipping reproduction check", e)
        audit["dev_vs_stored_pct"] = np.nan
        return audit
    stored = stored[["comid", "Q10_cfs", "Q50_cfs", "Q100_cfs"]].copy()
    stored["comid"] = pd.to_numeric(stored["comid"], errors="coerce").astype("Int64")
    m = audit.merge(stored, on="comid", how="left", suffixes=("", "_stored"))
    dev = []
    for rp in TRIGGER_RPS:
        a, s = m[f"Q{rp}_cfs"], m[f"Q{rp}_cfs_stored"]
        dev.append(((a - s).abs() / s.replace(0, np.nan) * 100.0))
    audit["dev_vs_stored_pct"] = pd.concat(dev, axis=1).max(axis=1).values
    return audit


def report(audit: pd.DataFrame) -> None:
    if audit.empty:
        log.warning("Nothing fitted — no report")
        return
    n = len(audit)
    log.info("─" * 74)
    log.info("LP3 FIT AUDIT — %d COMIDs", n)
    log.info("─" * 74)

    dev = audit["dev_vs_stored_pct"].dropna()
    if len(dev):
        worst = float(dev.max())
        (log.info if worst < 1.0 else log.warning)(
            "reproduction vs stored p03: worst deviation %.4f%% (%d compared)%s",
            worst, len(dev), "" if worst < 1.0 else "  <-- audit may not describe live fits")

    log.info("")
    for v in ("DEGENERATE", "UNDER", "OVER", "UNSTABLE", "OK"):
        k = int((audit["verdict"] == v).sum())
        log.info("  %-9s %6d  (%5.1f%%)", v, k, 100.0 * k / n)

    log.info("")
    log.info("flag frequency:")
    flat = audit["flags"].str.split(",").explode()
    for name, k in flat[flat != ""].value_counts().items():
        log.info("  %-28s %6d", name, k)

    log.info("")
    log.info("spread / shape distribution:")
    for col, label in (("std_log", "std_log"), ("skew", "weighted skew"),
                       ("r100_max", "Q100 / record max"), ("ppcc", "PPCC")):
        s = audit[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s):
            log.info("  %-20s p01 %9.3f  p50 %9.3f  p99 %9.3f",
                     label, s.quantile(0.01), s.median(), s.quantile(0.99))
    nb = int(audit["bounded_above"].sum())
    log.info("  negative skew (bounded above): %d of %d (%.1f%%)", nb, n, 100.0 * nb / n)

    log.info("")
    log.info("worst 20 by fold-error:")
    cols = ["comid", "verdict", "severity", "n_years", "std_log", "skew",
            "Q10_cfs", "Q100_cfs", "max_ann_cfs", "r100_max", "exc10_obs", "flags"]
    for _, r in audit.head(20).iterrows():
        log.info("  %-10d %-8s x%-9.1f n=%-3d sd=%-6.3f g=%-7.3f Q10=%-10.1f "
                 "Q100=%-10.1f max=%-10.1f r=%-6.2f exc10=%-3d %s",
                 r["comid"], r["verdict"], r["severity"], r["n_years"],
                 r["std_log"], r["skew"], r["Q10_cfs"], r["Q100_cfs"],
                 r["max_ann_cfs"], r["r100_max"], r["exc10_obs"], r["flags"])
    _ = cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=200, help="COMIDs per Zarr bulk-load")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--comids", type=int, nargs="+", default=None,
                    help="audit only these COMIDs (smoke test)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing audit table and start clean")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and report but write nothing to S3")
    args = ap.parse_args()

    b, _ = config.bucket_prefix()
    src = read_parquet(b, IN)
    src["comid"] = pd.to_numeric(src["comid"], errors="coerce").astype("Int64")
    src = src.dropna(subset=["comid"])
    # p03's exact latitude fallback — the audit must reproduce the live fit.
    lat_by_comid = src.dropna(subset=["lat"]).groupby("comid")["lat"].first().to_dict()

    comids = sorted({int(c) for c in src["comid"].dropna().unique()})
    if args.comids:
        comids = [c for c in args.comids]
        log.info("Smoke test on %d explicit COMID(s)", len(comids))
    elif args.limit:
        comids = comids[:args.limit]
    log.info("COMIDs to audit: %d", len(comids))

    rows: list[dict] = []
    ams_parts: list[pd.DataFrame] = []
    done: set[int] = set()
    if not args.no_resume and not args.comids:
        try:
            prev = read_parquet(b, OUT_AUDIT)
            rows = prev.to_dict("records")
            done = {int(r["comid"]) for r in rows}
            log.info("Resuming: %d COMIDs already audited", len(done))
            try:
                ams_parts.append(read_parquet(b, OUT_AMS))
            except Exception:
                log.warning("Audit table resumed but annual maxima missing — "
                            "they will only cover COMIDs audited from here on")
        except Exception:
            pass

    todo = [c for c in comids if c not in done]
    nwm10 = m04c._load_nwm10()
    log.info("Opening retrospective Zarr once (streamflow only)...")
    sf_da, available = p03.open_retro(nwm10)
    t0 = pd.Timestamp(nwm10.RETRO_START).tz_localize(None)
    t1 = pd.Timestamp(nwm10.RETRO_END).tz_localize(None)

    n_short = n_err = 0
    for bstart in range(0, len(todo), args.batch):
        batch = todo[bstart:bstart + args.batch]
        log.info("Batch %d-%d/%d: loading streamflow for %d COMIDs...",
                 bstart, bstart + len(batch), len(todo), len(batch))
        try:
            series = p03.load_batch_streamflow(sf_da, available, batch, t0, t1)
        except Exception as e:  # noqa: BLE001
            log.error("Batch load failed (%s) — skipping batch", e)
            continue
        for c in batch:
            sub = series.get(c)
            if sub is None or sub.empty:
                continue
            try:
                res = audit_one(c, sub, float(lat_by_comid.get(c, 40.0)))
            except Exception as e:  # noqa: BLE001
                log.debug("COMID %d audit failed: %s", c, e)
                n_err += 1
                continue
            if res is None:
                n_short += 1
                continue
            row, ams = res
            rows.append(row)
            ams_parts.append(ams)

        if not args.dry_run:
            write_parquet(_finalize(rows), b, OUT_AUDIT)
            write_parquet(pd.concat(ams_parts, ignore_index=True)
                          .drop_duplicates(["comid", "water_year"]), b, OUT_AMS)
            log.info("Checkpoint: %d audited", len(rows))

    audit = attach_stored_deviation(_finalize(rows), b)
    if args.dry_run:
        log.info("DRY RUN — writing nothing")
    else:
        write_parquet(audit, b, OUT_AUDIT)
        ams_all = (pd.concat(ams_parts, ignore_index=True)
                     .drop_duplicates(["comid", "water_year"])
                     .sort_values(["comid", "water_year"]).reset_index(drop=True))
        write_parquet(ams_all, b, OUT_AMS)
        log.info("Wrote %s (%d rows) and %s (%d rows)",
                 OUT_AUDIT, len(audit), OUT_AMS, len(ams_all))
    if n_short or n_err:
        log.info("skipped: %d short record (< %d WY), %d fit errors",
                 n_short, b04b.MIN_YEARS, n_err)
    report(audit)


if __name__ == "__main__":
    main()
