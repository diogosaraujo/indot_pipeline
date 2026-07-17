#!/usr/bin/env python3
"""07a_areal_precip_frequency.py

Atlas-14-style precipitation-frequency analysis for the MRMS WATERSHED-MEAN
(areal) series, using REGIONAL L-moment pooling so the tail (P200-P1000) is
robust despite the ~24-year MRMS record.

Method (mirrors Atlas 14's own approach where feasible)
-------------------------------------------------------
  1. For each station and each duration D in DURATIONS, build the annual maximum
     series (AMS) of the D-hour watershed-mean accumulation (water year, with a
     per-year completeness screen so MRMS gaps don't deflate the annual max).
  2. Fit a GEV by the method of L-moments (Hosking 1990) — Atlas 14's distribution
     and estimator.
  3. Estimate quantiles per station. Two modes (--method):
       atsite   (default) — fit the GEV to each station's OWN AMS L-moments. Simple;
                            the far tail (P200-P1000) is noisier on the ~24-yr record.
       regional           — index-flood L-moment pooling (Hosking & Wallis 1997):
                            one regional GEV growth curve scaled per station by the
                            at-site mean, trading spatial replication for record
                            length (more robust tail). Emits discordancy D_i per
                            site and heterogeneity H per duration.

Output (schema identical to Atlas 14 / script 07, so an areal 08c reads it):
    mrms/<PRODUCT>/areal_precip_frequency.parquet
        site_no, duration_hr, return_period_yr, depth_in
    analysis/areal_ddf_site_diag.parquet     (per site x duration L-moments, D)
    analysis/areal_ddf_region_diag.parquet   (per duration regional params, H)

NOTE on H: Hosking & Wallis recommend simulating the homogeneous region from a
4-parameter kappa distribution. A correct kappa fit is error-prone to hand-code,
so this uses the fitted regional GEV for the simulation instead — transparent and
defensible, marginally less flexible. Cross-check with lmoments3/R if H is
paper-critical.

Validate the math first:
    python scripts/07a_areal_precip_frequency.py --selftest
Run:
    python scripts/07a_areal_precip_frequency.py
    python scripts/07a_areal_precip_frequency.py --all          # pool ALL wshed stations
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma as _Gamma

from utils import load_config, write_parquet_to_s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("07a_areal_pf")

# Atlas-14-style durations: sub-daily (hourly floor — MRMS is hourly) plus the
# multi-day durations 1,2,3,4,7,10,20 day, so Kirpich Tc up to ~20 d is bracketed.
DURATIONS_HR    = [1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 240, 480]
RETURN_PERIODS  = [2, 5, 10, 25, 50, 100, 200, 500, 1000]   # AMS GEV: T>=2 only
MIN_YEARS       = 12       # min screened annual maxima for a station to enter
MIN_FRAC        = 0.80     # min fraction of a water year's hours present to count it
NSIM            = 500      # Monte-Carlo replicates for the H statistic
SEED            = 42
LN2, LN3        = np.log(2.0), np.log(3.0)
EULER           = 0.5772156649015329


# ══ L-moments, GEV (Hosking 1990) ═════════════════════════════════════════════

def sample_lmoments(x: np.ndarray) -> tuple[float, float, float, float, float]:
    """Unbiased sample L-moments via PWMs. Returns (l1, l2, t=L-CV, t3, t4)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l2 / l1, l3 / l2, l4 / l2


def gev_params_from_lmom(l1: float, l2: float, t3: float) -> tuple[float, float, float]:
    """GEV (xi, alpha, k) from L-moments — Hosking (1990) / Atlas 14."""
    c = 2.0 / (3.0 + t3) - LN2 / LN3
    k = 7.8590 * c + 2.9554 * c * c
    if abs(k) < 1e-6:                                   # Gumbel limit
        alpha = l2 / LN2
        return l1 - EULER * alpha, alpha, 0.0
    g = _Gamma(1.0 + k)
    alpha = l2 * k / ((1.0 - 2.0 ** (-k)) * g)
    xi = l1 - alpha * (1.0 - g) / k
    return xi, alpha, k


def gev_quantile(F, xi: float, alpha: float, k: float) -> np.ndarray:
    """GEV inverse CDF. x(F) = xi + (alpha/k)[1 - (-ln F)^k]."""
    F = np.asarray(F, dtype=float)
    y = -np.log(F)
    if abs(k) < 1e-6:
        return xi - alpha * np.log(y)
    return xi + (alpha / k) * (1.0 - y ** k)


# ══ Regional diagnostics (Hosking & Wallis 1997) ══════════════════════════════

def discordancy(U: np.ndarray) -> np.ndarray:
    """D_i for each site from its L-moment-ratio vector U_i = (t, t3, t4)."""
    N = len(U)
    dev = U - U.mean(axis=0)
    S = dev.T @ dev
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.zeros(N)
    return np.array([(N / 3.0) * dev[i] @ Sinv @ dev[i] for i in range(N)])


def heterogeneity_H(t_arr, n_arr, xi_g, alpha_g, k_g, nsim, rng) -> float:
    """H&W heterogeneity, simulated from the fitted regional GEV growth curve."""
    n_arr = np.asarray(n_arr, float)
    W = n_arr / n_arr.sum()
    tR = np.sum(W * t_arr)
    V_obs = np.sqrt(np.sum(W * (t_arr - tR) ** 2))
    Vs = np.empty(nsim)
    for s in range(nsim):
        ts = np.empty(len(n_arr))
        for i, n in enumerate(n_arr):
            x = gev_quantile(rng.uniform(size=int(n)), xi_g, alpha_g, k_g)
            ts[i] = sample_lmoments(x)[2]
        tRs = np.sum(W * ts)
        Vs[s] = np.sqrt(np.sum(W * (ts - tRs) ** 2))
    sd = Vs.std()
    return float((V_obs - Vs.mean()) / sd) if sd > 0 else float("nan")


def regional_fit(sites: list[dict], nsim: int, rng):
    """Index-flood regional GEV from a list of {site_no, ams}. Returns
    (region_dict, per_site_rows) where per_site_rows carry depths at RETURN_PERIODS."""
    recs = []
    for s in sites:
        l1, l2, t, t3, t4 = sample_lmoments(s["ams"])
        recs.append({"site_no": s["site_no"], "mean": l1, "t": t, "t3": t3,
                     "t4": t4, "n": len(s["ams"])})
    R = pd.DataFrame(recs)
    n = R["n"].to_numpy(float)
    W = n / n.sum()
    tR  = float(np.sum(W * R["t"]))
    t3R = float(np.sum(W * R["t3"]))
    t4R = float(np.sum(W * R["t4"]))

    # Growth curve: regional GEV with mean 1 (l1=1 => l2=tR)
    xi_g, alpha_g, k_g = gev_params_from_lmom(1.0, tR, t3R)

    R["D"] = discordancy(R[["t", "t3", "t4"]].to_numpy())
    H = heterogeneity_H(R["t"].to_numpy(), n, xi_g, alpha_g, k_g, nsim, rng)

    F = np.array([1.0 - 1.0 / T for T in RETURN_PERIODS])
    growth = gev_quantile(F, xi_g, alpha_g, k_g)         # dimensionless, mean 1
    site_rows = []
    for _, r in R.iterrows():
        depths = r["mean"] * growth                      # index-flood scaling
        for T, dep in zip(RETURN_PERIODS, depths):
            site_rows.append({"site_no": r["site_no"], "return_period_yr": T,
                              "depth_in": round(float(dep), 4)})
    region = {"tR": tR, "t3R": t3R, "t4R": t4R, "gev_xi": xi_g,
              "gev_alpha": alpha_g, "gev_k": k_g, "H": H, "n_sites": len(R),
              "n_years_median": int(R["n"].median())}
    return region, R, site_rows


def atsite_fit(sites: list[dict]):
    """At-site GEV per station (no pooling). Returns (per_site_rows, site_diag_df)."""
    F = np.array([1.0 - 1.0 / T for T in RETURN_PERIODS])
    recs, site_rows = [], []
    for s in sites:
        l1, l2, t, t3, t4 = sample_lmoments(s["ams"])
        xi, alpha, k = gev_params_from_lmom(l1, l2, t3)
        for T, dep in zip(RETURN_PERIODS, gev_quantile(F, xi, alpha, k)):
            site_rows.append({"site_no": s["site_no"], "return_period_yr": T,
                              "depth_in": round(float(dep), 4)})
        recs.append({"site_no": s["site_no"], "mean": l1, "t": t, "t3": t3, "t4": t4,
                     "gev_xi": xi, "gev_alpha": alpha, "gev_k": k, "n": len(s["ams"])})
    return site_rows, pd.DataFrame(recs)


# ══ AMS extraction ════════════════════════════════════════════════════════════

def annual_maxima(series: pd.Series, d: int, min_frac: float):
    """Water-year annual maxima of the d-hour accumulation, completeness-screened.
    `series` is hourly value_mean (gaps allowed). Returns np.array of maxima."""
    idx = series.index
    start = idx.min(); end = idx.max()
    wy0 = start.year - 1 if start.month < 10 else start.year
    wy1 = end.year if end.month < 10 else end.year + 1
    grid = pd.date_range(pd.Timestamp(year=wy0, month=10, day=1, tz="UTC"),
                         pd.Timestamp(year=wy1, month=9, day=30, hour=23, tz="UTC"),
                         freq="1h")
    s = series.reindex(grid)
    roll = s.rolling(window=d, min_periods=d).sum()
    wy = grid.year + (grid.month >= 10).astype(int)
    df = pd.DataFrame({"roll": roll.values, "present": s.notna().values.astype(int),
                       "wy": wy}, index=grid)
    maxima = []
    for _, g in df.groupby("wy"):
        if g["present"].mean() >= min_frac and g["roll"].notna().any():
            maxima.append(float(g["roll"].max()))
    return np.array(maxima)


# ══ I/O ═══════════════════════════════════════════════════════════════════════

def load_universe(bucket, prefix, key) -> set[str]:
    df = pd.read_parquet(f"s3://{bucket}/{prefix}{key}", columns=["site_no"])
    return set(df["site_no"].astype(str).unique())


def load_watershed_series(bucket, prefix, product, keep: set[str] | None):
    path = f"s3://{bucket}/{prefix}mrms/{product}/watershed_mean.parquet"
    log.info("Reading %s", path)
    df = pd.read_parquet(path, columns=["site_no", "datetime_utc", "value_mean"])
    df["site_no"] = df["site_no"].astype(str)
    if keep is not None:
        df = df[df["site_no"].isin(keep)]
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.dropna(subset=["value_mean"])
    return {sid: g.set_index("datetime_utc")["value_mean"].sort_index()
            for sid, g in df.groupby("site_no")}


# ══ Main ══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="QPE_01H_Pass2")
    ap.add_argument("--universe-key", default="analysis/event_confusion_matrix_tc.parquet")
    ap.add_argument("--all", action="store_true",
                    help="use ALL watershed-mean stations instead of the 106")
    ap.add_argument("--method", choices=["atsite", "regional"], default="atsite",
                    help="atsite = GEV per station (no pooling, default); "
                         "regional = index-flood L-moment pooling")
    ap.add_argument("--min-years", type=int, default=MIN_YEARS)
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC)
    ap.add_argument("--nsim", type=int, default=NSIM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.seed)

    cfg = load_config()
    bucket, prefix = cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
    rng = np.random.default_rng(args.seed)

    keep = None if args.all else load_universe(bucket, prefix, args.universe_key)
    series = load_watershed_series(bucket, prefix, args.product, keep)
    log.info("Watershed-mean series loaded for %d stations", len(series))

    ddf_rows, site_diag, region_diag = [], [], []
    for d in DURATIONS_HR:
        sites = []
        for sid, s in series.items():
            ams = annual_maxima(s, d, args.min_frac)
            if len(ams) >= args.min_years:
                sites.append({"site_no": sid, "ams": ams})
        if len(sites) < 5:
            log.warning("D=%dh: only %d eligible stations — skipping", d, len(sites))
            continue

        if args.method == "regional":
            region, R, site_rows = regional_fit(sites, args.nsim, rng)
            R["discordant"] = R["D"] > 3.0
            region["duration_hr"] = d
            region_diag.append(region)
            log.info("D=%2dh | %3d sites | tR=%.3f t3R=%.3f k=%.3f | H=%.2f%s | median n=%d yr",
                     d, region["n_sites"], region["tR"], region["t3R"], region["gev_k"],
                     region["H"], "  (heterogeneous!)" if region["H"] >= 2 else "",
                     region["n_years_median"])
        else:  # atsite
            site_rows, R = atsite_fit(sites)
            log.info("D=%2dh | %3d sites | at-site GEV | median k=%.3f | median n=%d yr",
                     d, len(R), float(R["gev_k"].median()), int(R["n"].median()))
        for row in site_rows:
            row["duration_hr"] = d
            ddf_rows.append(row)
        R["duration_hr"] = d
        site_diag.append(R)

    if not ddf_rows:
        log.error("No DDF produced — check inputs.")
        return

    ddf = pd.DataFrame(ddf_rows)[["site_no", "duration_hr", "return_period_yr", "depth_in"]]
    write_parquet_to_s3(ddf, bucket, f"{prefix}mrms/{args.product}/areal_precip_frequency.parquet")
    write_parquet_to_s3(pd.concat(site_diag, ignore_index=True), bucket,
                        f"{prefix}analysis/areal_ddf_site_diag.parquet")
    log.info("Wrote areal DDF (%d rows, method=%s) + site diagnostics to s3://%s/%smrms/%s/",
             len(ddf), args.method, bucket, prefix, args.product)

    if region_diag:                                        # regional mode only
        write_parquet_to_s3(pd.DataFrame(region_diag), bucket,
                            f"{prefix}analysis/areal_ddf_region_diag.parquet")
        het = [r["duration_hr"] for r in region_diag if r["H"] >= 2]
        if het:
            log.warning("Durations flagged HETEROGENEOUS (H>=2): %s — consider sub-regioning "
                        "before trusting these growth curves.", het)


# ══ Self-test ═════════════════════════════════════════════════════════════════

def selftest(seed: int) -> None:
    rng = np.random.default_rng(seed)
    print("── L-moment / GEV recovery ──")
    xi0, a0, k0 = 2.0, 0.5, -0.10               # true GEV (heavy-ish tail)
    x = gev_quantile(rng.uniform(size=20000), xi0, a0, k0)
    l1, l2, t, t3, t4 = sample_lmoments(x)
    xi, a, k = gev_params_from_lmom(l1, l2, t3)
    print(f"  true  xi={xi0:.3f} alpha={a0:.3f} k={k0:.3f}")
    print(f"  fit   xi={xi:.3f} alpha={a:.3f} k={k:.3f}")
    assert abs(xi - xi0) < 0.05 and abs(a - a0) < 0.05 and abs(k - k0) < 0.05, "GEV recovery off"
    q100_fit = gev_quantile(1 - 1/100, xi, a, k)
    q100_emp = np.quantile(x, 0.99)
    print(f"  Q100  fit={q100_fit:.3f}  empirical={q100_emp:.3f}")
    assert abs(q100_fit - q100_emp) / q100_emp < 0.05, "Q100 off"

    print("── Heterogeneity H: homogeneous region ~0, heterogeneous >2 ──")
    ns = [24] * 30
    homog = [{"site_no": str(i),
              "ams": gev_quantile(rng.uniform(size=24), 2.0, 0.5, -0.1)} for i in range(30)]
    Hh = regional_fit(homog, 200, rng)[0]["H"]
    # heterogeneous: vary the scale (=> vary L-CV) strongly across sites
    heterog = [{"site_no": str(i),
                "ams": gev_quantile(rng.uniform(size=24), 2.0, 0.2 + 0.6 * (i / 29), -0.1)}
               for i in range(30)]
    He = regional_fit(heterog, 200, rng)[0]["H"]
    print(f"  H(homogeneous)   = {Hh:.2f}   (expect < ~2)")
    print(f"  H(heterogeneous) = {He:.2f}   (expect > 2)")
    assert Hh < 2.0 < He, "H statistic not discriminating as expected"
    print("\n✓ selftest passed")


if __name__ == "__main__":
    main()
