"""04b_regression_flows.py  — at-site LP3 flood frequency analysis

Fits a Log-Pearson Type III (LP3) distribution to observed annual maximum
instantaneous discharge for Indiana gauges that lack USGS StreamStats estimates.

Methodology: Bulletin 17C (USGS TM 4-B5, 2019), Chapter 3 — Method of Moments
with weighted skew (Section 3.3):

    Ĝ = w₁·G + w₂·Ḡ
    w₁ = MSE(Ḡ) / (MSE(G) + MSE(Ḡ))   [weight on at-site]
    w₂ = MSE(G)  / (MSE(G) + MSE(Ḡ))   [weight on regional]

where:
    G    = at-site skew (from station record)
    Ḡ    = generalized regional skew (Weaver & Vogel 2010, spatially varying)
    MSE(G)  = at-site skew variance — B17C Appendix 3: 6(1 + 6·G²)/n
    MSE(Ḡ) = 0.302 (Weaver & Vogel 2010, national MSE of generalized skew map)

Station eligibility
───────────────────
  source == 'gage_stats'   → untouched (StreamStats values kept as-is)
  record < 10 water-years  → source set to 'insufficient_record'; Q values null;
                              downstream scripts should exclude these stations
  record ≥ 10 water-years  → LP3 fitted; source set to 'lp3_at_site'

A water year (Oct 1 – Sep 30) is counted if it contains ≥ 100 valid IV readings.

Outputs
───────
  s3://<bucket>/<prefix>flow_stats/per_gauge_flow_stats.parquet   (updated)
  s3://<bucket>/<prefix>analysis/lp3_frequency_curves/<site>_lp3.png   (one per station)
  s3://<bucket>/<prefix>analysis/lp3_frequency_curves/lp3_summary.csv
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import integrate, stats

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("04b_lp3")

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_YEARS        = 10     # minimum complete water years required
MIN_OBS_PER_YEAR = 100    # IV readings needed to count a water year as complete
RETURN_PERIODS   = [10, 25, 50, 100]
Q_COLS           = {10: "Q10", 25: "Q25", 50: "Q50", 100: "Q100"}
S3_PLOTS_PREFIX  = "analysis/lp3_frequency_curves/"

# Regional skew constants (Weaver & Vogel 2010 / B17C Section 3.3)
# MSE of the national generalized skew map — fixed at 0.302 per B17C.
REGIONAL_SKEW_MSE = 0.302

# Approximate generalized skew values for Indiana derived from the Weaver &
# Vogel (2010) national grid (B17C Figure 3.1).  Values vary smoothly from
# ~0.25 in the north to ~0.40 in the south.  This lookup can be replaced with
# the full USGS raster (USGS SIR 2010-5033) if higher spatial precision is
# needed.  Format: (lat_south, lat_north, skew_south, skew_north)
_IN_SKEW_BANDS = [
    (37.5, 38.5, 0.42, 0.38),
    (38.5, 39.5, 0.38, 0.32),
    (39.5, 40.5, 0.32, 0.27),
    (40.5, 42.0, 0.27, 0.23),
]


# ── Bulletin 17C LP3 core (Chapter 3, Wilson-Hilferty approximation) ──────────

def k_factor(exceedance_prob: float, skew: float) -> float:
    """Frequency factor K via Wilson-Hilferty approximation (B17C Appendix 3).

    Args:
        exceedance_prob: 1 / return_period (annual exceedance probability)
        skew: coefficient of skewness of the log10(Q) series
    """
    z = stats.norm.ppf(1.0 - exceedance_prob)
    k = skew / 6.0
    return (z
            + (z**2 - 1) * k
            + (z**3 - 6*z) * k**2 / 3.0
            - (z**2 - 1) * k**3
            + z * k**4
            + k**5 / 3.0)


def lp3_quantile(return_period: float, mean_log: float, std_log: float, skew: float) -> float:
    """Q_T (cfs) for a given return period using fitted LP3 parameters."""
    K = k_factor(1.0 / return_period, skew)
    return 10.0 ** (mean_log + K * std_log)


def generalized_skew(lat: float) -> float:
    """Spatially interpolated generalized skew Ḡ for Indiana (Weaver & Vogel 2010)."""
    for lat_s, lat_n, g_s, g_n in _IN_SKEW_BANDS:
        if lat_s <= lat <= lat_n:
            t = (lat - lat_s) / (lat_n - lat_s)
            return g_s + t * (g_n - g_s)
    # Clamp to nearest band edge
    return _IN_SKEW_BANDS[0][2] if lat < _IN_SKEW_BANDS[0][0] else _IN_SKEW_BANDS[-1][3]


def at_site_skew_mse(n: int, g: float) -> float:
    """Variance of the at-site skew estimator (B17C Appendix 3, Eq. A3-3)."""
    return 6.0 * (1.0 + 6.0 * g**2) / n


def weighted_skew(g_site: float, n: int, lat: float) -> tuple[float, float, float, float]:
    """Bulletin 17C weighted skew (Section 3.3).

    Returns (skew_weighted, skew_regional, w_site, w_regional).
    """
    g_regional  = generalized_skew(lat)
    mse_site    = at_site_skew_mse(n, g_site)
    mse_reg     = REGIONAL_SKEW_MSE
    w_site      = mse_reg  / (mse_site + mse_reg)
    w_regional  = mse_site / (mse_site + mse_reg)
    g_weighted  = w_site * g_site + w_regional * g_regional
    return g_weighted, g_regional, w_site, w_regional


# ── Grubbs-Beck low outlier test (B17C Section 4.3) ──────────────────────────

def _gb_critical_value(n: int) -> float:
    """Grubbs-Beck critical value at α=0.10 (B17B Table 3 polynomial fit).

    KN_crit = A + B·√(log₁₀ n) + C·log₁₀ n   (Pierson et al. 2001)
    Verified against B17B Table 3 for n = 10–149; extrapolated beyond that.
    """
    L = np.log10(max(n, 3))
    return -0.9043 + 3.345 * np.sqrt(L) - 0.4046 * L


def grubbs_beck_test(log_q: np.ndarray) -> tuple[int, float | None]:
    """Iterative Grubbs-Beck test for low outliers (B17C Section 4.3, α=0.10).

    At each step the smallest remaining value is tested.  Flagged values are
    treated as perceptually censored (not deleted) and passed to EMA.  The loop
    stops when no more outliers are detected or fewer than MIN_YEARS values remain.

    Returns
    -------
    n_censored : int  — how many low values are censored
    threshold  : float | None — log₁₀ of the smallest uncensored value (= EMA
                 perception threshold), or None when n_censored == 0.
    """
    sorted_q = np.sort(log_q)
    n_censor = 0

    while True:
        remaining = sorted_q[n_censor:]
        n = len(remaining)
        if n < MIN_YEARS:
            if n_censor > 0:
                n_censor -= 1       # roll back: keep at least MIN_YEARS uncensored
            break

        mean_y = float(np.mean(remaining))
        std_y  = float(np.std(remaining, ddof=1))
        if std_y == 0:
            break

        KN      = (mean_y - remaining[0]) / std_y
        KN_crit = _gb_critical_value(n)

        if KN > KN_crit:
            n_censor += 1
        else:
            break

    if n_censor == 0:
        return 0, None

    return n_censor, float(sorted_q[n_censor])   # threshold = smallest uncensored


# ── EMA (Expected Moments Algorithm, Cohn et al. 1997, B17C Section 3.2) ─────

def _truncated_central_moment(
    k: int,
    threshold: float,
    center: float,
    gamma: float,
    loc: float,
    scale: float,
) -> float:
    """E[(Y - center)^k | Y < threshold] for Pearson3(gamma, loc, scale).

    Returns NaN if the integration is numerically unstable; the caller treats a
    non-finite result as an EMA failure and falls back to method of moments.
    """
    if not (np.isfinite(loc) and np.isfinite(scale) and scale > 1e-9
            and np.isfinite(gamma)):
        return float("nan")

    dist = stats.pearson3(gamma, loc=loc, scale=scale)
    F_T  = float(dist.cdf(threshold))
    if not np.isfinite(F_T) or F_T < 1e-12:
        return float("nan")

    lower = float(dist.ppf(max(1e-9, F_T * 1e-6)))
    if not np.isfinite(lower):
        return float("nan")

    try:
        val, _ = integrate.quad(
            lambda y: float((y - center) ** k * dist.pdf(y)),
            lower,
            threshold,
            limit=200,
            epsabs=1e-9,
            epsrel=1e-7,
        )
    except Exception:
        return float("nan")

    return val / F_T if np.isfinite(val) else float("nan")


def ema_fit(
    log_q:    np.ndarray,
    threshold: float,
    max_iter:  int   = 100,
    tol:       float = 1e-7,
) -> tuple[float, float, float, dict]:
    """EMA for LP3 with n_c observations censored below *threshold* (log₁₀ space).

    Implements Cohn et al. (1997) / B17C Section 3.2.  When n_c = 0, converges
    to ordinary MOM.  Caller is responsible for applying weighted skew.

    Returns (mean_log, std_log, at_site_skew, info_dict).
    """
    uncensored = log_q[log_q >= threshold]
    n_c = int(np.sum(log_q < threshold))
    n_u = len(uncensored)
    n   = n_u + n_c

    if n_u < 4:
        mu    = float(np.mean(log_q))
        sigma = float(np.std(log_q, ddof=1))
        gamma = float(stats.skew(log_q, bias=False))
        return mu, sigma, gamma, {
            "n_censored": n_c, "n_uncensored": n_u,
            "threshold_log": threshold, "converged": False, "iterations": 0,
        }

    # Initialize from MOM on uncensored data
    mu    = float(np.mean(uncensored))
    sigma = float(np.std(uncensored, ddof=1))
    gamma = float(stats.skew(uncensored, bias=False))

    converged = False
    failed    = False
    n_iter    = 0

    for n_iter in range(1, max_iter + 1):
        # EMA mean: μ̂ = (Σᵤ yᵢ + n_c · E[Y | Y<T]) / n
        e1     = _truncated_central_moment(1, threshold, 0.0, gamma, mu, sigma)
        mu_new = (float(np.sum(uncensored)) + n_c * e1) / n

        # EMA variance: σ̂² = (Σᵤ(yᵢ−μ̂)² + n_c · E[(Y−μ̂)² | Y<T]) / (n−1)
        e2        = _truncated_central_moment(2, threshold, mu_new, gamma, mu, sigma)
        var_new   = (float(np.sum((uncensored - mu_new) ** 2)) + n_c * e2) / (n - 1)
        sigma_new = float(np.sqrt(max(var_new, 1e-12)))

        # EMA skew: ĝ = n/((n−1)(n−2)) · (Σᵤ Zᵢ³ + n_c · E[Z³ | Y<T])
        if sigma_new > 1e-6:
            e3        = _truncated_central_moment(3, threshold, mu_new, gamma, mu, sigma)
            sum_z3_u  = float(np.sum(((uncensored - mu_new) / sigma_new) ** 3))
            gamma_new = (n / ((n - 1.0) * (n - 2.0))) * (
                sum_z3_u + n_c * e3 / sigma_new ** 3
            )
        else:
            gamma_new = gamma   # degenerate variance — hold skew fixed

        # Abort if any truncated moment was numerically unstable
        if not (np.isfinite(mu_new) and np.isfinite(sigma_new)
                and np.isfinite(gamma_new)):
            failed = True
            break

        delta = abs(mu_new - mu) + abs(sigma_new - sigma) + abs(gamma_new - gamma)
        mu, sigma, gamma = mu_new, sigma_new, gamma_new

        if delta < tol:
            converged = True
            break

    # Fall back to full-sample method of moments if EMA was unstable
    if failed:
        mu    = float(np.mean(log_q))
        sigma = float(np.std(log_q, ddof=1))
        gamma = float(stats.skew(log_q, bias=False))

    return mu, sigma, gamma, {
        "n_censored":    n_c,
        "n_uncensored":  n_u,
        "threshold_log": threshold,
        "converged":     converged,
        "ema_failed":    failed,
        "iterations":    n_iter,
    }


def fit_lp3(log_q: np.ndarray, lat: float) -> dict:
    """Full B17C LP3 workflow: GB low outlier test → EMA or MOM → weighted skew.

    B17C Section 4.3 (Grubbs-Beck) → Section 3.2 (EMA) → Section 3.3 (weighted skew).
    """
    # Step 1: Grubbs-Beck low outlier detection
    n_censor, threshold = grubbs_beck_test(log_q)

    # Step 2: Moment estimation
    if n_censor > 0:
        mean_y, std_y, skew_site, ema_info = ema_fit(log_q, threshold)
        fitting_method = "MOM (EMA unstable)" if ema_info.get("ema_failed") else "EMA"
    else:
        mean_y    = float(np.mean(log_q))
        std_y     = float(np.std(log_q, ddof=1))
        skew_site = float(stats.skew(log_q, bias=False))
        ema_info  = {
            "n_censored": 0, "n_uncensored": len(log_q),
            "threshold_log": None, "converged": True, "iterations": 0,
        }
        fitting_method = "MOM"

    # Step 3: Weighted skew using total n for the MSE(G) denominator (B17C §3.3)
    n_total = len(log_q)
    g_w, g_reg, w_s, w_r = weighted_skew(skew_site, n_total, lat)

    return {
        "n":              n_total,
        "n_censored":     ema_info["n_censored"],
        "n_uncensored":   ema_info["n_uncensored"],
        "fitting_method": fitting_method,
        "threshold_log":  ema_info["threshold_log"],
        "mean_log":       mean_y,
        "std_log":        std_y,
        "skew":           g_w,           # weighted — used for all quantile estimates
        "skew_at_site":   skew_site,     # EMA or MOM at-site skew
        "skew_regional":  g_reg,
        "weight_at_site":  round(w_s, 3),
        "weight_regional": round(w_r, 3),
    }


# ── Annual maximum extraction ─────────────────────────────────────────────────

def _water_year(dt: pd.Timestamp) -> int:
    return dt.year + 1 if dt.month >= 10 else dt.year


def extract_annual_maxima(site_ts: pd.DataFrame) -> pd.Series:
    """Annual maximum instantaneous discharge indexed by water year.

    Excludes water years with fewer than MIN_OBS_PER_YEAR valid readings.
    """
    ts = site_ts[site_ts["value_cfs"] > 0].copy()
    if ts.empty:
        return pd.Series(dtype=float)
    ts["wy"]  = ts["datetime"].apply(_water_year)
    counts    = ts.groupby("wy")["value_cfs"].count()
    maxima    = ts.groupby("wy")["value_cfs"].max()
    valid_wy  = counts[counts >= MIN_OBS_PER_YEAR].index
    return maxima.loc[valid_wy].dropna()


# ── Goodness-of-fit ───────────────────────────────────────────────────────────

def compute_gof(log_q: np.ndarray, params: dict) -> dict:
    """PPCC, RMSE (log space), and KS test vs fitted LP3.

    For censored data, Weibull plotting positions are shifted so that the
    smallest uncensored observation starts at rank (n_c + 1) of the full n+1
    denominator.  GoF is evaluated on uncensored values only.
    """
    n_total   = params["n"]
    n_c       = params.get("n_censored", 0)
    mean_y    = params["mean_log"]
    std_y     = params["std_log"]
    skew      = params["skew"]
    threshold = params.get("threshold_log")

    # Use uncensored observations only
    if threshold is not None:
        fit_data = np.sort(log_q[log_q >= threshold])
    else:
        fit_data = np.sort(log_q)

    n_fit = len(fit_data)
    m     = np.arange(1, n_fit + 1)
    # Adjusted Weibull exceedance: F(y_(n_c+m)) = (n_c+m)/(n_total+1)
    p_exc    = 1.0 - (n_c + m) / (n_total + 1)
    fitted_y = np.array([mean_y + k_factor(float(p), skew) * std_y for p in p_exc])

    ppcc     = float(np.corrcoef(fit_data, fitted_y)[0, 1])
    rmse_log = float(np.sqrt(np.mean((fit_data - fitted_y) ** 2)))

    dist             = stats.pearson3(skew, loc=mean_y, scale=std_y)
    ks_stat, ks_pval = stats.ks_1samp(fit_data, dist.cdf)

    return {
        "ppcc":     round(ppcc, 4),
        "rmse_log": round(rmse_log, 4),
        "ks_stat":  round(float(ks_stat), 4),
        "ks_pval":  round(float(ks_pval), 4),
    }


# ── Frequency curve plot ──────────────────────────────────────────────────────

def plot_frequency_curve(
    site_no:     str,
    annual_max:  pd.Series,
    params:      dict,
    q_estimates: dict[int, float],
    gof:         dict,
) -> plt.Figure:
    """Return period vs discharge (log-log) with observed Weibull positions.

    Censored values (Grubbs-Beck low outliers) are plotted as open downward
    triangles at the censoring threshold; uncensored values use adjusted
    Weibull plotting positions that account for the censored observations.
    """
    n_total   = params["n"]
    n_c       = params.get("n_censored", 0)
    n_u       = params.get("n_uncensored", n_total)
    threshold = params.get("threshold_log")
    mean_y    = params["mean_log"]
    std_y     = params["std_log"]
    skew      = params["skew"]
    method    = params.get("fitting_method", "MOM")

    all_q  = np.sort(annual_max.values)
    log_all = np.log10(all_q)

    if threshold is not None:
        uncensored_q = all_q[log_all >= threshold]
        censored_q   = all_q[log_all < threshold]
    else:
        uncensored_q = all_q
        censored_q   = np.array([])

    # Adjusted Weibull plotting positions for uncensored observations
    m_u   = np.arange(1, n_u + 1)
    T_emp = (n_total + 1) / (n_total + 1 - (n_c + m_u))

    # Fitted LP3 curve
    T_fit = np.logspace(np.log10(1.01), np.log10(2000), 400)
    Q_fit = np.array([lp3_quantile(T, mean_y, std_y, skew) for T in T_fit])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(T_emp, uncensored_q, s=45, color="steelblue", zorder=5,
               label="Observed — uncensored (Weibull)", edgecolors="white", lw=0.4)

    if len(censored_q) > 0:
        T_cens  = 10 ** threshold
        # Plot censored observations as triangles at the threshold line
        ax.scatter(
            [1.5] * len(censored_q),  # arbitrary x position for display
            [T_cens] * len(censored_q),
            marker="v", s=60, color="none", edgecolors="grey", lw=1.2,
            zorder=5, label=f"Censored (GB low outlier, n={n_c})",
        )
        ax.axhline(T_cens, color="grey", ls=":", lw=1.0,
                   label=f"Censoring threshold = {T_cens:,.0f} cfs")

    ax.plot(T_fit, Q_fit, color="firebrick", lw=2, label="Fitted LP3")

    # Mark design return periods
    rp_colors = {10: "goldenrod", 25: "darkorange", 50: "coral", 100: "red"}
    for rp in RETURN_PERIODS:
        q_val = q_estimates.get(rp)
        if q_val is None:
            continue
        c = rp_colors.get(rp, "grey")
        ax.axvline(rp, color=c, ls="--", lw=0.9, alpha=0.7)
        ax.scatter([rp], [q_val], marker="D", s=50, color=c, zorder=6)
        ax.text(rp * 1.05, q_val * 1.1,
                f"Q{rp}\n{q_val:,.0f} cfs",
                fontsize=7.5, color=c, va="bottom")

    # Annotation box
    skew_site = params.get("skew_at_site", skew)
    skew_reg  = params.get("skew_regional", skew)
    w_s       = params.get("weight_at_site", 1.0)
    w_r       = params.get("weight_regional", 0.0)
    cens_line = (
        f"GB censored     = {n_c} below {10**threshold:,.0f} cfs\n"
        if threshold is not None else ""
    )
    txt = (
        f"n = {n_total} water-years  [{method}]\n"
        + cens_line
        + f"Mean(log Q)     = {mean_y:.3f}\n"
        f"Std(log Q)      = {std_y:.3f}\n"
        f"{'─' * 28}\n"
        f"Skew at-site    = {skew_site:.3f}  (w={w_s:.2f})\n"
        f"Skew regional   = {skew_reg:.3f}  (w={w_r:.2f})\n"
        f"Skew weighted   = {skew:.3f}\n"
        f"{'─' * 28}\n"
        f"PPCC            = {gof['ppcc']:.4f}\n"
        f"RMSE (log)      = {gof['rmse_log']:.4f}\n"
        f"KS stat         = {gof['ks_stat']:.4f}  (p = {gof['ks_pval']:.3f})"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.88, ec="lightgrey"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Return Period (years)", fontsize=11)
    ax.set_ylabel("Peak Discharge (cfs)", fontsize=11)
    ax.set_title(
        f"LP3 Frequency Curve — Station {site_no}\n"
        f"Bulletin 17C {method}, weighted skew "
        f"(at-site w={w_s:.2f} / regional w={w_r:.2f})",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25, ls="--")
    fig.tight_layout()
    return fig


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str, columns: list | None = None) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def _upload_png(fig: plt.Figure, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf, ContentType="image/png")
    log.info("    → s3://%s/%s", bucket, key)


def _upload_csv(df: pd.DataFrame, bucket: str, key: str) -> None:
    s3_client().put_object(
        Bucket=bucket, Key=key,
        Body=df.to_csv(index=False).encode(),
        ContentType="text/csv",
    )
    log.info("  → s3://%s/%s", bucket, key)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    # Load current flow stats
    flow_stats = _read_parquet_s3(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    flow_stats["site_no"] = flow_stats["site_no"].astype(str)

    # Stations to (re)fit with LP3:
    #   • every non-gage_stats station (as before), plus
    #   • gage_stats stations with an incomplete Q set (Q10/Q25/Q50/Q100) —
    #     these had gaps the old regression filled; refit entirely with LP3 so
    #     a single, internally consistent method drives all return periods.
    # gage_stats stations with a COMPLETE Q set are left untouched (authoritative).
    core_q   = [c for c in Q_COLS.values() if c in flow_stats.columns]
    is_gage  = flow_stats["source"] == "gage_stats"
    has_gap  = flow_stats[core_q].isna().any(axis=1)

    needs_lp3      = flow_stats[(~is_gage) | (is_gage & has_gap)]["site_no"].tolist()
    gage_gap_sites = set(flow_stats[is_gage & has_gap]["site_no"])

    n_gage_complete = int((is_gage & ~has_gap).sum())
    log.info("Stations to evaluate: %d (incl. %d gage_stats with gaps to refit) "
             "| complete gage_stats kept: %d",
             len(needs_lp3), len(gage_gap_sites), n_gage_complete)

    if not needs_lp3:
        log.info("All stations have a complete gage_stats Q set — nothing to do.")
        return

    # Load station coordinates (needed for regional skew lookup)
    inv = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet",
                           columns=["site_no", "dec_lat_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    lat_map = inv.set_index("site_no")["dec_lat_va"].to_dict()

    # Load IV streamflow for target stations only
    log.info("Loading IV streamflow data...")
    sf_all = _read_parquet_s3(
        bucket,
        f"{prefix}streamflow/instantaneous/all_gauges_long.parquet",
        columns=["site_no", "datetime", "value_cfs"],
    )
    sf_all["site_no"]   = sf_all["site_no"].astype(str)
    sf_all["datetime"]  = pd.to_datetime(sf_all["datetime"], utc=True)
    sf_all["value_cfs"] = pd.to_numeric(sf_all["value_cfs"], errors="coerce")
    sf_sub = sf_all[sf_all["site_no"].isin(set(needs_lp3))].copy()
    log.info("IV records loaded: {:,}".format(len(sf_sub)))

    # Process stations
    summary_rows: list[dict] = []
    flow_stats = flow_stats.set_index("site_no")

    n_total = len(needs_lp3)
    for i, site_no in enumerate(needs_lp3, 1):
        ts      = sf_sub[sf_sub["site_no"] == site_no]
        ann_max = extract_annual_maxima(ts)
        n_years = len(ann_max)

        log.info("[%d/%d] %s — %d water-years", i, n_total, site_no, n_years)

        if n_years < MIN_YEARS:
            # A gage_stats-with-gaps station too short for LP3: keep its existing
            # published StreamStats values rather than wiping good data.
            if site_no in gage_gap_sites:
                log.info("    gage_stats with gaps but < %d yr — keeping published values",
                         MIN_YEARS)
                summary_rows.append({
                    "site_no": site_no, "n_years": n_years,
                    "status": "gage_stats_gaps_kept",
                })
                continue

            flow_stats.at[site_no, "source"] = "insufficient_record"
            for col in Q_COLS.values():
                if col in flow_stats.columns:
                    flow_stats.at[site_no, col] = np.nan
            summary_rows.append({
                "site_no": site_no, "n_years": n_years,
                "status": "insufficient_record",
            })
            continue

        lat    = float(lat_map.get(site_no, 39.8))  # fallback to Indiana centroid
        log_q  = np.log10(ann_max.values)
        params = fit_lp3(log_q, lat)
        gof    = compute_gof(log_q, params)

        q_estimates: dict[int, float] = {}
        for rp in RETURN_PERIODS:
            q_t = lp3_quantile(rp, params["mean_log"], params["std_log"], params["skew"])
            q_estimates[rp] = round(q_t, 1)
            if Q_COLS[rp] in flow_stats.columns:
                flow_stats.at[site_no, Q_COLS[rp]] = q_t

        flow_stats.at[site_no, "source"] = "lp3_at_site"

        summary_rows.append({
            "site_no":          site_no,
            "n_years":          n_years,
            "status":           "lp3_refit_from_gage" if site_no in gage_gap_sites
                                else "lp3_fitted",
            "fitting_method":   params.get("fitting_method", "MOM"),
            "n_censored":       params.get("n_censored", 0),
            "threshold_cfs":    (
                round(10 ** params["threshold_log"], 1)
                if params.get("threshold_log") is not None else None
            ),
            "mean_log":         round(params["mean_log"],       4),
            "std_log":          round(params["std_log"],        4),
            "skew_at_site":     round(params["skew_at_site"],  4),
            "skew_regional":    round(params["skew_regional"], 4),
            "weight_at_site":   params["weight_at_site"],
            "weight_regional":  params["weight_regional"],
            "skew_weighted":    round(params["skew"],          4),
            **{f"Q{rp}": q_estimates.get(rp) for rp in RETURN_PERIODS},
            **gof,
        })

        try:
            fig = plot_frequency_curve(site_no, ann_max, params, q_estimates, gof)
            _upload_png(fig, bucket,
                        f"{prefix}{S3_PLOTS_PREFIX}{site_no}_lp3.png")
            plt.close(fig)
        except Exception as exc:
            log.warning("    Plot failed: %s", exc)

    # Save updated flow stats
    flow_stats = flow_stats.reset_index()
    write_parquet_to_s3(flow_stats, bucket,
                        f"{prefix}flow_stats/per_gauge_flow_stats.parquet")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    _upload_csv(summary_df, bucket,
                f"{prefix}{S3_PLOTS_PREFIX}lp3_summary.csv")

    n_fitted       = int((summary_df["status"] == "lp3_fitted").sum())
    n_refit_gage   = int((summary_df["status"] == "lp3_refit_from_gage").sum())
    n_gage_kept    = int((summary_df["status"] == "gage_stats_gaps_kept").sum())
    n_insufficient = int((summary_df["status"] == "insufficient_record").sum())
    log.info("LP3 fitted: %d  |  LP3 refit from gage gaps: %d  |  "
             "gage gaps kept (< %d yr): %d  |  Insufficient record: %d  |  Total: %d",
             n_fitted, n_refit_gage, MIN_YEARS, n_gage_kept, n_insufficient, n_total)
    log.info("Source counts: %s",
             flow_stats["source"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
