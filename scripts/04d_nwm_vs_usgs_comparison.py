"""04d — NWM retrospective vs USGS observed-peak LP3, as queryable tables + figures.

Why this exists
───────────────
04b fits LP3 to USGS instantaneous annual peaks; 04c fits the same B17C machinery
to NWM-retrospective water-year maxima.  Until now the two could only be compared
by eye, PNG against PNG, because neither persisted its fit parameters (fixed in
this commit's 04b/04c change).  And 04c's station universe was never the set of
stations that CAN be compared — it is 08c's trigger-study output:

    TC_KEY = "analysis/event_confusion_matrix_tc.parquet"   # 08c output → 106 stations

That set was filtered by clustering and the trigger study's own needs, which has
nothing to do with having enough annual maxima to fit.  04b fits 162 stations;
158 of those have a COMID.  So ~53 comparable stations — including discontinued
gauges with long records — were simply never compared.

What it does
────────────
Re-fits BOTH sides here, in one run, with b04b.fit_lp3, so the two are
guaranteed to come from identical code rather than from two scripts that might
have drifted.  Where 04b/04c stored a fit for the same station, the re-fit is
checked against it and the worst deviation reported — if that is not ~0, this
comparison is not describing the fits in production and nothing below is
trustworthy (the lesson from p08's reproduction check).

NWM annual maxima are taken from monitor/precompute/lp3_annual_maxima.parquet
(p08, keyed by COMID, 6,865 reaches) which already covers 153 of the 158, falling
back to 04c's nwm/retrospective_full.parquet cache.  Neither path touches the
Zarr, so this runs in minutes.

Three comparisons, each a table AND a figure
────────────────────────────────────────────
  1. fit parameters   mean_log, std_log, at-site & weighted skew, the B17C weight
  2. quantiles        Q2…Q500 and the NWM/USGS ratio at every return period
  3. annual maxima    the series themselves, paired by water year

Outputs (nothing existing is overwritten; all under analysis/nwm_vs_usgs/)
    fit_comparison.parquet / .csv              one row per station
    annual_maxima_paired.parquet / .csv        one row per (station, water year)
    annual_maxima_station_stats.parquet / .csv one row per station
    fig_parameters.png  fig_quantiles.png  fig_annual_maxima.png

Rule B note: 04b drops any station carrying a regulation/diversion peak flag, so
a regulated station has no observed-peak fit and cannot enter the parameter or
quantile comparison.  The ANNUAL-MAXIMA comparison needs no fit, so regulated
stations are included there and flagged `is_regulated` for filtering.
--include-regulated additionally fits them for the parameter/quantile tables.

Usage
─────
    python 04d_nwm_vs_usgs_comparison.py
    python 04d_nwm_vs_usgs_comparison.py --include-regulated
    python 04d_nwm_vs_usgs_comparison.py --dry-run        # compute + report, write nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

from utils import load_config, write_parquet_to_s3

_spec = importlib.util.spec_from_file_location(
    "regression_flows_04b", Path(__file__).with_name("04b_regression_flows.py"))
b04b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b04b)

_spec_c = importlib.util.spec_from_file_location(
    "nwm_regression_flows_04c", Path(__file__).with_name("04c_nwm_regression_flows.py"))
m04c = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(m04c)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("04d_nwm_vs_usgs")


def _config() -> dict:
    """Load config.yaml from the repo root, whatever the working directory is.

    utils.load_config defaults to a cwd-relative "config.yaml", so running this
    from scripts/ — the obvious place to run it from — fails with a bare
    FileNotFoundError.  A cwd-local config still wins if one is present.
    """
    from pathlib import Path as _P
    local = _P("config.yaml")
    if local.exists():
        return load_config()
    return load_config(_P(__file__).resolve().parents[1] / "config.yaml")

# ── Keys ─────────────────────────────────────────────────────────────────────
PEAKS_KEY   = b04b.PEAKS_KEY
COMID_KEY   = "nwm/comid_locations.parquet"
INV_KEY     = "stations/indiana_streamflow_sites.parquet"
RETRO_FULL  = "nwm/retrospective_full.parquet"
P08_AMS_KEY = "monitor/precompute/lp3_annual_maxima.parquet"
USGS_STORED = "flow_stats/per_gauge_flow_stats.parquet"
NWM_STORED  = "flow_stats/nwm_per_gauge_flow_stats.parquet"
OUT_PREFIX  = "analysis/nwm_vs_usgs/"

RPS = b04b.ALL_RETURN_PERIODS          # [2, 5, 10, 25, 50, 100, 200, 500]

# ── Palette (dataviz reference instance, light surface) ──────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
SERIES_1  = "#2a78d6"     # blue  — categorical slot 1
SERIES_2  = "#eb6834"     # orange — slot 2
SERIES_3  = "#1baf7a"     # aqua  — slot 3  (scatter caps at three)
CRITICAL  = "#d03b3b"
# Diverging blue↔red with a NEUTRAL gray midpoint, for ratios centred on 1.0.
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "nwm_usgs_ratio", [SERIES_1, "#f0efec", CRITICAL])


def _log_ticks(ax, which: str, ticks, labels) -> None:
    """Pin explicit ticks on a log axis.

    A log scale keeps its own minor locator, which otherwise draws 6x10^-1 /
    3x10^0 alongside the readable multiples we actually want.
    """
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FixedFormatter(labels))
    axis.set_minor_locator(mticker.NullLocator())
    axis.set_minor_formatter(mticker.NullFormatter())


def _safe_r(x, y) -> float:
    """Pearson r, or NaN when a side has no variance (e.g. a constant)."""
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _style(ax, *, grid_axis="both"):
    """Hairline, solid, recessive chrome.  Dashed grids are an anti-pattern."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2); ax.yaxis.label.set_color(INK_2)


# ── Annual-maximum sources ───────────────────────────────────────────────────

def usgs_annual_maxima(bucket: str, prefix: str) -> tuple[dict, dict]:
    """(peaks_by_site, regulated_by_site) using 04b's own cleaning rules."""
    peaks = b04b._read_parquet_s3(bucket, f"{prefix}{PEAKS_KEY}")
    peaks["site_no"] = peaks["site_no"].astype(str)
    out, reg = {}, {}
    for site, g in peaks.groupby("site_no"):
        s, is_reg = b04b.clean_peaks_for_site(g)
        reg[site] = bool(is_reg)
        if len(s):
            out[site] = s
    log.info("USGS annual peaks: %d stations with a clean series (%d regulated)",
             len(out), sum(reg.values()))
    return out, reg


def nwm_annual_maxima(bucket: str, prefix: str, comid_by_site: dict) -> dict:
    """NWM water-year maxima per station, from p08's table then 04c's cache.

    p08's lp3_annual_maxima.parquet was produced by the same
    m04c.annual_max_series over the same retrospective window, so the two
    sources are interchangeable; p08 simply covers far more reaches.
    """
    out: dict[str, pd.Series] = {}
    try:
        ams = b04b._read_parquet_s3(bucket, f"{prefix}{P08_AMS_KEY}")
        by_comid = {int(c): g for c, g in ams.groupby("comid")}
        for site, comid in comid_by_site.items():
            g = by_comid.get(int(comid))
            if g is not None and len(g):
                out[site] = (g.set_index("water_year")["peak_cfs"]
                              .sort_index().astype(float))
        log.info("NWM maxima from p08 table: %d stations", len(out))
    except Exception as e:                                  # noqa: BLE001
        log.warning("p08 annual-maxima table unavailable (%s)", e)

    missing = [s for s in comid_by_site if s not in out]
    if missing:
        try:
            retro = b04b._read_parquet_s3(bucket, f"{prefix}{RETRO_FULL}")
            retro["site_no"] = retro["site_no"].astype(str)
            retro["datetime_utc"] = pd.to_datetime(retro["datetime_utc"], utc=True)
            got = 0
            for site, g in retro[retro["site_no"].isin(missing)].groupby("site_no"):
                ann = m04c.annual_max_series(g)
                if len(ann):
                    out[site] = ann.astype(float)
                    got += 1
            log.info("NWM maxima from 04c retro cache: +%d stations", got)
        except Exception as e:                              # noqa: BLE001
            log.warning("04c retrospective cache unavailable (%s)", e)

    still = [s for s in comid_by_site if s not in out]
    if still:
        log.warning("%d station(s) have no NWM annual maxima from either source "
                    "— excluded: %s", len(still), still[:10])
    return out


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit_side(ann: pd.Series, lat: float) -> dict | None:
    """Fit LP3 and return the flat parameter record plus every quantile."""
    if len(ann) < b04b.MIN_YEARS:
        return None
    log_q = np.log10(ann.values.astype(float))
    params = b04b.fit_lp3(log_q, lat)
    gof = b04b.compute_gof(log_q, params)
    rec = b04b.fit_param_record(params, gof)
    rec.update({"n": int(len(ann)),
                "wy_start": int(ann.index.min()), "wy_end": int(ann.index.max()),
                "max_cfs": float(np.max(ann.values)),
                "median_cfs": float(np.median(ann.values))})
    for rp in RPS:
        rec[f"Q{rp}"] = float(b04b.lp3_quantile(
            rp, params["mean_log"], params["std_log"], params["skew"]))
    return rec


PARAM_KEYS = ["n", "wy_start", "wy_end", "max_cfs", "median_cfs",
              *b04b.FIT_PARAM_COLS]


def build_fit_comparison(usgs_ams, nwm_ams, reg_by_site, comid_by_site,
                         dist_by_site, lat_map, include_regulated: bool) -> pd.DataFrame:
    rows = []
    for site in sorted(set(usgs_ams) & set(nwm_ams)):
        if reg_by_site.get(site) and not include_regulated:
            continue
        lat = float(lat_map.get(site, 39.8))
        u = fit_side(usgs_ams[site], lat)
        n = fit_side(nwm_ams[site], lat)
        if u is None or n is None:
            continue
        row = {"site_no": site, "comid": int(comid_by_site[site]),
               "distance_km": dist_by_site.get(site),
               "is_regulated": bool(reg_by_site.get(site, False)),
               "lat": lat}
        for k in PARAM_KEYS:
            row[f"usgs_{k}"] = u.get(k)
            row[f"nwm_{k}"] = n.get(k)
        for rp in RPS:
            row[f"usgs_Q{rp}"] = u[f"Q{rp}"]
            row[f"nwm_Q{rp}"] = n[f"Q{rp}"]
            row[f"ratio_Q{rp}"] = (n[f"Q{rp}"] / u[f"Q{rp}"]
                                   if u[f"Q{rp}"] else np.nan)
        # Signed differences, NWM minus USGS, for the parameters that matter.
        for k in ("mean_log", "std_log", "skew_at_site", "skew_weighted",
                  "weight_at_site"):
            row[f"d_{k}"] = _num(n.get(k)) - _num(u.get(k))
        # Overlapping water years — the two records do not cover the same period.
        wy_u = set(usgs_ams[site].index.astype(int))
        wy_n = set(nwm_ams[site].index.astype(int))
        row["n_overlap_wy"] = len(wy_u & wy_n)
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info("Fit comparison: %d stations", len(df))
    return df


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def build_ams_comparison(usgs_ams, nwm_ams, reg_by_site,
                         comid_by_site) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired annual maxima, and per-station agreement statistics.

    Pairing is by water year, so only years both records cover are compared —
    the USGS peak is instantaneous while the NWM value is the maximum of hourly
    routed output, which cannot resolve a sharp crest, so a negative bias is
    expected rather than surprising.
    """
    paired, stats_rows = [], []
    for site in sorted(set(usgs_ams) & set(nwm_ams)):
        u, n = usgs_ams[site], nwm_ams[site]
        wy = sorted(set(u.index.astype(int)) & set(n.index.astype(int)))
        if not wy:
            continue
        uu = u.reindex(wy).astype(float).values
        nn = n.reindex(wy).astype(float).values
        ok = np.isfinite(uu) & np.isfinite(nn) & (uu > 0) & (nn > 0)
        if ok.sum() < 3:
            continue
        lr = np.log10(nn[ok] / uu[ok])
        paired.append(pd.DataFrame({
            "site_no": site, "comid": int(comid_by_site[site]),
            "water_year": np.asarray(wy)[ok],
            "usgs_peak_cfs": uu[ok], "nwm_peak_cfs": nn[ok],
            "log10_ratio": lr,
            "is_regulated": bool(reg_by_site.get(site, False))}))
        iu, inn = int(np.argmax(uu[ok])), int(np.argmax(nn[ok]))
        wy_ok = np.asarray(wy)[ok]
        r_log = (float(stats.pearsonr(np.log10(uu[ok]), np.log10(nn[ok]))[0])
                 if ok.sum() >= 3 else np.nan)
        r_sp = (float(stats.spearmanr(uu[ok], nn[ok])[0]) if ok.sum() >= 3 else np.nan)
        stats_rows.append({
            "site_no": site, "comid": int(comid_by_site[site]),
            "is_regulated": bool(reg_by_site.get(site, False)),
            "n_paired_wy": int(ok.sum()),
            "median_log10_ratio": float(np.median(lr)),
            "median_ratio": float(10 ** np.median(lr)),
            "mad_log10_ratio": float(np.median(np.abs(lr - np.median(lr)))),
            "pearson_r_log": r_log, "spearman_r": r_sp,
            "usgs_max_cfs": float(uu[ok][iu]), "nwm_max_cfs": float(nn[ok][inn]),
            "usgs_max_wy": int(wy_ok[iu]), "nwm_max_wy": int(wy_ok[inn]),
            "max_wy_agree": bool(wy_ok[iu] == wy_ok[inn]),
            "max_ratio": float(nn[ok][inn] / uu[ok][iu]),
        })
    pd_paired = (pd.concat(paired, ignore_index=True) if paired
                 else pd.DataFrame())
    pd_stats = pd.DataFrame(stats_rows)
    log.info("Annual maxima: %d paired station-years across %d stations",
             len(pd_paired), len(pd_stats))
    return pd_paired, pd_stats


# ── Reproduction check against the stored 04b / 04c fits ─────────────────────

def reproduction_check(cmp_df: pd.DataFrame, bucket: str, prefix: str) -> None:
    for label, key, side in (("04b USGS", USGS_STORED, "usgs"),
                             ("04c NWM", NWM_STORED, "nwm")):
        try:
            st = b04b._read_parquet_s3(bucket, f"{prefix}{key}")
        except Exception as e:                              # noqa: BLE001
            log.warning("  %s: stored table unreadable (%s)", label, e)
            continue
        st["site_no"] = st["site_no"].astype(str)
        qc = [f"Q{rp}" for rp in RPS if f"Q{rp}" in st.columns]
        m = cmp_df.merge(st[["site_no", *qc]], on="site_no", how="inner",
                         suffixes=("", "_stored"))
        if m.empty:
            log.warning("  %s: no stations in common", label)
            continue
        dev = []
        for c in qc:
            a, s = m[f"{side}_{c}"], m[c]
            dev.append((a - s).abs() / s.replace(0, np.nan) * 100.0)
        worst = float(pd.concat(dev, axis=1).max(axis=1).max())
        (log.info if worst < 1.0 else log.warning)(
            "  %s: worst quantile deviation %.4f%% over %d station(s)%s",
            label, worst, len(m),
            "" if worst < 1.0 else "   <-- re-fit does NOT match production")


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_parameters(d: pd.DataFrame) -> plt.Figure:
    """One-to-one scatters: does the NWM fit land where the gauge fit does?"""
    # Record length is deliberately NOT a panel: the NWM side is a constant 43
    # water years, so a one-to-one scatter of it is a horizontal line carrying
    # no information.  It lives in the tables and the report instead.  PPCC
    # takes the slot — it asks whether LP3 describes each source equally well.
    panels = [("mean_log",       "mean log10 Q"),
              ("std_log",        "std log10 Q"),
              ("skew_at_site",   "at-site skew"),
              ("skew_weighted",  "B17C weighted skew"),
              ("weight_at_site", "weight on at-site skew"),
              ("ppcc",           "probability-plot correlation")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), facecolor=SURFACE)
    for ax, (key, label) in zip(axes.ravel(), panels):
        x, y = d[f"usgs_{key}"].astype(float), d[f"nwm_{key}"].astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        _style(ax)
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        lim = (lo - pad, hi + pad)
        # 1:1 reference first, so the marks sit on top of it.
        ax.plot(lim, lim, color=AXIS, linewidth=1.0, zorder=1)
        ax.scatter(x, y, s=26, color=SERIES_1, alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"USGS observed peaks — {label}")
        ax.set_ylabel(f"NWM retrospective — {label}")
        med = float(np.median(y - x))
        r = _safe_r(x.values, y.values)
        rtxt = "n/a" if not np.isfinite(r) else f"{r:.2f}"
        ax.text(0.03, 0.97,
                f"n = {len(x)}\nmedian NWM−USGS = {med:+.3f}\nr = {rtxt}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, linewidth=0.6,
                          boxstyle="round,pad=0.35"))
    fig.suptitle("LP3 fit parameters — NWM retrospective vs USGS observed peaks",
                 fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.005,
             "Points on the 1:1 line mean the two data sources imply the same fitted "
             "parameter. Scatter in skew is expected: at n≈50 the third moment is "
             "barely estimable.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.022, 1, 0.962))
    return fig


def fig_quantiles(d: pd.DataFrame) -> plt.Figure:
    # A dedicated narrow column for the colorbar: letting matplotlib steal space
    # from panel (b) pushed its bar onto panel (c)'s y-axis label.
    fig = plt.figure(figsize=(15, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 0.045, 1.1], wspace=0.34)
    ratios = np.array([d[f"ratio_Q{rp}"].astype(float).values for rp in RPS])
    RTICKS = [0.25, 0.5, 1, 2, 4]
    RLABS = ["0.25×", "0.5×", "1×", "2×", "4×"]

    # (a) ratio vs return period — median with an IQR band
    ax = fig.add_subplot(gs[0, 0]); _style(ax)
    med = np.nanmedian(ratios, axis=1)
    q1 = np.nanpercentile(ratios, 25, axis=1)
    q3 = np.nanpercentile(ratios, 75, axis=1)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.fill_between(RPS, q1, q3, color=SERIES_1, alpha=0.16, zorder=2,
                    label="interquartile range")
    ax.plot(RPS, med, color=SERIES_1, linewidth=2.0, zorder=3, label="median")
    ax.set_xscale("log"); ax.set_yscale("log")
    _log_ticks(ax, "x", RPS, [str(r) for r in RPS])
    _log_ticks(ax, "y", RTICKS, RLABS)
    ax.set_xlabel("return period (years)")
    ax.set_ylabel("NWM Q / USGS Q")
    ax.set_title("(a)  divergence grows with return period",
                 fontsize=10, color=INK, loc="left")
    ax.annotate(f"{med[-1]:.2f}×", xy=(RPS[-1], med[-1]), xytext=(-6, 8),
                textcoords="offset points", fontsize=9, color=INK_2, ha="right")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    # (b) Q100 one-to-one, hue = ratio on a diverging scale centred at 1
    ax = fig.add_subplot(gs[0, 1]); _style(ax)
    x = d["usgs_Q100"].astype(float).values
    y = d["nwm_Q100"].astype(float).values
    r = d["ratio_Q100"].astype(float).values
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lim = (np.nanmin([x[ok], y[ok]]) * 0.6, np.nanmax([x[ok], y[ok]]) * 1.7)
    ax.plot(lim, lim, color=AXIS, linewidth=1.0, zorder=1)
    rr = np.clip(r[ok], 0.25, 4.0)
    sc = ax.scatter(x[ok], y[ok], c=rr, cmap=RATIO_CMAP,
                    norm=TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=4.0),
                    s=30, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("USGS Q100 (cfs)"); ax.set_ylabel("NWM Q100 (cfs)")
    ax.set_title("(b)  Q100, station by station", fontsize=10, color=INK, loc="left")
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(sc, cax=cax, ticks=RTICKS)
    cb.ax.set_yticklabels(RLABS)
    # Caption above the bar, not rotated beside it — a rotated label here sits
    # right on panel (c)'s y-axis label.
    cb.ax.set_title("NWM / USGS", fontsize=8, color=INK_2, pad=6)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)

    # (c) ratio spread per return period
    ax = fig.add_subplot(gs[0, 3]); _style(ax, grid_axis="y")
    data = [ratios[i][np.isfinite(ratios[i])] for i in range(len(RPS))]
    bp = ax.boxplot(data, positions=range(len(RPS)), widths=0.6,
                    showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor=SERIES_1, alpha=0.20, edgecolor=SERIES_1, linewidth=1.2)
    for part in ("whiskers", "caps"):
        for a in bp[part]:
            a.set(color=SERIES_1, linewidth=1.0)
    for m in bp["medians"]:
        m.set(color=INK, linewidth=1.4)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(RPS))); ax.set_xticklabels([str(r) for r in RPS])
    _log_ticks(ax, "y", RTICKS, RLABS)
    ax.set_xlabel("return period (years)"); ax.set_ylabel("NWM Q / USGS Q")
    ax.set_title("(c)  spread across stations", fontsize=10, color=INK, loc="left")

    fig.suptitle("LP3 quantiles — NWM retrospective vs USGS observed peaks",
                 fontsize=13, color=INK, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.13)
    return fig


def fig_annual_maxima(paired: pd.DataFrame, st: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.26)

    # (a) every paired station-year
    ax = fig.add_subplot(gs[0, 0]); _style(ax)
    x = paired["usgs_peak_cfs"].values; y = paired["nwm_peak_cfs"].values
    lim = (max(np.nanmin([x, y]) * 0.6, 1e-1), np.nanmax([x, y]) * 1.7)
    ax.plot(lim, lim, color=AXIS, linewidth=1.0, zorder=2)
    ax.scatter(x, y, s=7, color=SERIES_1, alpha=0.22,
               linewidth=0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("USGS instantaneous peak (cfs)")
    ax.set_ylabel("NWM water-year maximum (cfs)")
    ax.set_title(f"(a)  {len(paired):,} paired station-years",
                 fontsize=10, color=INK, loc="left")
    mr = float(10 ** np.median(paired["log10_ratio"]))
    # Named precisely: this pools every station-year, whereas (b) is the median
    # OF the per-station medians, so the two numbers differ slightly.
    ax.text(0.03, 0.97, f"median over all station-years = {mr:.2f}×",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, linewidth=0.6,
                      boxstyle="round,pad=0.35"))

    # (b) per-station bias
    ax = fig.add_subplot(gs[0, 1]); _style(ax, grid_axis="y")
    v = st["median_ratio"].astype(float).values
    v = v[np.isfinite(v) & (v > 0)]
    ax.hist(np.log10(v), bins=28, color=SERIES_1, alpha=0.75,
            edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0.0, color=AXIS, linewidth=1.0)
    mlog = float(np.median(np.log10(v)))
    ax.axvline(mlog, color=SERIES_2, linewidth=1.6)
    # One series needs no legend box — direct-label the median line instead.
    ax.annotate(f"median {10 ** mlog:.2f}×", xy=(mlog, 0.97), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                color=SERIES_2, va="top", ha="left")
    ax.set_xticks([-1, -np.log10(2), 0, np.log10(2), 1])
    ax.set_xticklabels(["0.1×", "0.5×", "1×", "2×", "10×"])
    ax.set_xlabel("per-station median NWM/USGS")
    ax.set_ylabel("stations")
    ax.set_title("(b)  bias by station", fontsize=10, color=INK, loc="left")

    # (c) rank agreement, and whether the record-max year is the same
    ax = fig.add_subplot(gs[0, 2]); _style(ax, grid_axis="y")
    r = st["spearman_r"].astype(float).values
    r = r[np.isfinite(r)]
    ax.hist(r, bins=24, color=SERIES_3, alpha=0.8, edgecolor=SURFACE,
            linewidth=0.8)
    mr_ = float(np.median(r))
    ax.axvline(mr_, color=SERIES_2, linewidth=1.6)
    ax.annotate(f"median ρ = {mr_:.2f}", xy=(mr_, 0.72), xycoords=("data", "axes fraction"),
                xytext=(-5, 0), textcoords="offset points", fontsize=8.5,
                color=SERIES_2, va="top", ha="right")
    agree = float(st["max_wy_agree"].mean()) * 100.0
    ax.set_xlabel("Spearman ρ of paired annual maxima")
    ax.set_ylabel("stations")
    ax.set_title("(c)  do they rank floods alike?", fontsize=10, color=INK,
                 loc="left")
    ax.text(0.03, 0.97,
            f"record-max water year\nagrees at {agree:.0f}% of stations",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, linewidth=0.6,
                      boxstyle="round,pad=0.35"))

    fig.suptitle("Annual maxima — NWM retrospective vs USGS observed peaks, "
                 "paired by water year", fontsize=13, color=INK, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.13)
    return fig


# ── Report ───────────────────────────────────────────────────────────────────

def report(cmp_df: pd.DataFrame, paired: pd.DataFrame, st: pd.DataFrame) -> None:
    log.info("=" * 74)
    log.info("NWM RETROSPECTIVE vs USGS OBSERVED PEAKS")
    log.info("=" * 74)
    log.info("stations compared: %d   (04c previously compared 105)", len(cmp_df))
    # The two records do not cover the same period, which is itself a source of
    # disagreement independent of the models.
    log.info("record length: USGS median %.0f yr (p10 %.0f, p90 %.0f)   "
             "NWM median %.0f yr   overlapping water years: median %.0f",
             cmp_df["usgs_n"].median(), cmp_df["usgs_n"].quantile(.10),
             cmp_df["usgs_n"].quantile(.90), cmp_df["nwm_n"].median(),
             cmp_df["n_overlap_wy"].median())

    log.info("\nparameter differences (NWM − USGS):")
    for k, lab in (("mean_log", "mean log10 Q"), ("std_log", "std log10 Q"),
                   ("skew_at_site", "at-site skew"),
                   ("skew_weighted", "weighted skew"),
                   ("weight_at_site", "weight on at-site")):
        s = cmp_df[f"d_{k}"].astype(float).dropna()
        if len(s):
            log.info("  %-18s median %+7.3f   p10 %+7.3f   p90 %+7.3f",
                     lab, s.median(), s.quantile(.10), s.quantile(.90))

    log.info("\nquantile ratio NWM/USGS:")
    for rp in RPS:
        s = cmp_df[f"ratio_Q{rp}"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s):
            log.info("  Q%-4d median %5.2fx   p10 %5.2fx   p90 %5.2fx   "
                     "NWM higher at %3.0f%% of stations",
                     rp, s.median(), s.quantile(.10), s.quantile(.90),
                     100.0 * (s > 1).mean())

    if not st.empty:
        log.info("\nannual maxima, paired by water year:")
        log.info("  %d paired station-years at %d stations", len(paired), len(st))
        mr = st["median_ratio"].astype(float).dropna()
        log.info("  per-station median NWM/USGS : median %.2fx  p10 %.2fx  p90 %.2fx",
                 mr.median(), mr.quantile(.10), mr.quantile(.90))
        log.info("  NWM lower than the gauge at %.0f%% of stations",
                 100.0 * (mr < 1).mean())
        sp = st["spearman_r"].astype(float).dropna()
        log.info("  Spearman rho : median %.2f   p10 %.2f", sp.median(), sp.quantile(.10))
        log.info("  record-max water year agrees at %.0f%% of stations",
                 100.0 * st["max_wy_agree"].mean())
        log.info("  paired years per station: median %.0f   p10 %.0f",
                 st["n_paired_wy"].median(), st["n_paired_wy"].quantile(.10))
        # Require a usable overlap: a 5-year pairing produces a wild ratio that
        # says more about the sample than about the model.
        enough = st[st["n_paired_wy"] >= 15]
        worst = enough.nsmallest(6, "median_ratio")[
            ["site_no", "comid", "n_paired_wy", "median_ratio", "spearman_r"]]
        log.info("\n  most NWM-deficient stations (>= 15 paired years):")
        for r in worst.itertuples():
            log.info("    %s  COMID %-10d n=%-3d  median %.2fx  rho %.2f",
                     r.site_no, r.comid, r.n_paired_wy, r.median_ratio, r.spearman_r)


def _write(df: pd.DataFrame, bucket: str, prefix: str, name: str, dry: bool) -> None:
    if df.empty:
        log.warning("  %s is empty — not written", name)
        return
    if dry:
        log.info("  DRY RUN: would write %s (%d rows)", name, len(df))
        return
    write_parquet_to_s3(df, bucket, f"{prefix}{OUT_PREFIX}{name}.parquet")
    b04b._upload_csv(df, bucket, f"{prefix}{OUT_PREFIX}{name}.csv")
    log.info("  wrote %s%s.{parquet,csv}  (%d rows)", OUT_PREFIX, name, len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-regulated", action="store_true",
                    help="also fit Rule-B regulated stations for the parameter "
                         "and quantile tables (they are always in the annual-maxima one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report but write nothing to S3")
    ap.add_argument("--local-figs", metavar="DIR", default=None,
                    help="also save the figures to a local directory")
    args = ap.parse_args()

    cfg = _config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    cl = b04b._read_parquet_s3(bucket, f"{prefix}{COMID_KEY}")
    cl["site_no"] = cl["site_no"].astype(str)
    cl = cl.drop_duplicates("site_no")
    comid_by_site = {r.site_no: int(r.comid) for r in cl.itertuples()
                     if pd.notna(r.comid)}
    dist_by_site = {r.site_no: float(r.distance_km) for r in cl.itertuples()
                    if pd.notna(getattr(r, "distance_km", None))}

    inv = b04b._read_parquet_s3(bucket, f"{prefix}{INV_KEY}",
                                columns=["site_no", "dec_lat_va"])
    inv["site_no"] = inv["site_no"].astype(str)
    lat_map = inv.set_index("site_no")["dec_lat_va"].to_dict()

    usgs_ams, reg_by_site = usgs_annual_maxima(bucket, prefix)
    comid_by_site = {s: c for s, c in comid_by_site.items() if s in usgs_ams}
    nwm_ams = nwm_annual_maxima(bucket, prefix, comid_by_site)

    cmp_df = build_fit_comparison(usgs_ams, nwm_ams, reg_by_site, comid_by_site,
                                  dist_by_site, lat_map, args.include_regulated)
    if cmp_df.empty:
        raise SystemExit("No station could be fitted on both sides.")

    log.info("reproduction check against the stored fits:")
    reproduction_check(cmp_df, bucket, prefix)

    paired, st = build_ams_comparison(usgs_ams, nwm_ams, reg_by_site, comid_by_site)

    log.info("writing tables:")
    _write(cmp_df, bucket, prefix, "fit_comparison", args.dry_run)
    _write(paired, bucket, prefix, "annual_maxima_paired", args.dry_run)
    _write(st, bucket, prefix, "annual_maxima_station_stats", args.dry_run)

    log.info("building figures:")
    figs = {"fig_parameters": fig_parameters(cmp_df),
            "fig_quantiles": fig_quantiles(cmp_df)}
    if not paired.empty:
        figs["fig_annual_maxima"] = fig_annual_maxima(paired, st)
    for name, fig in figs.items():
        if args.local_figs:
            Path(args.local_figs).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(args.local_figs) / f"{name}.png", dpi=150,
                        facecolor=SURFACE)
            log.info("  local %s.png", name)
        if not args.dry_run:
            b04b._upload_png(fig, bucket, f"{prefix}{OUT_PREFIX}{name}.png")
            log.info("  wrote %s%s.png", OUT_PREFIX, name)
        plt.close(fig)

    report(cmp_df, paired, st)


if __name__ == "__main__":
    main()
