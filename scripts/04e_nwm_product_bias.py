"""04e — bias of the operational NWM products (A&A, Open-Loop) against the
Retrospective v3.0, overall and on annual maxima.

Why this exists
───────────────
Every flow threshold in the monitor comes from an LP3 fit to NWM Retrospective
v3.0 (04c for gauges, p03 for the fleet), but the monitor compares live
*operational* flow against them. If the operational products do not sit on the
same distribution as the retrospective, a perfectly good threshold still
misfires — and that is a different defect from the gauge bias 04d measured.

COMID 13437963 is the standing case: 12 bridges have been above their Q100 gate
since 2026-08-18 while operational reads ~2,557 cfs, which is 4.6x the LARGEST
annual maximum in that reach's entire 43-year retrospective record. The LP3 fit
there is provably healthy (p08: reproduces to 0.0000%, every detector OK), so
the suspicion falls on a retro-vs-operational mismatch.

What it measures
────────────────
Two kinds of bias, for each product pair:

  OVERALL      every hour both products cover, paired on (comid, timestamp),
               reported BY FLOW PERCENTILE BAND. A single pooled number would
               be dominated by low flow; the thresholds are about the top of
               the distribution, so the bias has to be resolved by magnitude.
  ANNUAL MAX   water-year maxima from each product, paired on (comid, water
               year), using 04c's own annual_max_series so the completeness
               rule matches how the thresholds were built.

Pairs, and the overlap each one actually gets:
  aa_vs_retro   2018-09 .. 2023-02   ~4.4 yr,  4 complete WY,  105 comids
  ol_vs_retro   2021-04 .. 2023-02   ~1.8 yr,  1 complete WY,  105 comids
  aa_vs_ol      2021-04 .. 2026-05   ~5.1 yr,  4 complete WY,  264 comids

READ THE ol_vs_retro ANNUAL-MAXIMUM RESULT AS INDICATIVE ONLY. Open-Loop's
archive starts 2021-04-20 and the retrospective ends 2023-02-01, so exactly one
complete water year is shared. The hourly comparison over that window is still
worth having; a one-year annual-maximum "bias" is not a bias estimate.

A note on sample size: hourly streamflow is strongly autocorrelated, so a pair
with ten million matched hours does not carry ten million independent
observations. Everything here is reported as a distribution (medians, spreads,
per-comid summaries) and no p-value is computed, deliberately.

Outputs (nothing existing is overwritten; all under analysis/nwm_product_bias/)
    hourly_bias_by_comid.parquet / .csv      one row per (pair, comid, band)
    annual_max_bias.parquet / .csv           one row per (pair, comid, water year)
    annual_max_bias_by_comid.parquet / .csv  one row per (pair, comid)
    fig_overall_bias.png   fig_annual_max_bias.png

Usage
─────
    python scripts/04e_nwm_product_bias.py              # from the repo root
    python scripts/04e_nwm_product_bias.py --dry-run --local-figs /tmp/figs
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
log = logging.getLogger("04e_nwm_product_bias")

AA_KEY    = "nwm/analysis_assim.parquet"
OL_KEY    = "nwm/open_loop.parquet"
RETRO_KEY = "nwm/retrospective_full.parquet"
OUT_PREFIX = "analysis/nwm_product_bias/"

CFS = 35.3146667

# Bias is resolved by where the hour sits in the REFERENCE product's own flow
# distribution at that reach, so every comid contributes to every band.
BANDS = [(0, 50, "p0-50"), (50, 90, "p50-90"), (90, 99, "p90-99"),
         (99, 99.9, "p99-99.9"), (99.9, 100, "p99.9-100")]

PAIRS = [("aa_vs_retro", "retro", "aa", "A&A vs Retrospective"),
         ("ol_vs_retro", "retro", "ol", "Open-Loop vs Retrospective"),
         ("aa_vs_ol",    "ol",    "aa", "A&A vs Open-Loop")]

# ── Palette (dataviz reference instance, light surface) ──────────────────────
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {"aa_vs_retro": "#2a78d6", "ol_vs_retro": "#eb6834", "aa_vs_ol": "#1baf7a"}


def _style(ax, *, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2); ax.yaxis.label.set_color(INK_2)


def _log_ticks(ax, which, ticks, labels):
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FixedFormatter(labels))
    axis.set_minor_locator(mticker.NullLocator())
    axis.set_minor_formatter(mticker.NullFormatter())


# ── Loading ──────────────────────────────────────────────────────────────────

def load_product(bucket: str, prefix: str, key: str, label: str) -> pd.DataFrame:
    """comid, datetime_utc, q_cfs — deduplicated to one series per comid.

    Several site_no can share a reach (264 comids over 273 sites); the routed
    series is identical, so keeping both would double-weight those reaches.
    """
    d = b04b._read_parquet_s3(bucket, f"{prefix}{key}",
                              columns=["comid", "datetime_utc", "streamflow_cms"])
    d["comid"] = pd.to_numeric(d["comid"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["comid"])
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"], utc=True)
    d["q_cfs"] = d["streamflow_cms"].astype(float) * CFS
    d = (d.drop(columns=["streamflow_cms"])
           .drop_duplicates(["comid", "datetime_utc"]))
    log.info("  %-6s %10d rows  %4d comids  %s .. %s", label, len(d),
             d["comid"].nunique(), d["datetime_utc"].min().date(),
             d["datetime_utc"].max().date())
    return d


# ── Overall (hourly) bias ────────────────────────────────────────────────────

def hourly_bias(ref: pd.DataFrame, test: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Per (comid, flow band) ratio of test to reference over matched hours."""
    m = ref.merge(test, on=["comid", "datetime_utc"], suffixes=("_ref", "_test"))
    m = m[(m["q_cfs_ref"] > 0) & (m["q_cfs_test"] > 0)]
    if m.empty:
        return pd.DataFrame()
    rows = []
    for comid, g in m.groupby("comid"):
        ref_q = g["q_cfs_ref"].to_numpy()
        ratio = g["q_cfs_test"].to_numpy() / ref_q
        # Percentile edges of the REFERENCE flow at this reach.
        edges = {lo: np.percentile(ref_q, lo) for lo, _, _ in BANDS}
        edges[100] = ref_q.max()
        for lo, hi, name in BANDS:
            sel = (ref_q >= np.percentile(ref_q, lo)) & (ref_q <= np.percentile(ref_q, hi))
            if sel.sum() < 10:
                continue
            rows.append({
                "pair": pair, "comid": int(comid), "band": name,
                "n_hours": int(sel.sum()),
                "ref_q_lo_cfs": float(np.percentile(ref_q, lo)),
                "ref_q_hi_cfs": float(np.percentile(ref_q, hi)),
                "median_ratio": float(np.median(ratio[sel])),
                "mean_log10_ratio": float(np.mean(np.log10(ratio[sel]))),
                "p10_ratio": float(np.percentile(ratio[sel], 10)),
                "p90_ratio": float(np.percentile(ratio[sel], 90)),
            })
    out = pd.DataFrame(rows)
    log.info("  %s: %d matched hours over %d comids -> %d band rows",
             pair, len(m), m["comid"].nunique(), len(out))
    return out


# ── Annual-maximum bias ──────────────────────────────────────────────────────

def annual_maxima(d: pd.DataFrame) -> pd.DataFrame:
    """Water-year maxima per comid via 04c's completeness rule."""
    rows = []
    for comid, g in d.groupby("comid"):
        sub = g[["datetime_utc"]].copy()
        sub["streamflow_cms"] = g["q_cfs"].to_numpy() / CFS
        ann = m04c.annual_max_series(sub)
        for wy, q in ann.items():
            rows.append({"comid": int(comid), "water_year": int(wy),
                         "max_cfs": float(q)})
    return pd.DataFrame(rows)


def annual_max_bias(ref: pd.DataFrame, test: pd.DataFrame,
                    pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = annual_maxima(ref), annual_maxima(test)
    if a.empty or b.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = a.merge(b, on=["comid", "water_year"], suffixes=("_ref", "_test"))
    m = m[(m["max_cfs_ref"] > 0) & (m["max_cfs_test"] > 0)].copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    m["pair"] = pair
    m["ratio"] = m["max_cfs_test"] / m["max_cfs_ref"]
    m["log10_ratio"] = np.log10(m["ratio"])
    per = (m.groupby("comid")
             .agg(n_wy=("water_year", "size"),
                  median_ratio=("ratio", "median"),
                  min_ratio=("ratio", "min"), max_ratio=("ratio", "max"))
             .reset_index())
    per["pair"] = pair
    log.info("  %s: %d paired comid-water-years over %d comids (%d water years)",
             pair, len(m), m["comid"].nunique(), m["water_year"].nunique())
    return m, per


# ── Figures ──────────────────────────────────────────────────────────────────

RT = [0.25, 0.5, 1, 2, 4]
RL = ["0.25×", "0.5×", "1×", "2×", "4×"]


def fig_overall(hb: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.24})
    names = [b[2] for b in BANDS]

    ax = axes[0]; _style(ax)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    for pair, _, _, label in PAIRS:
        sub = hb[hb["pair"] == pair]
        if sub.empty:
            continue
        med = [sub.loc[sub["band"] == n, "median_ratio"].median() for n in names]
        q1 = [sub.loc[sub["band"] == n, "median_ratio"].quantile(.25) for n in names]
        q3 = [sub.loc[sub["band"] == n, "median_ratio"].quantile(.75) for n in names]
        x = np.arange(len(names))
        ax.fill_between(x, q1, q3, color=SERIES[pair], alpha=0.13, zorder=2)
        ax.plot(x, med, color=SERIES[pair], linewidth=2.0, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                zorder=3, label=label)
    ax.set_yscale("log"); _log_ticks(ax, "y", RT, RL)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_xlabel("where the hour sits in the reference product's flow distribution")
    ax.set_ylabel("test / reference")
    ax.set_title("(a)  bias by flow magnitude — bands, not a single number",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    ax = axes[1]; _style(ax, grid_axis="y")
    top = hb[hb["band"] == "p99.9-100"]
    data, labels, colors = [], [], []
    for pair, _, _, label in PAIRS:
        v = top.loc[top["pair"] == pair, "median_ratio"].dropna()
        if len(v):
            data.append(v.values); labels.append(label.replace(" vs ", "\nvs ")); colors.append(SERIES[pair])
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.20, edgecolor=c, linewidth=1.2)
    for part in ("whiskers", "caps"):
        for a, c in zip(bp[part], np.repeat(colors, 2)):
            a.set(color=c, linewidth=1.0)
    for m_ in bp["medians"]:
        m_.set(color=INK, linewidth=1.4)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yscale("log"); _log_ticks(ax, "y", RT, RL)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("test / reference")
    ax.set_title("(b)  the top 0.1% of flow — what the thresholds see",
                 fontsize=10, color=INK, loc="left")

    fig.suptitle("NWM operational vs retrospective — hourly flow bias",
                 fontsize=13, color=INK, y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.14)
    return fig


def fig_annual(am: pd.DataFrame, per: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.27})

    ax = axes[0]; _style(ax)
    for pair, _, _, label in PAIRS:
        s = am[am["pair"] == pair]
        if s.empty:
            continue
        ax.scatter(s["max_cfs_ref"], s["max_cfs_test"], s=16, alpha=0.5,
                   color=SERIES[pair], edgecolor=SURFACE, linewidth=0.5,
                   zorder=3, label=label)
    if not am.empty:
        lim = (am[["max_cfs_ref", "max_cfs_test"]].to_numpy().min() * 0.6,
               am[["max_cfs_ref", "max_cfs_test"]].to_numpy().max() * 1.7)
        ax.plot(lim, lim, color=AXIS, linewidth=1.0, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reference annual maximum (cfs)")
    ax.set_ylabel("test annual maximum (cfs)")
    ax.set_title("(a)  paired water-year maxima", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    ax = axes[1]; _style(ax, grid_axis="y")
    data, labels, colors = [], [], []
    for pair, _, _, label in PAIRS:
        v = per.loc[per["pair"] == pair, "median_ratio"].dropna()
        if len(v):
            data.append(np.log10(v.values))
            labels.append(label.replace(" vs ", "\nvs ")); colors.append(SERIES[pair])
    if data:
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
        for box, c in zip(bp["boxes"], colors):
            box.set(facecolor=c, alpha=0.20, edgecolor=c, linewidth=1.2)
        for part in ("whiskers", "caps"):
            for a, c in zip(bp[part], np.repeat(colors, 2)):
                a.set(color=c, linewidth=1.0)
        for m_ in bp["medians"]:
            m_.set(color=INK, linewidth=1.4)
        ax.set_xticklabels(labels, fontsize=8.5)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    ax.set_ylabel("per-reach median ratio")
    ax.set_title("(b)  annual-maximum bias by reach", fontsize=10, color=INK, loc="left")

    ax = axes[2]; _style(ax)
    for pair, _, _, label in PAIRS:
        s = am[am["pair"] == pair]
        if s.empty:
            continue
        g = s.groupby("water_year")["ratio"].median()
        ax.plot(g.index, g.values, color=SERIES[pair], linewidth=1.8, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=label, zorder=3)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yscale("log"); _log_ticks(ax, "y", RT, RL)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("water year"); ax.set_ylabel("median test / reference")
    ax.set_title("(c)  does the bias drift?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    fig.suptitle("NWM operational vs retrospective — annual-maximum bias",
                 fontsize=13, color=INK, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.14)
    return fig


# ── Report ───────────────────────────────────────────────────────────────────

def report(hb: pd.DataFrame, am: pd.DataFrame, per: pd.DataFrame) -> None:
    log.info("=" * 76)
    log.info("NWM OPERATIONAL vs RETROSPECTIVE — BIAS")
    log.info("=" * 76)
    log.info("Hourly flow is autocorrelated, so these are distributions over reaches,")
    log.info("not independent samples. No p-values, by design.")

    for pair, ref, test, label in PAIRS:
        sub = hb[hb["pair"] == pair]
        if sub.empty:
            continue
        log.info("\n%s  (%d reaches)", label, sub["comid"].nunique())
        for _, _, name in BANDS:
            v = sub.loc[sub["band"] == name, "median_ratio"].dropna()
            if len(v):
                log.info("   %-11s median %5.2fx   p25 %5.2fx   p75 %5.2fx   "
                         "above 1x at %3.0f%% of reaches",
                         name, v.median(), v.quantile(.25), v.quantile(.75),
                         100.0 * (v > 1).mean())

    log.info("\n" + "-" * 76)
    log.info("ANNUAL MAXIMA")
    for pair, ref, test, label in PAIRS:
        s = am[am["pair"] == pair]
        p = per[per["pair"] == pair]
        if s.empty:
            continue
        nwy = s["water_year"].nunique()
        warn = "   <-- ONE water year: indicative only, not a bias estimate" if nwy < 2 else ""
        log.info("\n%s: %d pairs, %d reaches, %d water year(s)%s",
                 label, len(s), s["comid"].nunique(), nwy, warn)
        v = p["median_ratio"].dropna()
        log.info("   per-reach median ratio: median %.2fx   p10 %.2fx   p90 %.2fx",
                 v.median(), v.quantile(.10), v.quantile(.90))
        log.info("   test exceeds reference at %.0f%% of reaches", 100.0 * (v > 1).mean())
        big = p.nlargest(5, "median_ratio")[["comid", "n_wy", "median_ratio"]]
        log.info("   most operational-hot reaches:")
        for r in big.itertuples():
            log.info("     COMID %-10d n_wy=%-2d  %.2fx", r.comid, r.n_wy, r.median_ratio)

    # The standing case this was built to explain.
    for pair in ("aa_vs_retro", "ol_vs_retro"):
        row = per[(per["pair"] == pair) & (per["comid"] == 13437963)]
        if len(row):
            log.info("\nCOMID 13437963 (%s): median %.2fx over %d water year(s)",
                     pair, float(row["median_ratio"].iloc[0]), int(row["n_wy"].iloc[0]))


def _write(df, bucket, prefix, name, dry):
    if df.empty:
        log.warning("  %s is empty — not written", name); return
    if dry:
        log.info("  DRY RUN: would write %s (%d rows)", name, len(df)); return
    write_parquet_to_s3(df, bucket, f"{prefix}{OUT_PREFIX}{name}.parquet")
    b04b._upload_csv(df, bucket, f"{prefix}{OUT_PREFIX}{name}.csv")
    log.info("  wrote %s%s.{parquet,csv}  (%d rows)", OUT_PREFIX, name, len(df))


def _config() -> dict:
    local = Path("config.yaml")
    if local.exists():
        return load_config()
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="write nothing to S3")
    ap.add_argument("--local-figs", metavar="DIR", default=None)
    args = ap.parse_args()

    cfg = _config()
    bucket = cfg["aws"]["output_bucket"]; prefix = cfg["aws"]["output_prefix"]

    log.info("loading products:")
    prod = {"aa": load_product(bucket, prefix, AA_KEY, "A&A"),
            "ol": load_product(bucket, prefix, OL_KEY, "OL"),
            "retro": load_product(bucket, prefix, RETRO_KEY, "retro")}

    hb_parts, am_parts, per_parts = [], [], []
    for pair, ref_k, test_k, label in PAIRS:
        ref, test = prod[ref_k], prod[test_k]
        comids = sorted(set(ref["comid"]) & set(test["comid"]))
        t0 = max(ref["datetime_utc"].min(), test["datetime_utc"].min())
        t1 = min(ref["datetime_utc"].max(), test["datetime_utc"].max())
        log.info("\n%s: %d shared comids, overlap %s .. %s (%.1f yr)",
                 label, len(comids), t0.date(), t1.date(),
                 (t1 - t0).days / 365.25)
        r = ref[ref["comid"].isin(comids) & ref["datetime_utc"].between(t0, t1)]
        t = test[test["comid"].isin(comids) & test["datetime_utc"].between(t0, t1)]
        hb_parts.append(hourly_bias(r, t, pair))
        a, p = annual_max_bias(r, t, pair)
        am_parts.append(a); per_parts.append(p)

    hb = pd.concat([x for x in hb_parts if not x.empty], ignore_index=True)
    am = pd.concat([x for x in am_parts if not x.empty], ignore_index=True)
    per = pd.concat([x for x in per_parts if not x.empty], ignore_index=True)

    log.info("\nwriting tables:")
    _write(hb, bucket, prefix, "hourly_bias_by_comid", args.dry_run)
    _write(am, bucket, prefix, "annual_max_bias", args.dry_run)
    _write(per, bucket, prefix, "annual_max_bias_by_comid", args.dry_run)

    log.info("building figures:")
    figs = {"fig_overall_bias": fig_overall(hb), "fig_annual_max_bias": fig_annual(am, per)}
    for name, fig in figs.items():
        if args.local_figs:
            Path(args.local_figs).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(args.local_figs) / f"{name}.png", dpi=150, facecolor=SURFACE)
            log.info("  local %s.png", name)
        if not args.dry_run:
            b04b._upload_png(fig, bucket, f"{prefix}{OUT_PREFIX}{name}.png")
            log.info("  wrote %s%s.png", OUT_PREFIX, name)
        plt.close(fig)

    report(hb, am, per)


if __name__ == "__main__":
    main()
