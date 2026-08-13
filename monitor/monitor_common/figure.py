"""Alert PDF builder.

One page, four panels + header, for a bridge whose trigger fired:

    [ MRMS 1-h QPE map, bridge-centered ]  [ 24-h Tc-accumulation precip + P10/50/100 ]
    [ NWM reach streamflow map          ]  [ 24-h streamflow + velocity + Q10/50/100  ]

Reuses the ARI-snapping helpers from scripts/visualize_bridge_event.py.
Renders to PDF bytes (matplotlib's built-in pdf backend — no extra deps).
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from . import config  # noqa: E402

PRECIP_RPS = config.SEVERITY_RPS          # [10, 50, 100]
FLOW_RPS = config.SEVERITY_RPS
TIER_C = {10: "#f0ad4e", 50: "#d9534f", 100: "#7b1fa2"}   # amber / red / purple
BRIDGE_C = "#111111"

# Digest-map palette — slots 1-3 of the validated categorical set (these three
# clear the all-pairs CVD gate, which is the one that applies to point maps).
C_CONF, C_OPEN, C_PRECIP = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def ari_from_value(value, rps, vals):
    """Return period of `value` by log-log interpolation of the (RP -> value) curve."""
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order]
    r = np.asarray(rps, float)[order]
    good = np.isfinite(v) & (v > 0)
    v, r = v[good], r[good]
    if v.size < 2 or not np.isfinite(value) or value <= 0:
        return float("nan")
    return 10.0 ** float(np.interp(np.log10(value), np.log10(v), np.log10(r)))


def _fired_label(fired: list[dict]) -> str:
    parts = []
    for f in sorted(fired, key=lambda d: -d["severity_rp"]):
        parts.append(f"{f['type'].upper()} ≥ {'P' if f['type']=='precip' else 'Q'}{f['severity_rp']}")
    return "  •  ".join(parts) if parts else "—"


def _threshold_lines(ax, thresholds: dict[int, float], unit: str, observed: float | None):
    """Horizontal P/Q tier lines; bold the highest one exceeded."""
    exceeded = [rp for rp, thr in thresholds.items()
                if np.isfinite(thr) and observed is not None and observed >= thr]
    top = max(exceeded) if exceeded else None
    for rp in sorted(thresholds):
        thr = thresholds[rp]
        if not np.isfinite(thr):
            continue
        lw = 2.6 if rp == top else 1.4
        ax.axhline(thr, color=TIER_C[rp], ls="--", lw=lw,
                   label=f"{unit}{rp} = {thr:,.2f}" if unit == "P" else f"{unit}{rp} = {thr:,.0f} cfs")


# ── Maps ─────────────────────────────────────────────────────────────────────

def _mrms_map(ax, lat, lon, grid, half_deg=0.35):
    ax.set_title("MRMS 1-h QPE (in) — bridge-centered", fontsize=10)
    if grid is not None:
        arr, glat_desc, glon_asc = grid
        latmask = (glat_desc <= lat + half_deg) & (glat_desc >= lat - half_deg)
        lonmask = (glon_asc <= lon + half_deg) & (glon_asc >= lon - half_deg)
        if latmask.any() and lonmask.any():
            sub = arr[np.ix_(latmask, lonmask)]
            sub = np.where(np.isfinite(sub), sub, 0.0)
            La, Lo = glat_desc[latmask], glon_asc[lonmask]
            pm = ax.pcolormesh(Lo, La, sub, cmap="YlGnBu",
                               vmin=0, vmax=max(0.2, float(np.nanmax(sub))), shading="nearest")
            ax.figure.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label="in/hr")
    ax.plot([lon], [lat], marker="^", ms=13, color=BRIDGE_C, mec="white", mew=1.2, zorder=6)
    ax.set_xlim(lon - half_deg, lon + half_deg)
    ax.set_ylim(lat - half_deg, lat + half_deg)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1.0 / np.cos(np.radians(lat)))


def _reach_map(ax, lat, lon, comid, reach, q_cfs, q_thresholds, half_deg=0.35):
    ax.set_title(f"NWM reach COMID {comid}" + (f" — Q={q_cfs:,.0f} cfs" if np.isfinite(q_cfs) else ""),
                 fontsize=10)
    qmax = max([v for v in q_thresholds.values() if np.isfinite(v)] + [q_cfs if np.isfinite(q_cfs) else 0]) or 1.0
    frac = float(np.clip((q_cfs / qmax) if np.isfinite(q_cfs) else 0.0, 0.0, 1.0))
    col = plt.get_cmap("plasma")(frac)
    if reach:
        xs = [c[0] for c in reach]
        ys = [c[1] for c in reach]
        ax.plot(xs, ys, color=col, lw=3.5, solid_capstyle="round", zorder=4,
                label=f"reach ({frac*100:.0f}% of Q100)")
    ax.plot([lon], [lat], marker="^", ms=13, color=BRIDGE_C, mec="white", mew=1.2, zorder=6)
    ax.set_xlim(lon - half_deg, lon + half_deg)
    ax.set_ylim(lat - half_deg, lat + half_deg)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1.0 / np.cos(np.radians(lat)))
    if reach:
        ax.legend(loc="upper right", fontsize=7)


# ── Time series ──────────────────────────────────────────────────────────────

def _precip_ts(ax, hours, precip_1h, tc_accum, tc_dur, p_thr):
    t = [h.to_pydatetime() for h in hours]
    width = (mdates.date2num(t[1]) - mdates.date2num(t[0])) * 0.9 if len(t) > 1 else 0.03
    ax.bar(t, np.nan_to_num(precip_1h), width=width, color="#9ecae1", label="1-h QPE")
    ax.plot(t, tc_accum, color="#08519c", lw=2.0, label=f"{tc_dur}-h accum (Tc)")
    peak = float(np.nanmax(tc_accum)) if np.isfinite(tc_accum).any() else 0.0
    _threshold_lines(ax, p_thr, "P", peak)
    ax.set_ylabel("Precip (in)", fontsize=9)
    ax.set_title("Past 24 h precipitation — MRMS", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)


def _flow_ts(ax, hours, q_aa, q_ol, v_aa, v_ol, q_thr):
    t = [h.to_pydatetime() for h in hours]
    ax.plot(t, q_ol, color="#0f7d7d", lw=2.2, label="NWM open-loop (trigger)")
    if np.isfinite(q_aa).any():
        ax.plot(t, q_aa, color="#54278f", lw=1.6, ls="-.", label="NWM A&A (with DA)")
    peak = float(np.nanmax(q_ol)) if np.isfinite(q_ol).any() else 0.0
    _threshold_lines(ax, q_thr, "Q", peak)
    ax.set_ylabel("Streamflow (cfs)", fontsize=9)
    ax.set_title("Past 24 h streamflow + velocity — NWM", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    axv = ax.twinx()
    vel = v_ol if np.isfinite(v_ol).any() else v_aa
    axv.plot(t, vel, color="#d95f02", lw=1.3, alpha=0.8)
    axv.set_ylabel("Velocity (m/s)", color="#d95f02", fontsize=9)
    axv.tick_params(axis="y", labelcolor="#d95f02", labelsize=7)


def build_pdf(bridge: dict, valid_hour: pd.Timestamp, fired: list[dict],
              hours, precip_1h, tc_accum, q_aa_cfs, q_ol_cfs, v_aa, v_ol,
              mrms_grid, reach) -> bytes:
    lat, lon = bridge["lat"], bridge["lon"]
    p_thr = {rp: bridge.get(f"P{rp}", np.nan) for rp in PRECIP_RPS}
    q_thr = {rp: bridge.get(f"Q{rp}_cfs", np.nan) for rp in FLOW_RPS}
    q_now = float(q_ol_cfs[-1]) if len(q_ol_cfs) and np.isfinite(q_ol_cfs[-1]) else float("nan")

    fig = plt.figure(figsize=(11.0, 8.5))
    gs = GridSpec(3, 2, height_ratios=[0.5, 1.0, 1.0], hspace=0.42, wspace=0.22,
                  left=0.07, right=0.95, top=0.95, bottom=0.07)

    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    sc = "SCOUR-CRITICAL" if bridge.get("scour") else "over-water"
    axh.text(0.0, 0.85, f"BRIDGE FLOOD ALERT — {bridge['asset']}", fontsize=16,
             fontweight="bold", va="top")
    axh.text(0.0, 0.42, f"{sc}   •   lat {lat:.5f}, lon {lon:.5f}   •   "
             f"COMID {bridge.get('comid','—')}   •   valid {valid_hour:%Y-%m-%d %H:%MZ}",
             fontsize=10, va="top", color="0.25")
    axh.text(0.0, 0.05, "TRIGGERED:  " + _fired_label(fired), fontsize=12,
             fontweight="bold", color="#b30000", va="top")

    _mrms_map(fig.add_subplot(gs[1, 0]), lat, lon, mrms_grid)
    _precip_ts(fig.add_subplot(gs[1, 1]), hours, precip_1h, tc_accum,
               bridge.get("tc_dur_hr", 1), p_thr)
    _reach_map(fig.add_subplot(gs[2, 0]), lat, lon, bridge.get("comid", "?"),
               reach, q_now, q_thr)
    axf = fig.add_subplot(gs[2, 1])
    _flow_ts(axf, hours, q_aa_cfs, q_ol_cfs, v_aa, v_ol, q_thr)
    axf.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    axf.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    fig.text(0.07, 0.015,
             "Precip trigger: trailing round(Tc)-h MRMS ≥ Atlas-14 P.  "
             "Flow trigger: NWM open-loop ≥ retro-LP3 Q (04c).  "
             "Alerts de-duplicated on a 24-h event separation.",
             fontsize=7, color="0.4")

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


# ── Digest (one PDF for the whole run) ───────────────────────────────────────

ROWS_PER_PAGE = 34


def _draw_counties(ax, counties: pd.DataFrame | None) -> None:
    """County outlines from the flattened p07 table (no geopandas in Lambda)."""
    if counties is None or counties.empty:
        return
    for _, ring in counties.groupby("part_id"):
        ax.fill(ring["lon"].to_numpy(), ring["lat"].to_numpy(),
                facecolor="#f4f3ef", edgecolor="#e1e0d9", linewidth=0.5, zorder=1)


def _digest_map(ax, ev: pd.DataFrame, cfg: pd.DataFrame, counties) -> None:
    _draw_counties(ax, counties)
    if cfg is not None and len(cfg):
        ax.scatter(cfg["lon"], cfg["lat"], s=1.2, c=MUTED, alpha=0.28,
                   linewidths=0, zorder=2)
    groups = (("flow_conf", C_CONF, "o", 34), ("flow_open", C_OPEN, "o", 34),
              ("precip", C_PRECIP, "^", 46))
    for cls, col, mark, size in groups:
        d = ev[ev["map_class"] == cls]
        if len(d):
            ax.scatter(d["lon"], d["lat"], s=size, c=col, marker=mark,
                       edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_aspect(1.0 / np.cos(np.radians(39.8)))
    if counties is not None and not counties.empty:      # tight to the state
        ax.set_xlim(counties["lon"].min() - 0.08, counties["lon"].max() + 0.08)
        ax.set_ylim(counties["lat"].min() - 0.08, counties["lat"].max() + 0.08)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _digest_cover(pdf, ev: pd.DataFrame, cfg, counties, mrms_hour, nwm_hour) -> None:
    fig = plt.figure(figsize=(11.0, 8.5), facecolor="white")
    gs = GridSpec(1, 2, width_ratios=[1.05, 1], left=0.04, right=0.97,
                  top=0.80, bottom=0.05, wspace=0.06)

    n_flow = int((ev["trigger_type"] == "flow").sum())
    n_prec = int((ev["trigger_type"] == "precip").sum())
    n_conf = int((ev["map_class"] == "flow_conf").sum())
    n_open = int((ev["map_class"] == "flow_open").sum())
    n_scour = int(ev["scour"].astype(bool).sum())
    n_reach = int(ev.loc[ev["trigger_type"] == "flow", "comid"].nunique())
    hour = nwm_hour or mrms_hour

    fig.text(0.04, 0.965, "INDOT BRIDGE FLOOD ALERT — DIGEST", fontsize=19,
             fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.922,
             f"{ev['bridge_id'].nunique()} bridges triggered   ·   valid "
             f"{hour:%Y-%m-%d %H:%M} UTC   ·   MRMS {mrms_hour:%H%M}Z / NWM {nwm_hour:%H%M}Z"
             if (mrms_hour is not None and nwm_hour is not None) else
             f"{ev['bridge_id'].nunique()} bridges triggered",
             fontsize=11.5, color=INK2, va="top")

    bits = [f"{n_flow} streamflow", f"{n_prec} precipitation",
            f"{n_scour} scour-critical", f"{n_reach} distinct reaches"]
    fig.text(0.04, 0.884, "   ·   ".join(bits), fontsize=11, color=INK2, va="top")
    sev = ev.groupby("severity_rp").size().to_dict()
    fig.text(0.04, 0.850, "Severity:   " + "   ".join(
        f"{sev.get(rp, 0)} × {rp}-yr" for rp in config.SEVERITY_RPS),
        fontsize=11, color=INK2, va="top")

    ax = fig.add_subplot(gs[0])
    _digest_map(ax, ev, cfg, counties)

    # Fixed 0-1 coordinate frame: markers and text must share one transform or
    # autoscaling on the marker points throws the labels off-panel.
    axl = fig.add_subplot(gs[1]); axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.set_autoscale_on(False)
    y = 0.97
    axl.text(0, y, "What confirms each alert", fontsize=13, fontweight="bold",
             va="top", color=INK)
    y -= 0.075
    for col, mark, lbl, n in (
            (C_CONF, "o", "Streamflow — NWM A&A corroborates", n_conf),
            (C_OPEN, "o", "Streamflow — open-loop only, A&A disagrees", n_open),
            (C_PRECIP, "^", "Precipitation — MRMS ≥ Atlas-14", n_prec)):
        axl.plot([0.025], [y - 0.014], marker=mark, ms=9, color=col,
                 mec="white", mew=1.0, clip_on=False)
        axl.text(0.075, y, f"{lbl}   ({n})", fontsize=10, va="top", color=INK2)
        y -= 0.062
    y -= 0.02
    axl.text(0, y, "The open-loop product drives the trigger because it is\n"
                   "gauge-free. Where A&A disagrees, treat the alert as\n"
                   "unconfirmed — data assimilation did not reproduce it.",
             fontsize=9.5, va="top", color=MUTED)
    y -= 0.16
    axl.text(0, y, "Top 12 by exceedance", fontsize=12, fontweight="bold",
             va="top", color=INK)
    y -= 0.055
    top = ev.assign(r=ev["observed"] / ev["threshold"]).nlargest(12, "r")
    for _, r in top.iterrows():
        axl.text(0.02, y, f"{str(r['asset'])[:24]:<24}", fontsize=8.6, va="top",
                 color=INK, family="monospace")
        axl.text(0.55, y, f"{r['trigger_type'][:4].upper()}  {r['severity_rp']:>3}-yr"
                          f"   {r['observed'] / r['threshold']:.2f}×",
                 fontsize=8.6, va="top", color=INK2, family="monospace")
        y -= 0.038

    fig.text(0.04, 0.018,
             "Precip: trailing round(Tc)-h MRMS ≥ Atlas-14 P.   "
             "Flow: NWM open-loop ≥ retro-LP3 Q (04c).   "
             "Scour-critical fire at ≥10-yr; all others at ≥50-yr.   "
             "24-h event separation applied.",
             fontsize=7.5, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _digest_table(pdf, ev: pd.DataFrame) -> None:
    cols = [("Bridge  (* = scour-critical)", 0.030, "l"), ("Lat", 0.330, "r"),
            ("Lon", 0.405, "r"), ("Trigger", 0.455, "l"), ("Sev", 0.585, "r"),
            ("Observed", 0.700, "r"), ("Threshold", 0.820, "r"), ("A&A", 0.900, "l")]
    ev = ev.sort_values(["severity_rp", "observed"], ascending=[False, False])
    pages = int(np.ceil(len(ev) / ROWS_PER_PAGE))
    for pg in range(pages):
        chunk = ev.iloc[pg * ROWS_PER_PAGE:(pg + 1) * ROWS_PER_PAGE]
        fig = plt.figure(figsize=(11.0, 8.5), facecolor="white")
        fig.text(0.030, 0.968, f"Triggered bridges — page {pg + 1} of {pages}",
                 fontsize=14, fontweight="bold", color=INK, va="top")
        y = 0.915
        for name, x, al in cols:
            fig.text(x, y, name, fontsize=9, fontweight="bold", color=INK2,
                     ha={"l": "left", "r": "right"}[al])
        fig.lines.append(plt.Line2D([0.030, 0.970], [y - 0.012, y - 0.012],
                                    transform=fig.transFigure, color=HAIRLINE, lw=0.8))
        y -= 0.032
        for _, r in chunk.iterrows():
            unit = "in" if r["trigger_type"] == "precip" else "cfs"
            fmt = "{:,.2f}" if r["trigger_type"] == "precip" else "{:,.0f}"
            tier = TIER_C.get(int(r["severity_rp"]), INK)
            aa = "—" if r["trigger_type"] == "precip" else (
                "yes" if r["aa_confirms"] else "NO")
            name = str(r["asset"])[:26] + ("*" if bool(r["scour"]) else "")
            vals = [(name, 0.030, "l", INK),
                    (f"{r['lat']:.4f}", 0.330, "r", INK2),
                    (f"{r['lon']:.4f}", 0.405, "r", INK2),
                    ("FLOW" if r["trigger_type"] == "flow" else "PRECIP", 0.455, "l", INK2),
                    (f"{int(r['severity_rp'])}-yr", 0.585, "r", tier),
                    (fmt.format(r["observed"]) + " " + unit, 0.700, "r", INK),
                    (fmt.format(r["threshold"]) + " " + unit, 0.820, "r", INK2),
                    (aa, 0.900, "l", INK2 if aa != "NO" else C_OPEN)]
            for txt, x, al, col in vals:
                fig.text(x, y, txt, fontsize=8.0, color=col, va="top",
                         ha={"l": "left", "r": "right"}[al], family="monospace")
            y -= 0.0255
        pdf.savefig(fig); plt.close(fig)


def build_digest_pdf(ev: pd.DataFrame, cfg: pd.DataFrame | None,
                     counties: pd.DataFrame | None,
                     mrms_hour, nwm_hour) -> bytes:
    """One PDF for the whole run: cover map + paginated bridge table."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _digest_cover(pdf, ev, cfg, counties, mrms_hour, nwm_hour)
        _digest_table(pdf, ev)
    return buf.getvalue()
