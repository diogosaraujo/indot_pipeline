"""Alert PDF builder.

One page, four panels + header, for a bridge whose trigger fired:

    [ MRMS 1-h QPE map, bridge-centered ]  [ 24-h Tc-accumulation precip + P10/50/100 ]
    [ NWM reach streamflow map          ]  [ 24-h streamflow + velocity + Q10/50/100  ]

Reuses the ARI-snapping helpers from scripts/visualize_bridge_event.py.
Renders to PDF bytes (matplotlib's built-in pdf backend — no extra deps).
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from . import config, maps, state  # noqa: E402

log = logging.getLogger("monitor.figure")

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

# x position and alignment per column, kept as data so widths can be retuned
# without touching the drawing loop.
COLS = [("Bridge  (* = scour-critical)", 0.022, "l"), ("Lat", 0.300, "r"),
        ("Lon", 0.363, "r"), ("County", 0.372, "l"), ("City", 0.462, "l"),
        ("River", 0.572, "l"), ("Trigger", 0.700, "l"), ("Sev", 0.775, "r"),
        ("Observed", 0.862, "r"), ("Thresh", 0.940, "r"), ("Age", 0.988, "r")]
HA = {"l": "left", "r": "right"}


def _asset(key_name):
    """Load a digest asset, degrading the page rather than failing the alert."""
    from .s3io import read_parquet
    k = config.keys()
    try:
        return read_parquet(k["bucket"], k[key_name])
    except Exception as e:  # noqa: BLE001
        log.warning("digest asset %s unavailable (%s) — page will omit it", key_name, e)
        return None


def _digest_cover(pdf, ev, cfg, mrms_hour, nwm_hour) -> None:
    """Three panels: rainfall accumulated over the 24 h ENDING at the alert
    hour, then each NWM product AT the alert hour — the values the trigger just
    read, not a daily summary."""
    counties, flow = _asset("counties"), _asset("flowlines")
    hour = nwm_hour or mrms_hour
    la, lo = maps.STATE_EXTENT["lat"], maps.STATE_EXTENT["lon"]

    fig = plt.figure(figsize=maps.PANEL_FIG, facecolor=maps.SURFACE)
    axes = [fig.add_axes(r) for r in maps.panel_rects()]

    # panel 1 — trailing 24 h, summed from the grids the poller banked hourly
    acc = lats = lons = None
    nhr = 0
    if mrms_hour is not None:
        hrs = list(pd.date_range(mrms_hour - pd.Timedelta(hours=23), mrms_hour, freq="1h"))
        acc, lats, lons, nhr = state.accumulate_grid(hrs)
    ax, pm = axes[0], None
    maps.draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
    if acc is not None:
        pm = ax.pcolormesh(lons, lats, np.ma.masked_less(acc, 0.05),
                           cmap=maps.precip_cmap(), vmin=0.05,
                           vmax=max(0.5, float(np.nanpercentile(acc, 99.8))),
                           shading="nearest", zorder=2, alpha=maps.PRECIP_ALPHA,
                           rasterized=True)
    else:
        ax.text(0.5, 0.5, "no stored MRMS grids yet", transform=ax.transAxes,
                ha="center", color=maps.MUTED, fontsize=12)
    maps.draw_counties(ax, counties, lw=0.6, overlay=True)
    ax.set_title("MRMS 24-h accumulation"
                 + (f" to {mrms_hour:%H%M}Z" if mrms_hour is not None else ""),
                 fontsize=13, color=maps.INK, loc="left", pad=8)

    # panels 2 & 3 — NWM at the alert hour, sharing one scale
    q100 = (cfg.dropna(subset=["comid"]).drop_duplicates("comid")
            .set_index("comid")["Q100_cfs"]) if cfg is not None else pd.Series(dtype=float)
    cur = state.read_recent("nwm", [hour]).get(hour) if hour is not None else None
    for ax, col, lbl in ((axes[1], "q_ol_cms", "NWM open-loop (trigger)"),
                         (axes[2], "q_aa_cms", "NWM A&A (with DA)")):
        maps.draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")
        if cur is not None and col in cur.columns:
            c = cur.set_index("comid")
            maps.draw_flowlines(ax, flow,
                                (c[col] * config.CFS_PER_CMS) / q100.reindex(c.index),
                                vmax=1.5, lw_base=0.55, lat=la, lon=lo)
        else:
            ax.text(0.5, 0.5, "product unavailable", transform=ax.transAxes,
                    ha="center", color=maps.MUTED, fontsize=12)
        maps.draw_counties(ax, counties, lw=0.5, overlay=True)
        ax.set_title(lbl + (f" at {hour:%H%M}Z" if hour is not None else ""),
                     fontsize=13, color=maps.INK, loc="left", pad=8)

    one = ev.drop_duplicates("bridge_id")
    for ax in axes:
        maps.draw_bridges(ax, one, 1.9)
        maps.set_geo(ax, la, lo)

    cb_rect, ramp_rect = maps.panel_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("24-h accumulation (in)", fontsize=9, color=maps.INK2)
        cb.ax.tick_params(labelsize=8, colors=maps.INK2)
    maps.river_ramp_legend(fig, ramp_rect,
                           label="flow ÷ reach 100-yr Q  (both NWM panels share this scale)")

    n_scour = int(ev["scour"].astype(bool).sum())
    fig.text(0.028, 0.975, f"INDOT BRIDGE FLOOD ALERT — {ev['bridge_id'].nunique()} bridge(s)",
             fontsize=20, fontweight="bold", color=maps.INK, va="top")
    bits = [f"valid {hour:%Y-%m-%d %H:%M} UTC" if hour is not None else "valid time unknown",
            f"{int((ev['trigger_type'] == 'flow').sum())} streamflow",
            f"{int((ev['trigger_type'] == 'precip').sum())} precipitation"]
    if n_scour:
        bits.append(f"{n_scour} scour-critical")
    if 0 < nhr < 24:
        bits.append(f"{nhr}/24 MRMS hours available")
    fig.text(0.028, 0.938, "   ·   ".join(bits), fontsize=11.5, color=maps.INK2, va="top")

    x = 0.560
    for cls, (c, m, lbl) in maps.CLASS_STYLE.items():
        n = int((ev["map_class"] == cls).sum())
        if not n:
            continue
        fig.text(x, 0.975, "●" if m == "o" else "▲", fontsize=13, color=c, va="top")
        fig.text(x + 0.013, 0.973, f"{lbl} ({n})", fontsize=10.5, color=maps.INK2, va="top")
        x += 0.150
    x = 0.560
    fig.text(x, 0.938, "severity = size:", fontsize=10.5, color=maps.INK, va="top")
    x += 0.082
    for rp in (10, 50, 100):
        n = int((ev["severity_rp"] == rp).sum())
        fig.text(x, 0.938, "•", fontsize=6 + np.sqrt(maps.SEV_SIZE[rp]) * 0.95,
                 color=maps.INK2, va="top")
        fig.text(x + 0.016, 0.938, f"{rp}-yr ({n})", fontsize=10, color=maps.INK2, va="top")
        x += 0.098
    pdf.savefig(fig, dpi=150); plt.close(fig)


def _digest_table(pdf, ev: pd.DataFrame) -> None:
    """The bridges in THIS alert, with the location context the daily reports
    carry — county, nearest city, and the named waterway."""
    places = _asset("places")
    d = ev.sort_values(["severity_rp", "observed"], ascending=[False, False])
    if places is not None:
        d = d.merge(places, on="bridge_id", how="left")
    for c in ("county", "city", "city_mi", "river"):
        if c not in d.columns:
            d[c] = np.nan

    pages = int(np.ceil(len(d) / ROWS_PER_PAGE)) or 1
    for pg in range(pages):
        chunk = d.iloc[pg * ROWS_PER_PAGE:(pg + 1) * ROWS_PER_PAGE]
        fig = plt.figure(figsize=(11.0, 8.5), facecolor="white")
        fig.text(0.022, 0.968, "Triggered bridges", fontsize=16, fontweight="bold",
                 color=maps.INK, va="top")
        fig.text(0.022, 0.934,
                 f"page {pg + 1} of {pages}   ·   {len(d)} row(s)   ·   "
                 "city is the NEAREST place   ·   FLOW·NO = open-loop only, "
                 "A&A does not corroborate",
                 fontsize=9, color=maps.MUTED, va="top")
        y = 0.900
        for name, x, al in COLS:
            fig.text(x, y, name, fontsize=8, fontweight="bold", color=maps.INK2, ha=HA[al])
        fig.lines.append(plt.Line2D([0.022, 0.978], [y - 0.011, y - 0.011],
                                    transform=fig.transFigure, color=maps.HAIRLINE, lw=0.8))
        y -= 0.030
        for _, r in chunk.iterrows():
            unit = "in" if r["trigger_type"] == "precip" else "cfs"
            fmt = "{:,.2f}" if unit == "in" else "{:,.0f}"
            tier = maps.TIER_C.get(int(r["severity_rp"]), maps.INK)
            trig = ("PRECIP" if r["trigger_type"] == "precip"
                    else ("FLOW" if r.get("aa_confirms") else "FLOW·NO"))
            city = str(r["city"])[:12] if pd.notna(r["city"]) else "—"
            if pd.notna(r.get("city_mi")):
                city = f"{city} {r['city_mi']:.0f}mi"
            vals = [
                (str(r["asset"])[:26] + ("*" if bool(r["scour"]) else ""), 0.022, "l", maps.INK),
                (f"{r['lat']:.4f}", 0.300, "r", maps.INK2),
                (f"{r['lon']:.4f}", 0.363, "r", maps.INK2),
                (str(r["county"])[:11] if pd.notna(r["county"]) else "—", 0.372, "l", maps.INK2),
                (city, 0.462, "l", maps.INK2),
                (str(r["river"])[:19] if pd.notna(r["river"]) else "unnamed", 0.572, "l",
                 maps.INK2 if pd.notna(r["river"]) else maps.MUTED),
                (trig, 0.700, "l", maps.C_OPEN if trig == "FLOW·NO" else maps.INK2),
                (f"{int(r['severity_rp'])}-yr", 0.775, "r", tier),
                (fmt.format(r["observed"]) + " " + unit, 0.862, "r", maps.INK),
                (fmt.format(r["threshold"]) + " " + unit, 0.940, "r", maps.INK2),
                (f"{r['event_hours']:.0f}h" if pd.notna(r.get("event_hours"))
                 else "—", 0.988, "r",
                 maps.C_OPEN if (r.get("event_hours") or 0) >= 24 else maps.INK2),
            ]
            for txt, x, al, col in vals:
                fig.text(x, y, txt, fontsize=6.8, color=col, va="top",
                         ha=HA[al], family="monospace")
            y -= 0.0245
        pdf.savefig(fig); plt.close(fig)


def build_digest_pdf(ev: pd.DataFrame, cfg: pd.DataFrame | None,
                     counties: pd.DataFrame | None = None,
                     mrms_hour=None, nwm_hour=None) -> bytes:
    """One PDF per run: 3-panel cover + paginated bridge table.

    `counties` is accepted for backward compatibility and ignored — the cover
    loads the assets it needs so callers don't have to know which those are.
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _digest_cover(pdf, ev, cfg, mrms_hour, nwm_hour)
        _digest_table(pdf, ev)
    return buf.getvalue()
