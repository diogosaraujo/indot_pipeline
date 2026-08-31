"""Daily summary PDF — what the last calendar day did, whether or not it fired.

Sent every morning at DAILY_SEND_HOUR local for the previous LOCAL calendar day.
It goes out on quiet days too, deliberately: a report that only arrives on bad
days cannot distinguish "nothing happened" from "the monitor is dead", and that
ambiguity is exactly what hid a 13-day outage.

    page 1   MRMS 24-h accumulation        | the ARI that accumulation represents
    page 2   NWM A&A daily peak flow       | the ARI of that peak
    page 3   NWM open-loop daily peak flow | the ARI of that peak
    page 4+  the bridges involved, still-above-threshold ones highlighted

Why the DAILY PEAK for flow. Page 1's left panel is an aggregate over the whole
day (a sum), so its flow analogue has to be an aggregate too. Streamflow cannot
be summed, and the instantaneous value at an arbitrary cutoff hour would miss a
crest that passed at 03:00. The peak is what threatens a bridge, so the peak is
what the day is summarised by.

Bridges shown come from the hourly ALARM STATE — those still above threshold,
plus those whose alarm tripped during the window — never from the ARI rasters.
The rasters describe the weather; only the alarm knows about a bridge, and a
100-yr rain over a reach with no bridge on it is not an inspection.
"""
from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from . import config, maps, state  # noqa: E402

log = logging.getLogger("monitor.daily")

# How stale the last wet hour may be and still count as "currently above". One
# missed poll must not silently downgrade a standing exceedance to "receded".
STILL_ABOVE_SLACK_H = 3

# Reaches carrying less than this fraction of their 100-yr flow are drawn as
# quiet context rather than coloured, mirroring the 0.05 in floor on the
# precipitation panel. Both exist so a calm day looks calm.
RATIO_FLOOR = 0.05


# ── window ───────────────────────────────────────────────────────────────────

def previous_local_day(now_utc: pd.Timestamp | None = None):
    """(start_utc, end_utc, local_date) for the last complete local day.

    end is EXCLUSIVE. Returned in UTC because every stored hour is UTC.
    """
    now = pd.Timestamp.utcnow() if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    loc = now.tz_convert(config.DAILY_TZ)
    day = (loc.normalize() - pd.Timedelta(days=1)).date()
    start_l = pd.Timestamp(day, tz=config.DAILY_TZ)
    end_l = start_l + pd.Timedelta(days=1)
    return start_l.tz_convert("UTC"), end_l.tz_convert("UTC"), day


def window_hours(start_utc, end_utc) -> list[pd.Timestamp]:
    """Hourly stamps in [start, end) — the hours the summary aggregates."""
    return list(pd.date_range(start_utc, end_utc - pd.Timedelta(hours=1), freq="1h"))


# ── ARI ──────────────────────────────────────────────────────────────────────

def ari_from_curves(values: np.ndarray, curve: np.ndarray,
                    aris: np.ndarray) -> np.ndarray:
    """Return period of `values`, given a per-element depth/discharge curve.

    values : shape S
    curve  : shape (n_ari, *S) — curve[i] is the value at aris[i], ascending
    Log-log interpolation, matching figure.ari_from_value and the log-log
    duration interpolation p04 uses; the shape here is only so it can run over a
    whole raster at once instead of per cell.

    Outside the curve's span it extrapolates on the end segment. That is fine
    for precipitation (ten anchors, 1..1000 yr) and coarse for flow (three:
    Q10/Q50/Q100) — the caller says so in the caption rather than pretending.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log10(np.where(np.asarray(values, float) > 0, values, np.nan))
        ld = np.log10(np.where(np.asarray(curve, float) > 0, curve, np.nan))
    la = np.log10(np.asarray(aris, float))
    n = ld.shape[0]
    if n < 2:
        return np.full(np.shape(values), np.nan)

    # index of the last anchor at or below the value, clamped to a usable segment
    k = np.nansum(ld <= lv[None, ...], axis=0) - 1
    k = np.clip(k, 0, n - 2)
    d0 = np.take_along_axis(ld, k[None, ...], 0)[0]
    d1 = np.take_along_axis(ld, (k + 1)[None, ...], 0)[0]
    a0, a1 = la[k], la[k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (lv - d0) / (d1 - d0)
    out = 10.0 ** (a0 + t * (a1 - a0))
    return np.where(np.isfinite(out), out, np.nan)


def _asset_npz(key_name: str):
    from .s3io import read_bytes
    k = config.keys()
    try:
        return np.load(io.BytesIO(read_bytes(k["bucket"], k[key_name])))
    except Exception as e:  # noqa: BLE001
        log.warning("asset %s unavailable (%s)", key_name, e)
        return None


def _asset(key_name: str):
    from .s3io import read_parquet
    k = config.keys()
    try:
        return read_parquet(k["bucket"], k[key_name])
    except Exception as e:  # noqa: BLE001
        log.warning("asset %s unavailable (%s) — page will omit it", key_name, e)
        return None


# ── aggregation ──────────────────────────────────────────────────────────────

def accumulate_precip(hours):
    """(sum_in, lats, lons, n_hours_found) over the window."""
    return state.accumulate_grid(hours)


def peak_flow(hours) -> pd.DataFrame:
    """Per-COMID daily PEAK of both NWM products, and the hour each peaked.

    Returns q_ol_cfs / q_aa_cfs and peak_ol_hour / peak_aa_hour, indexed by
    comid. n_hours records how many of the window's hours were actually stored,
    so a partial day can be labelled rather than silently under-reported.
    """
    got = state.read_recent("nwm", hours)
    frames = []
    for ts, df in sorted((got or {}).items()):
        if df is None or getattr(df, "empty", True):
            continue
        d = df.copy()
        if d.index.name != "comid":
            if "comid" not in d.columns:
                continue
            d = d.set_index("comid")
        keep = [c for c in ("q_ol_cms", "q_aa_cms") if c in d.columns]
        if not keep:
            continue
        d = d[keep].copy()
        d["hour"] = ts
        frames.append(d)
    if not frames:
        return pd.DataFrame()

    allh = pd.concat(frames).rename_axis("comid").reset_index()
    out = pd.DataFrame(index=pd.Index(allh["comid"].unique(), name="comid"))
    for col, tag in (("q_ol_cms", "ol"), ("q_aa_cms", "aa")):
        if col not in allh.columns:
            continue
        s = allh.dropna(subset=[col])
        if s.empty:
            continue
        # sort then take the last row per comid = the row holding the max
        top = s.sort_values(col).groupby("comid").last()
        out[f"q_{tag}_cfs"] = top[col] * config.CFS_PER_CMS
        out[f"peak_{tag}_hour"] = top["hour"]
    out.attrs["n_hours"] = len(frames)
    return out


# ── bridge selection ─────────────────────────────────────────────────────────

def bridges_for_window(alert_state: pd.DataFrame, cfg: pd.DataFrame,
                       start_utc, end_utc, latest_hour) -> pd.DataFrame:
    """Bridges still above threshold, plus those whose alarm tripped in-window.

    Taken from the alarm state only. `first_trigger` is when THIS event first
    tripped, not when it was last seen — that is the question an inspector asks.
    """
    empty = pd.DataFrame(columns=[
        "bridge_id", "trigger_type", "severity_rp", "observed", "threshold",
        "first_trigger", "still_above", "lat", "lon", "comid", "scour", "asset",
        "map_class", "event_hours"])
    if alert_state is None or alert_state.empty:
        return empty

    st = alert_state.copy()
    for c in ("last_wet_hour", "last_alert_hour", "event_start_hour"):
        st[c] = pd.to_datetime(st.get(c), errors="coerce", utc=True)

    still = pd.Series(False, index=st.index)
    if latest_hour is not None:
        still = st["last_wet_hour"] >= (latest_hour - pd.Timedelta(hours=STILL_ABOVE_SLACK_H))
    tripped = st["last_alert_hour"].between(start_utc, end_utc - pd.Timedelta(seconds=1))
    # A bridge wet at any point inside the window counts too: it may have
    # tripped before the window and receded during it, which is still the day's
    # news even though nothing "fired" today.
    wet_in = st["last_wet_hour"].between(start_utc, end_utc - pd.Timedelta(seconds=1))

    sel = st[still | tripped | wet_in].copy()
    if sel.empty:
        return empty
    sel["still_above"] = still[sel.index].fillna(False)
    sel["first_trigger"] = sel["event_start_hour"].fillna(sel["last_alert_hour"])
    sel["severity_rp"] = pd.to_numeric(sel.get("last_severity_rp"), errors="coerce").fillna(0).astype(int)
    sel["observed"] = pd.to_numeric(sel.get("last_observed"), errors="coerce")
    sel["threshold"] = pd.to_numeric(sel.get("last_threshold"), errors="coerce")
    sel["event_hours"] = ((latest_hour - sel["first_trigger"]).dt.total_seconds() / 3600.0
                          if latest_hour is not None else np.nan)

    meta = cfg.drop_duplicates("bridge_id").set_index("bridge_id")
    for c, src in (("lat", "lat"), ("lon", "lon"), ("comid", "comid"),
                   ("tc_dur_hr", "tc_dur_hr")):
        sel[c] = sel["bridge_id"].map(meta[src]) if src in meta.columns else np.nan
    sel["scour"] = sel["bridge_id"].map(meta[config.SCOUR_COL]).fillna(False).astype(bool) \
        if config.SCOUR_COL in meta.columns else False
    sel["asset"] = (sel["bridge_id"].map(meta[config.ASSET_COL])
                    if config.ASSET_COL in meta.columns else sel["bridge_id"])
    sel["asset"] = sel["asset"].fillna(sel["bridge_id"])
    return sel.reset_index(drop=True)


def _classify(ev: pd.DataFrame, peaks: pd.DataFrame,
              cfg: pd.DataFrame | None = None) -> pd.DataFrame:
    """map_class for the shared symbology: precip / flow A&A-confirmed / flow open-loop.

    Corroboration is judged against the bridge's OWN gate recomputed from the
    config (Q10 if scour-critical, else Q50), not against the threshold stored
    on the alarm row. The stored value is absent for events that began before
    the summary existed, and a missing threshold would silently demote every
    such bridge to "open-loop only" — the label that says the products disagree.
    """
    d = ev.copy()
    d["q_aa_cfs"] = np.nan
    if peaks is not None and not peaks.empty and "q_aa_cfs" in peaks.columns:
        d["q_aa_cfs"] = pd.to_numeric(d["comid"], errors="coerce").map(peaks["q_aa_cfs"])

    gate = pd.to_numeric(d.get("threshold"), errors="coerce")
    if cfg is not None and not cfg.empty:
        meta = cfg.drop_duplicates("bridge_id").set_index("bridge_id")
        scour = d["bridge_id"].map(meta.get(config.SCOUR_COL)).fillna(False).astype(bool)
        q10 = pd.to_numeric(d["bridge_id"].map(meta.get("Q10_cfs")), errors="coerce")
        q50 = pd.to_numeric(d["bridge_id"].map(meta.get("Q50_cfs")), errors="coerce")
        own = np.where(scour, q10, q50)
        gate = gate.where(gate.notna(), pd.Series(own, index=d.index))
    d["gate_cfs"] = gate

    d["aa_confirms"] = (d["trigger_type"].eq("flow")
                        & (d["q_aa_cfs"] >= gate)).fillna(False)
    d["map_class"] = np.where(
        d["trigger_type"].eq("precip"), "precip",
        np.where(d["aa_confirms"], "flow_conf",
                 np.where(d["q_aa_cfs"].isna(), "unknown", "flow_open")))
    return d


# ── pages ────────────────────────────────────────────────────────────────────

def _header(fig, title: str, subtitle: str) -> None:
    fig.text(maps.PAIR_X0, 0.972, title, fontsize=19, fontweight="bold",
             color=maps.INK, va="top")
    fig.text(maps.PAIR_X0, 0.936, subtitle, fontsize=11, color=maps.INK2, va="top")


def _footer(fig, text: str) -> None:
    fig.text(maps.PAIR_X0, 0.022, text, fontsize=7.6, color=maps.MUTED, va="bottom")


def _bridge_legend(fig, ev: pd.DataFrame, x0: float = 0.560) -> None:
    """Marker key, drawn only for the classes actually present."""
    x = x0
    for cls, (c, m, lbl) in maps.CLASS_STYLE.items():
        n = int((ev.get("map_class", pd.Series(dtype=str)) == cls).sum())
        if not n:
            continue
        fig.text(x, 0.972, "●" if m == "o" else "▲", fontsize=13, color=c, va="top")
        fig.text(x + 0.014, 0.970, f"{lbl} ({n})", fontsize=10, color=maps.INK2, va="top")
        x += 0.150


def _quiet_note(ax, text: str) -> None:
    """Say 'nothing here' explicitly. An empty panel is ambiguous between a calm
    day and a broken render, and this report exists to remove that ambiguity."""
    ax.text(0.5, 0.035, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=maps.INK2, zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=maps.HAIRLINE, linewidth=0.7, alpha=0.94))


def _draw_points(axes, ev: pd.DataFrame) -> None:
    if ev is None or ev.empty:
        return
    one = ev.drop_duplicates("bridge_id")
    for ax in axes:
        maps.draw_bridges(ax, one, 1.9)


def _precip_page(pdf, ev, hours, counties, atlas, day, nhr_expected) -> None:
    acc, lats, lons, nhr = accumulate_precip(hours)
    la, lo = maps.STATE_EXTENT["lat"], maps.STATE_EXTENT["lon"]
    fig = plt.figure(figsize=maps.PAIR_FIG, facecolor=maps.SURFACE)
    axes = [fig.add_axes(r) for r in maps.pair_rects()]

    pm = am = None
    for ax in axes:
        maps.draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")

    if acc is None:
        for ax in axes:
            ax.text(0.5, 0.5, "no stored MRMS grids for this day", transform=ax.transAxes,
                    ha="center", color=maps.MUTED, fontsize=12)
    else:
        pm = axes[0].pcolormesh(lons, lats, np.ma.masked_less(acc, 0.05),
                                cmap=maps.precip_cmap(), vmin=0.05,
                                vmax=max(0.5, float(np.nanpercentile(acc, 99.8))),
                                shading="nearest", zorder=2, alpha=maps.PRECIP_ALPHA,
                                rasterized=True)
        if atlas is not None:
            ari = ari_from_curves(acc, atlas["depth"], atlas["ari"])
            cm, norm = maps.ari_cmap_norm()
            # No scalar alpha here — it is baked into the colormap, because a
            # scalar would repaint the transparent under/bad colours opaque.
            am = axes[1].pcolormesh(lons, lats, maps.mask_below_ari(ari),
                                    cmap=cm, norm=norm, shading="nearest",
                                    zorder=2, rasterized=True)
            top = float(np.nanmax(ari)) if np.isfinite(ari).any() else 0.0
            if top < maps.ARI_BOUNDS[0]:
                _quiet_note(axes[1], "no cell reached a 1-year recurrence")
        else:
            axes[1].text(0.5, 0.5, "Atlas-14 grid unavailable", transform=axes[1].transAxes,
                         ha="center", color=maps.MUTED, fontsize=12)

    for ax in axes:
        maps.draw_counties(ax, counties, lw=0.6, overlay=True)
    _draw_points(axes, ev)
    for ax in axes:
        maps.set_geo(ax, la, lo)
    axes[0].set_title("MRMS 24-h accumulation", fontsize=13, color=maps.INK, loc="left", pad=8)
    axes[1].set_title("Return period of that accumulation — Atlas-14, 24 h",
                      fontsize=13, color=maps.INK, loc="left", pad=8)

    lrect, rrect = maps.pair_legend_rects()
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(lrect), orientation="horizontal")
        cb.set_label("24-h accumulation (in)", fontsize=9, color=maps.INK2)
        cb.ax.tick_params(labelsize=8, colors=maps.INK2)
    if am is not None:
        maps.ari_legend(fig, rrect)

    bits = [f"{day}", f"{config.DAILY_WINDOW_HOURS}-h window, local midnight to midnight"]
    if 0 < nhr < nhr_expected:
        bits.append(f"{nhr}/{nhr_expected} MRMS hours available")
    _header(fig, "DAILY SUMMARY — precipitation", "   ·   ".join(bits))
    _bridge_legend(fig, ev)
    _footer(fig, "Bridge markers come from the hourly alarm state, never from the ARI raster. "
                 "Atlas-14 depths are the published gridded 24-h surface (NOAA Atlas 14 Vol. 2), "
                 "sampled on the MRMS grid.")
    pdf.savefig(fig, dpi=150); plt.close(fig)


def _flow_page(pdf, ev, peaks, cfg, counties, flow, tag, label, day) -> None:
    la, lo = maps.STATE_EXTENT["lat"], maps.STATE_EXTENT["lon"]
    fig = plt.figure(figsize=maps.PAIR_FIG, facecolor=maps.SURFACE)
    axes = [fig.add_axes(r) for r in maps.pair_rects()]
    for ax in axes:
        maps.draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")

    col = f"q_{tag}_cfs"
    have = peaks is not None and not peaks.empty and col in peaks.columns
    if not have:
        for ax in axes:
            ax.text(0.5, 0.5, "product unavailable for this day", transform=ax.transAxes,
                    ha="center", color=maps.MUTED, fontsize=12)
    else:
        q = peaks[col].dropna()
        reach = (cfg.dropna(subset=["comid"]).drop_duplicates("comid")
                 .set_index("comid")[["Q10_cfs", "Q50_cfs", "Q100_cfs"]])
        r = reach.reindex(q.index)
        # Floor the ratio panel the way page 1 floors accumulation at 0.05 in.
        # Without it a quiet day paints all ~7,000 reaches at the bottom of the
        # ramp — a solid yellow state that arrives every morning and directs the
        # eye at nothing. Below the floor a reach reverts to the quiet grey that
        # is the network's context.
        ratio = q / r["Q100_cfs"]
        maps.draw_flowlines(axes[0], flow, ratio[ratio >= RATIO_FLOOR], vmax=1.5,
                            lw_base=0.55, lat=la, lon=lo, lw_min=1.5)
        curve = np.vstack([r["Q10_cfs"].to_numpy(), r["Q50_cfs"].to_numpy(),
                           r["Q100_cfs"].to_numpy()])
        ari = pd.Series(ari_from_curves(q.to_numpy(), curve, np.array([10, 50, 100])),
                        index=q.index)
        # Reaches under a 1-year recurrence are dropped rather than coloured:
        # the colormap's under-colour is transparent, so they would be drawn as
        # invisible lines and punch holes in the network. Left unvalued they
        # fall through to the quiet grey that gives the loud reaches context.
        cm, norm = maps.ari_cmap_norm()
        hot = ari[ari >= maps.ARI_BOUNDS[0]]
        maps.draw_flowlines(axes[1], flow, hot, cmap=cm, norm=norm,
                            width_vmax=100.0, lw_base=0.55, lat=la, lon=lo,
                            lw_min=2.4)
        if hot.empty:
            _quiet_note(axes[1], "no reach reached a 1-year recurrence")
        if (ratio >= RATIO_FLOOR).sum() == 0:
            _quiet_note(axes[0], f"no reach above {RATIO_FLOOR:.0%} of its 100-yr flow")

    for ax in axes:
        maps.draw_counties(ax, counties, lw=0.5, overlay=True)
    _draw_points(axes, ev)
    for ax in axes:
        maps.set_geo(ax, la, lo)
    axes[0].set_title(f"{label} — daily peak streamflow", fontsize=13,
                      color=maps.INK, loc="left", pad=8)
    axes[1].set_title(f"Return period of that peak — retro-LP3 (04c)", fontsize=13,
                      color=maps.INK, loc="left", pad=8)

    lrect, rrect = maps.pair_legend_rects()
    maps.river_ramp_legend(fig, lrect, label="peak flow ÷ reach 100-yr Q")
    maps.ari_legend(fig, rrect)

    _header(fig, f"DAILY SUMMARY — {label}",
            f"{day}   ·   daily maximum over the local calendar day")
    _bridge_legend(fig, ev)
    _footer(fig, "Peak = the highest hourly value in the window; streamflow cannot be summed, "
                 "and the crest is what threatens a bridge. Flow ARI interpolates a THREE-point "
                 "LP3 curve (Q10/Q50/Q100), so it is coarse below 10 yr and above 100 yr.")
    pdf.savefig(fig, dpi=150); plt.close(fig)


# ── bridge table ─────────────────────────────────────────────────────────────

ROWS_PER_PAGE = 30

# (header, x, align) — kept as data so widths retune without touching the loop.
COLS = [("Bridge  (* = scour-critical)", 0.020, "l"), ("Lat", 0.268, "r"),
        ("Lon", 0.330, "r"), ("County", 0.338, "l"), ("City", 0.424, "l"),
        ("River", 0.524, "l"), ("Trigger", 0.646, "l"), ("Sev", 0.716, "r"),
        ("Observed", 0.800, "r"), ("Threshold", 0.878, "r"),
        ("First trigger (UTC)", 0.960, "r"), ("For", 0.995, "r")]
HA = {"l": "left", "r": "right"}

HILITE = "#fdf1d6"          # still-above row band — amber-tinted, prints legibly


def _bridge_pages(pdf, ev: pd.DataFrame, day, latest_hour) -> None:
    places = _asset("places")
    d = ev.copy()
    if places is not None:
        d = d.merge(places, on="bridge_id", how="left")
    for c in ("county", "city", "city_mi", "river"):
        if c not in d.columns:
            d[c] = np.nan

    # Still-above first: those are the ones an inspector acts on today.
    d = d.sort_values(["still_above", "severity_rp", "observed"],
                      ascending=[False, False, False])
    n_above = int(d["still_above"].sum())
    pages = int(np.ceil(len(d) / ROWS_PER_PAGE)) or 1

    for pg in range(pages):
        chunk = d.iloc[pg * ROWS_PER_PAGE:(pg + 1) * ROWS_PER_PAGE]
        fig = plt.figure(figsize=(17.0, 9.6), facecolor="white")
        fig.text(0.020, 0.968, "Bridges involved", fontsize=17, fontweight="bold",
                 color=maps.INK, va="top")
        fig.text(0.020, 0.934,
                 f"{day}   ·   page {pg + 1} of {pages}   ·   {len(d)} row(s)   ·   "
                 f"{n_above} STILL ABOVE THRESHOLD (highlighted)   ·   "
                 "city is the NEAREST place   ·   FLOW·NO = open-loop only, A&A does not corroborate",
                 fontsize=9, color=maps.MUTED, va="top")

        y = 0.900
        for name, x, al in COLS:
            fig.text(x, y, name, fontsize=8, fontweight="bold", color=maps.INK2, ha=HA[al])
        fig.lines.append(plt.Line2D([0.020, 0.997], [y - 0.011, y - 0.011],
                                    transform=fig.transFigure, color=maps.HAIRLINE, lw=0.8))
        y -= 0.030
        row_h = 0.0265
        for _, r in chunk.iterrows():
            above = bool(r["still_above"])
            if above:
                fig.patches.append(plt.Rectangle(
                    (0.016, y - row_h + 0.012), 0.985, row_h, transform=fig.transFigure,
                    facecolor=HILITE, edgecolor="none", zorder=0))
            unit = "in" if r["trigger_type"] == "precip" else "cfs"
            fmt = "{:,.2f}" if unit == "in" else "{:,.0f}"
            tier = maps.TIER_C.get(int(r["severity_rp"]), maps.INK)
            trig = ("PRECIP" if r["trigger_type"] == "precip"
                    else ("FLOW" if r.get("aa_confirms") else "FLOW·NO"))
            city = str(r["city"])[:12] if pd.notna(r["city"]) else "—"
            if pd.notna(r.get("city_mi")):
                city = f"{city} {r['city_mi']:.0f}mi"
            ft = r.get("first_trigger")
            ft_s = f"{ft:%Y-%m-%d %H:%M}" if pd.notna(ft) else "—"
            hrs = r.get("event_hours")
            for_s = f"{hrs:.0f}h" if pd.notna(hrs) else "—"
            w = "bold" if above else "normal"
            vals = [
                (str(r["asset"])[:30] + ("*" if bool(r["scour"]) else ""), 0.020, "l", maps.INK),
                (f"{r['lat']:.4f}" if pd.notna(r["lat"]) else "—", 0.268, "r", maps.INK2),
                (f"{r['lon']:.4f}" if pd.notna(r["lon"]) else "—", 0.330, "r", maps.INK2),
                (str(r["county"])[:11] if pd.notna(r["county"]) else "—", 0.338, "l", maps.INK2),
                (city, 0.424, "l", maps.INK2),
                (str(r["river"])[:18] if pd.notna(r["river"]) else "unnamed", 0.524, "l",
                 maps.INK2 if pd.notna(r["river"]) else maps.MUTED),
                (trig, 0.646, "l", maps.C_OPEN if trig == "FLOW·NO" else maps.INK2),
                (f"{int(r['severity_rp'])}-yr", 0.716, "r", tier),
                (fmt.format(r["observed"]) + " " + unit if pd.notna(r["observed"]) else "—",
                 0.800, "r", maps.INK),
                (fmt.format(r["threshold"]) + " " + unit if pd.notna(r["threshold"]) else "—",
                 0.878, "r", maps.INK2),
                (ft_s, 0.960, "r", maps.INK if above else maps.INK2),
                (for_s, 0.995, "r", maps.INK2),
            ]
            for txt, x, al, colr in vals:
                fig.text(x, y, txt, fontsize=7.0, color=colr, va="top",
                         ha=HA[al], family="monospace", fontweight=w, zorder=2)
            y -= row_h
        fig.text(0.020, 0.026,
                 "Highlighted + bold = still above its firing threshold at the latest "
                 f"evaluated hour ({latest_hour:%Y-%m-%d %H:%M} UTC). "
                 "'First trigger' is when THIS event first crossed, which may predate the window.",
                 fontsize=7.6, color=maps.MUTED, va="bottom")
        pdf.savefig(fig); plt.close(fig)


def _empty_note(pdf, day, latest_hour) -> None:
    fig = plt.figure(figsize=(17.0, 9.6), facecolor="white")
    fig.text(0.020, 0.968, "Bridges involved", fontsize=17, fontweight="bold",
             color=maps.INK, va="top")
    fig.text(0.020, 0.934, f"{day}", fontsize=9, color=maps.MUTED, va="top")
    fig.text(0.5, 0.5,
             "No bridge was above its firing threshold at any point in this window.",
             fontsize=15, color=maps.INK2, ha="center", va="center")
    fig.text(0.5, 0.44,
             f"Latest evaluated hour {latest_hour:%Y-%m-%d %H:%M} UTC."
             if latest_hour is not None else "No evaluated hours found.",
             fontsize=10, color=maps.MUTED, ha="center", va="center")
    pdf.savefig(fig); plt.close(fig)


# ── entry point ──────────────────────────────────────────────────────────────

def build_daily_pdf(cfg: pd.DataFrame, alert_state: pd.DataFrame,
                    now_utc=None) -> tuple[bytes, dict, pd.DataFrame]:
    """(pdf_bytes, meta, events) for the previous local calendar day.

    The event table comes back with the PDF so the caller can write the email
    body from it — rebuilding it there would re-read every stored NWM hour to
    recompute the same peaks.
    """
    start, end, day = previous_local_day(now_utc)
    hours = window_hours(start, end)

    counties, flow = _asset("counties"), _asset("flowlines")
    atlas = _asset_npz("atlas14_grid")
    peaks = peak_flow(hours)

    # "Still above" is judged against the most recent hour the poller actually
    # evaluated, not against the window end — the window closed hours ago and a
    # bridge that receded since then is no longer an inspection.
    seen = sorted(state.existing_hours("nwm"))
    latest_hour = seen[-1] if seen else (hours[-1] if hours else None)

    ev = bridges_for_window(alert_state, cfg, start, end, latest_hour)
    ev = _classify(ev, peaks, cfg)
    # Fall back to the day's peak when the alarm row predates last_observed, so
    # the table shows a real reading rather than an em dash.
    if not ev.empty and peaks is not None and not peaks.empty:
        cid = pd.to_numeric(ev["comid"], errors="coerce")
        for col, src in (("observed", "q_ol_cfs"),):
            if src in peaks.columns:
                ev[col] = ev[col].where(ev[col].notna(),
                                        cid.map(peaks[src]).where(ev["trigger_type"].eq("flow")))
        ev["threshold"] = ev["threshold"].where(ev["threshold"].notna(), ev["gate_cfs"])

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _precip_page(pdf, ev, hours, counties, atlas, day, len(hours))
        _flow_page(pdf, ev, peaks, cfg, counties, flow, "aa", "NWM A&A (with DA)", day)
        _flow_page(pdf, ev, peaks, cfg, counties, flow, "ol", "NWM open-loop (trigger)", day)
        if ev.empty:
            _empty_note(pdf, day, latest_hour)
        else:
            _bridge_pages(pdf, ev, day, latest_hour)

    meta = {
        "day": str(day), "start": start.isoformat(), "end": end.isoformat(),
        "window_hours": len(hours),
        "bridges": int(ev["bridge_id"].nunique()) if not ev.empty else 0,
        "still_above": int(ev["still_above"].sum()) if not ev.empty else 0,
        "nwm_hours": int(peaks.attrs.get("n_hours", 0)) if peaks is not None else 0,
        "latest_hour": None if latest_hour is None else latest_hour.isoformat(),
    }
    return buf.getvalue(), meta, ev
