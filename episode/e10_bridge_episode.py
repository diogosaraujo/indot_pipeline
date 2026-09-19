"""e10 — episode-long animation and peak-of-period plate for ONE bridge's area.

e04/e05 cut the episode into local days: one GIF and one plate per day, per
extent. That is the right unit for a statewide review and the wrong one for a
post-mortem of a single structure, where the question is what the WHOLE sequence
did over that bridge's own neighbourhood. This script answers it with two
products on identical ground:

  GIF     every hour of the period, two panels — MRMS 1-h QPE | NWM streamflow —
          with the alert markers accumulating across DAYS rather than resetting
          at local midnight, and a timeline strip so a 120-frame loop can still
          be located in time. Runs faster than the daily GIFs (default 6 fps):
          five days of frames at the daily 2.5 fps is a 48-second loop.

  PLATE   the same two panels reduced to the period's peaks —
            * rainfall: the PEAK 24-h accumulation, i.e. the largest of every
              rolling 24-h window in the period, per cell. Not the largest
              calendar-day total: a storm that straddles midnight is cut in half
              by the day boundary and never shows its real 24-h depth.
            * streamflow: the PEAK INSTANTANEOUS hourly flow per reach over the
              whole period, as a fraction of that reach's 100-yr Q.

The extent is derived from the bridge's own coordinates (--span-mi), not from
the fixed region catalog, so the structure sits in the middle of its frame; its
lon:lat span ratio is the one that fills the panel at this latitude.

Writes  episode/bridge/{bridge}_episode.gif
        episode/bridge/{bridge}_peak_period.{png,pdf}

Usage:
    python episode/e10_bridge_episode.py --bridge 29-00250
    python episode/e10_bridge_episode.py --bridge 29-00250 --fps 8 --span-mi 14
    python episode/e10_bridge_episode.py --bridge 29-00250 --only plate
"""
from __future__ import annotations

import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import basemap  # noqa: E402
from PIL import Image  # noqa: E402

from common import (BASELINE, CLASS_STYLE, DAYS, HAIRLINE, INK, INK2,  # noqa: E402
                    MUTED, PANEL_FIG, PRECIP_ALPHA, SEV_SIZE, SURFACE, TZ,
                    bucket, draw_bridges, draw_counties, draw_county_outline,
                    draw_flowlines, draw_roads, ep_key, hour_range, load_config,
                    load_counties, load_counties_named, load_events,
                    load_flowlines, load_mrms_hour, load_nwm_hour, load_roads,
                    panel_legend_rects, panel_rects, precip_cmap,
                    river_ramp_legend, set_geo)
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e10")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)

CFS = 35.3146667
FAIL_C = "#a8402a"                 # the failure red e09's hydrograph already uses
ACCUM_HOURS = 24                   # rolling window for the plate's rainfall panel
PLATE_DPI = 200                    # the still's render dpi; sets its basemap zoom

# Panel height, shared by BOTH products so the GIF and the plate put the same
# ground in the same place; the plate spends its extra top margin on the two
# keys, the GIF spends its extra bottom margin on the timeline strip.
PANEL_H = 0.590
PANEL_W_IN = 0.445 * 13.333        # one panel's rendered width, inches
# ...and the frame's aspect, derived from that panel: set_geo applies a 1/cos(lat)
# correction, so a frame whose lon span is not ~2.19x its lat span letterboxes
# inside the panel rect and throws the spare width away.
LON_PER_LAT = 2.19

# What the structure carries and crosses. This is in NO table we hold — the
# inventory names neither the route nor the feature intersected (see e07) — so
# it is recorded here from the field report rather than derived. Worth recording
# precisely: 29-00250 carries Hazel Dell Parkway OVER Blue Woods Creek; it does
# not pass beneath the parkway.
CARRIES = {"29-00250": "Hazel Dell Parkway over Blue Woods Creek"}

# The monitored reach is not always the stream under the deck. 29-00250 snaps to
# COMID 18476771, the White River (retro-LP3 Q100 ~ 73,000 cfs), while the
# structure spans Blue Woods Creek, a tributary orders of magnitude smaller. The
# flow panel therefore describes the reach the MONITOR watches, and both products
# say so rather than leaving the reader to assume otherwise.
RIVER_OF = {18476771: "White River"}


# ── extent ───────────────────────────────────────────────────────────────────

def bridge_extent(br: pd.Series, span_mi: float, name: str) -> dict:
    """A frame centred on the bridge, shaped to the panel.

    The lat span comes from --span-mi; the lon span is LON_PER_LAT x that, which
    is what fills the panel once set_geo applies its 1/cos(lat) aspect correction.
    """
    dlat = span_mi / 69.0
    dlon = dlat * LON_PER_LAT
    return dict(lat=(round(float(br["lat"]) - dlat / 2, 4),
                     round(float(br["lat"]) + dlat / 2, 4)),
                lon=(round(float(br["lon"]) - dlon / 2, 4),
                     round(float(br["lon"]) + dlon / 2, 4)),
                name=name, roads=True, county=None, highlight=None)


# ── data over the whole period ───────────────────────────────────────────────

def period_hours(days: list[str]) -> list[pd.Timestamp]:
    return [ts for d in days for ts in hour_range(d)]


def place_of(bridge: str) -> str:
    """'Carmel, Hamilton County' from e07's lookup, or '' if it never ran.

    Neither field is in the inventory; e07 derives them. A missing lookup costs
    the plate a line of context, not the plate.
    """
    from monitor_common.s3io import read_parquet
    try:
        p = read_parquet(bucket(), ep_key("bridge_places.parquet"))
        r = p[p["bridge_id"].astype(str) == bridge].iloc[0]
    except Exception as e:  # noqa: BLE001
        log.warning("bridge_places unavailable (%s) — run e07", e)
        return ""
    bits = [str(r["city"]), f"{r['county']} County"]
    return ", ".join(b for b in bits if b and b != "nan")


def load_mrms(hours, lat, lon, workers: int, pad: float = 0.05):
    """(stack[T, ny, nx] in inches, NaN for absent hours, lats, lons).

    Clipped to the frame on the way in — the stored grid is 415x345 and nothing
    outside the box is ever drawn or accumulated.
    """
    def grab(ts):
        return ts, load_mrms_hour(ts)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = dict(ex.map(grab, hours))

    first = next((v for v in got.values() if v is not None), None)
    if first is None:
        raise SystemExit("no MRMS hours in the episode store for this period")
    _, lats, lons = first
    rs = np.where((lats >= lat[0] - pad) & (lats <= lat[1] + pad))[0]
    cs = np.where((lons >= lon[0] - pad) & (lons <= lon[1] + pad))[0]
    if not rs.size or not cs.size:
        raise SystemExit("the requested extent falls outside the MRMS grid")

    stack = np.full((len(hours), rs.size, cs.size), np.nan, np.float32)
    for i, ts in enumerate(hours):
        v = got.get(ts)
        if v is not None:
            stack[i] = v[0][np.ix_(rs, cs)]
    n = int(sum(got.get(ts) is not None for ts in hours))
    log.info("MRMS: %d/%d hours, %dx%d cells over the frame", n, len(hours),
             rs.size, cs.size)
    return stack, lats[rs], lons[cs]


def peak_accum(stack: np.ndarray, have: np.ndarray, window: int = ACCUM_HOURS):
    """Per-cell maximum over every COMPLETE rolling `window`-hour window.

    A window containing an hour MRMS never published is dropped, not summed with
    that hour treated as zero: a gap would silently understate the depth, and the
    24-h depth is the one number this panel exists to report.
    """
    filled = np.nan_to_num(stack, nan=0.0).astype(np.float64)
    cum = np.concatenate([np.zeros((1, *stack.shape[1:])), np.cumsum(filled, axis=0)])
    best, used = None, 0
    for i in range(len(stack) - window + 1):
        if not have[i:i + window].all():
            continue
        w = cum[i + window] - cum[i]
        best = w if best is None else np.fmax(best, w)
        used += 1
    if best is None:
        log.warning("no complete %d-h window in the period — falling back to the "
                    "sum of the hours that do exist", window)
        return filled.sum(axis=0), 0
    return best, used


def peak_flow(hours, workers: int) -> pd.DataFrame:
    """Per-COMID peak hourly flow (cms) over the period, both NWM products."""
    best, n = None, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(load_nwm_hour, hours):
            if d is None:
                continue
            n += 1
            cur = d[[c for c in ("q_ol_cms", "q_aa_cms") if c in d.columns]]
            best = cur if best is None else np.fmax(best, cur.reindex(best.index))
    log.info("NWM: %d/%d hours went into the period peak", n, len(hours))
    return best if best is not None else pd.DataFrame()


def ratio_series(q_cms: pd.Series, q100: pd.Series) -> pd.Series:
    return (q_cms * CFS) / q100.reindex(q_cms.index)


# ── figure pieces ────────────────────────────────────────────────────────────

def reach_halo(ax, flow, comid, color=FAIL_C, lw=5.0, zorder=2.8) -> None:
    """A casing under ONE reach, so the monitored COMID is identifiable.

    Drawn beneath the coloured network rather than over it: the reach keeps its
    own value colour and the halo only says which line it is. On this bridge the
    halo is the point — the monitored reach is the White River, not the creek the
    structure actually spans.
    """
    if flow is None or getattr(flow, "empty", True) or comid is None:
        return
    segs = [g[["lon", "lat"]].to_numpy()
            for _pid, g in flow[flow["comid"] == comid].groupby("part_id", sort=False)
            if len(g) > 1]
    if not segs:
        return
    # White outer casing first. A bare red halo under a reach the ramp has already
    # painted red — which is exactly what a flooding reach looks like — is
    # invisible; the white edge is what separates the two.
    for c, w, z in ((("white"), lw + 2.4, zorder - 0.1), (color, lw, zorder)):
        ax.add_collection(LineCollection(segs, colors=c, linewidths=w, zorder=z,
                                         capstyle="round", joinstyle="round"))


def subject_state(phase: str, cls: str | None = None) -> dict:
    """Colour for the subject ring, one per phase of its own story.

    quiet -> alerting -> collapsed. The alerting colour is the map_class colour
    the rest of the report already uses for that trigger, so the ring says WHICH
    product fired without a key: this bridge fired on open-loop alone, and its
    ring turns the open-loop orange rather than a generic "alarm" colour.
    """
    if phase == "collapsed":
        return dict(phase=phase, color=FAIL_C)
    if phase == "alert":
        return dict(phase=phase,
                    color=CLASS_STYLE.get(cls, CLASS_STYLE["unknown"])[0])
    return dict(phase=phase, color=INK)


def subject_marker(ax, br, label: str, state: dict, lat, lon,
                   failed_on: str | None = None) -> None:
    """The bridge this figure is about: a ring no alert marker can hide.

    Sized well above the severity markers and drawn at the top of the stack — on
    a frame holding ~90 alerts the subject is otherwise one dot among many and
    the reader has no way to find it.

    The tag carries the bridge ID ONLY. What the structure carries and crosses is
    in the title; repeating it here put a two-line plate across the middle of both
    maps, which is where the subject bridge and its river are.
    """
    c = state["color"]
    x, y = float(br["lon"]), float(br["lat"])

    # Every stroke is cased in white. Over a street basemap the ring has to cross
    # roads, water and labels of every value, and a single-colour ring
    # disappears against whichever of them it happens to land on.
    for s, ec, lw, z in ((760, "white", 6.0, 9.6), (760, c, 2.8, 9.7),
                         (330, "white", 5.0, 9.8), (330, c, 2.6, 9.9)):
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=ec, linewidths=lw,
                   zorder=z)
    ax.scatter([x], [y], s=46, c=c, edgecolors="white", linewidths=1.4, zorder=10)

    # Tag west of the ring, not under it: this reach runs north-south through the
    # bridge, so a tag hung below the marker sits exactly on the halo that says
    # which reach is being monitored. A leader keeps the pairing unambiguous once
    # the tag is clear of the rings.
    tx = x - (lon[1] - lon[0]) * 0.047   # clear of the outer ring
    ax.plot([tx, x], [y, y], color=c, lw=1.6, zorder=9.5,
            path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
    if state["phase"] == "collapsed" and failed_on:
        d = pd.Timestamp(failed_on)      # %-d is glibc-only; build the day by hand
        txt = f"{label}\ncollapsed {d.day} {d:%b}"
    else:
        txt = label
    ax.annotate(txt, xy=(tx, y), ha="right", va="center", fontsize=12.5,
                color=c, fontweight="bold", zorder=11, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=c, lw=1.4))


def corner_note(ax, lines: list[str], color=INK2) -> None:
    """A small plate of numbers INSIDE the panel they describe.

    The alternative is another header row, and at 13.3 in wide the header is
    already carrying a 0.64-canvas title plus two keys. Numbers also read better
    next to the field they came from than in a run-on subtitle.
    """
    # Top-left, not bottom-left: in this frame the alerts run diagonally from the
    # north-east down to the south-west, so the bottom-left corner is full of the
    # markers a note would cover and the top-left is the empty one.
    ax.text(0.015, 0.978, "\n".join(lines), transform=ax.transAxes, fontsize=12,
            color=color, va="top", ha="left", zorder=11, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=HAIRLINE,
                      lw=0.8, alpha=0.93))


def timeline(fig, rect, hours, i, days, failed: str | None) -> None:
    """Where this frame sits in the period: day blocks, cursor, failure tint.

    A 120-frame loop with only a clock in the title gives the viewer no sense of
    how far through the sequence they are or how much is left; the strip makes
    both readable at a glance without spending panel height on it.
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(0, len(hours)); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#efeee8")
    for s in ax.spines.values():
        s.set_color(BASELINE); s.set_linewidth(0.6)

    local = [f"{h.tz_convert(TZ):%Y-%m-%d}" for h in hours]
    for d in days:
        idx = [k for k, t in enumerate(local) if t == d]
        if not idx:
            continue
        a, b = idx[0], idx[-1] + 1
        hot = failed is not None and d == failed
        if hot:
            ax.axvspan(a, b, color=FAIL_C, alpha=0.16, lw=0)
        if a:
            ax.axvline(a, color=BASELINE, lw=0.8)
        ax.text((a + b) / 2, 0.5, f"{pd.Timestamp(d):%a %d %b}", ha="center",
                va="center", fontsize=12, color=FAIL_C if hot else INK2,
                fontweight="bold" if hot else "normal")
    ax.axvspan(0, i + 1, color=INK2, alpha=0.12, lw=0)
    ax.axvline(i + 1, color=INK, lw=1.8)


def finish_panel(ax, extent, counties, roads, cnamed, de, br, label, state,
                 mscale, bm=None, failed_on=None) -> None:
    """Every layer that sits above the field, in the order it must stack.

    With a basemap the county rings come off: the tiles already draw county and
    township lines, and a second set on top of them is noise. The TIGER roads
    stay, because they are drawn ABOVE the rainfall mesh — where accumulation
    covers the frame, the basemap's own roads are under the field and only this
    overlay still says where the interstates run.
    """
    la, lo = extent["lat"], extent["lon"]
    if bm is None:
        draw_counties(ax, counties, lw=0.6, overlay=True)
    if extent.get("roads"):
        draw_roads(ax, roads, lat=la, lon=lo, zorder=2.5,
                   highlight=extent.get("highlight"))
    if extent.get("county"):
        draw_county_outline(ax, cnamed, extent["county"], zorder=7)
    # The fleet's markers are OFF by default. These figures are a post-mortem of
    # one structure, and ninety-odd other alerts in the same frame answer a
    # different question — e04/e05 are where the fleet view lives. --show-alerts
    # brings them back for a frame that is about the cluster rather than the
    # bridge.
    if de is not None:
        draw_bridges(ax, de, mscale)
    subject_marker(ax, br, label, state, la, lo, failed_on)
    set_geo(ax, la, lo)


def ground(ax, counties, bm) -> None:
    """The bottom layer of a panel: basemap tiles, or the plain county fill."""
    if bm is not None:
        basemap.draw(ax, bm, zorder=0)
    else:
        draw_counties(ax, counties, lw=0.5, fc="#f7f6f2")


def attribution(ax, bm) -> None:
    """Tile credit, inside the panel where the tiles are, with a white halo."""
    if bm is None:
        return
    # 12 pt like every other label on the plate, but low-contrast and cased: it
    # is a credit the licence requires, not something the reader is meant to read
    # before the map.
    ax.text(0.995, 0.012, bm[2], transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, color=MUTED, alpha=0.8, zorder=11,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


# ── the animation ────────────────────────────────────────────────────────────

def frame(i, ts, hours, days, extent, mr, mlats, mlons, nwm_h, q100, de, br,
          label, failed_day, vmax_p, vmax_q, dpi, colors, mscale, flow,
          counties, roads, cnamed, note, reach_label, bm, precip_alpha, subj):
    la, lo = extent["lat"], extent["lon"]
    local = ts.tz_convert(TZ)
    failed = failed_day is not None and f"{local:%Y-%m-%d}" >= failed_day

    fig = plt.figure(figsize=PANEL_FIG, facecolor=SURFACE, dpi=dpi)
    # Panels sit higher than e04's so the timeline strip gets a band of its own
    # at the bottom; the legends move up with them.
    axes = [fig.add_axes(r) for r in panel_rects(y=0.200, h=PANEL_H)]
    for ax in axes:
        ground(ax, counties, bm)

    # panel 1 — MRMS 1-h QPE
    ax = axes[0]
    pm = None
    if np.isfinite(mr[i]).any():
        pm = ax.pcolormesh(mlons, mlats, np.ma.masked_less(mr[i], 0.01),
                           cmap=precip_cmap(), vmin=0.01, vmax=vmax_p,
                           shading="nearest", zorder=2, alpha=precip_alpha)
    else:
        ax.text(0.5, 0.5, "no MRMS this hour", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    ax.set_title("MRMS 1-h QPE", fontsize=16, color=INK, loc="left", pad=8)

    # panel 2 — NWM A&A, the same product the daily plates draw
    ax = axes[1]
    nw = nwm_h.get(ts)
    if nw is not None and "q_aa_cms" in nw.columns:
        draw_flowlines(ax, flow, ratio_series(nw["q_aa_cms"], q100), vmax=vmax_q,
                       lw_base=0.75 if bm is not None else 0.55, lat=la, lon=lo)
    else:
        ax.text(0.5, 0.5, "no NWM this hour", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    reach_halo(ax, flow, int(br["comid"]))
    ax.set_title("NWM streamflow", fontsize=16, color=INK, loc="left", pad=8)
    # One line, not the plate's four: in the animation the numbers change every
    # frame, so only the caveat that does NOT change belongs on the map.
    corner_note(ax, [reach_label])

    # The subject's own state this hour: quiet until its trigger fires, then the
    # colour of the product that fired it, then the failure red from the day it
    # collapsed. With the fleet markers off, this ring is the only thing on the
    # map carrying the monitor's verdict, so it has to carry all three.
    alerted = subj is not None and ts >= subj["first_hour"]
    phase = "collapsed" if failed else ("alert" if alerted else "quiet")
    state = subject_state(phase, subj["map_class"] if subj is not None else None)

    # Markers accumulate across the whole PERIOD, not the local day: the point of
    # a five-day loop is watching the alert set build, and resetting it at each
    # midnight would throw away everything the earlier days found.
    shown = de[de["first_hour"] <= ts] if de is not None else None
    for ax in axes:
        finish_panel(ax, extent, counties, roads, cnamed, shown, br, label,
                     state, mscale, bm, failed_day)

    attribution(axes[0], bm)
    # 0.132, not e05's 0.090: the colourbar's LABEL hangs below its tick labels,
    # and at 0.090 it landed on the timeline strip.
    cb_rect, ramp_rect = panel_legend_rects(y=0.132)
    if pm is not None:
        cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
        cb.set_label("1-h QPE (in)", fontsize=13, color=INK2)
        cb.ax.tick_params(labelsize=12, colors=INK2)
    river_ramp_legend(fig, ramp_rect, vmax=vmax_q, label="flow ÷ reach 100-yr Q")
    timeline(fig, [0.030, 0.012, 0.940, 0.030], hours, i, days, failed_day)

    fig.text(0.030, 0.978, f"{local:%A %d %B %Y  ·  %H:%M %Z}", fontsize=20,
             fontweight="bold", color=INK, va="top")
    # What the monitor knew about THIS bridge at this hour — the counter it
    # replaces ("95 bridge(s) triggered so far") described a marker layer that
    # is no longer drawn, and never described the subject anyway.
    #
    # Two texts, not one: the status is the only part that changes colour when
    # the trigger fires, and running it into the same string tinted the bridge
    # name and the failure date orange along with it. Right-aligning it also
    # keeps it on the canvas — appended, it ran off the right edge.
    if alerted:
        rp = int(subj["severity_rp"])
        cls = CLASS_STYLE.get(subj["map_class"], CLASS_STYLE["unknown"])[2]
        first = subj["first_hour"].tz_convert(TZ)
        status = (f"{rp}-yr alert, {cls.replace('Flow — ', '')} · "
                  f"fired {first:%d %b %H:%M}")
    else:
        status = "no alert on this bridge yet"
    fig.text(0.030, 0.930, f"{extent['name']}   ·   {note}", fontsize=13,
             color=INK2, va="top")
    fig.text(0.985, 0.930, status, fontsize=13, va="top", ha="right",
             color=state["color"] if alerted else INK2,
             fontweight="bold" if alerted else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=colors)


# ── the plate ────────────────────────────────────────────────────────────────

def plate(acc, mlats, mlons, ratio, extent, de, br, label, days, flow, counties,
          roads, cnamed, mscale, vmax_q, facts, failed_txt,
          alert_txt, failed_on, bm, precip_alpha):
    la, lo = extent["lat"], extent["lon"]
    fig = plt.figure(figsize=PANEL_FIG, facecolor=SURFACE)
    # Slightly shorter than e05's panels: this plate's title is a bridge name
    # rather than a date, so it needs a row of its own and the keys need theirs.
    axes = [fig.add_axes(r) for r in panel_rects(y=0.155, h=PANEL_H)]
    for ax in axes:
        ground(ax, counties, bm)

    ax = axes[0]
    vmax = max(0.5, float(np.nanpercentile(acc, 99.8)))
    pm = ax.pcolormesh(mlons, mlats, np.ma.masked_less(acc, 0.05),
                       cmap=precip_cmap(), vmin=0.05, vmax=vmax,
                       shading="nearest", zorder=2, alpha=precip_alpha)
    ax.set_title(f"Peak {ACCUM_HOURS}-h MRMS accumulation", fontsize=16,
                 color=INK, loc="left", pad=8)

    ax = axes[1]
    if ratio is None:
        ax.text(0.5, 0.5, "product unavailable", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=15)
    else:
        draw_flowlines(ax, flow, ratio, vmax=vmax_q,
                       lw_base=0.75 if bm is not None else 0.55, lat=la, lon=lo)
    reach_halo(ax, flow, int(br["comid"]))
    ax.set_title("Peak instantaneous NWM streamflow", fontsize=16, color=INK,
                 loc="left", pad=8)

    for ax in axes:
        finish_panel(ax, extent, counties, roads, cnamed, de, br, label,
                     subject_state("collapsed"), mscale, bm, failed_on)
    attribution(axes[0], bm)

    cb_rect, ramp_rect = panel_legend_rects()
    cb = fig.colorbar(pm, cax=fig.add_axes(cb_rect), orientation="horizontal")
    cb.set_label(f"peak {ACCUM_HOURS}-h accumulation (in)", fontsize=13, color=INK2)
    cb.ax.tick_params(labelsize=12, colors=INK2)
    river_ramp_legend(fig, ramp_rect, vmax=vmax_q,
                      label="peak flow ÷ reach 100-yr Q")

    # Header rows, measured rather than guessed. At 13.333 in the title alone is
    # 0.64 of the canvas, so nothing may share row 1.
    #   row 1  title                                    (alone)
    #   row 2  period + place            | class key, only with --show-alerts
    #   row 3  failure date + own alert  | severity key, likewise
    d0, d1 = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
    fig.text(0.030, 0.980, extent["name"], fontsize=20, fontweight="bold",
             color=INK, va="top")
    period = f"Peak of {d0:%d}–{d1:%d %B %Y}"
    fig.text(0.030, 0.934,
             f"{period}   ·   {extent['place']}" if extent.get("place") else period,
             fontsize=13, color=INK2, va="top")
    fig.text(0.030, 0.890, failed_txt, fontsize=12.5, color=FAIL_C, va="top",
             fontweight="bold")
    fig.text(0.030, 0.846, alert_txt, fontsize=12.5, color=INK2, va="top")

    # The period's numbers, right-aligned on the same three rows. They used to
    # sit in boxes inside the panels, which covered the map they described; the
    # header has the room now that the marker keys are gone, and a right column
    # keeps each number on the row of the story line it belongs to.
    for y, txt in zip((0.934, 0.890, 0.846), facts):
        if txt:
            fig.text(0.985, y, txt, fontsize=12, color=INK2, va="top", ha="right")

    if de is None:               # nothing to decode without the fleet markers
        return fig

    handles = []
    for cls, (c, m, lbl) in CLASS_STYLE.items():
        n = int((de["map_class"] == cls).sum())
        if n:
            # Drop the shared "Flow — " prefix: the panel this key decodes is
            # titled "NWM streamflow", and the full labels make the key 0.61 of
            # the canvas, which runs it into the subtitle on the same row.
            handles.append(Line2D([], [], marker=m, linestyle="", markerfacecolor=c,
                                  markeredgecolor="white", markeredgewidth=1.0,
                                  markersize=11,
                                  label=f"{lbl.replace('Flow — ', '')} ({n})"))
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                          markeredgecolor=FAIL_C, markeredgewidth=2.0,
                          markersize=13, label=f"{br['bridge_id']} (failed)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.951),
               ncol=len(handles), frameon=False, fontsize=12.5,
               handletextpad=0.5, columnspacing=1.5)

    sev = [Line2D([], [], marker="o", linestyle="", markerfacecolor=INK2,
                  markeredgecolor="white", markeredgewidth=0.8,
                  markersize=np.sqrt(SEV_SIZE[rp] * mscale) * 0.95,
                  label=f"{rp}-yr ({int((de['severity_rp'] == rp).sum())})")
           for rp in (10, 50, 100)]
    fig.legend(handles=sev, loc="upper right", bbox_to_anchor=(0.985, 0.893),
               ncol=3, frameon=False, fontsize=12.5, handletextpad=0.5,
               columnspacing=1.5, title="severity = size", title_fontsize=12.5,
               alignment="left")
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="29-00250")
    ap.add_argument("--days", nargs="*", default=DAYS, help="local days, in order")
    ap.add_argument("--failed", default="2026-08-15", help="local day it failed")
    ap.add_argument("--carries", default=None,
                    help="what the structure carries/crosses (overrides CARRIES)")
    ap.add_argument("--span-mi", type=float, default=18.0,
                    help="north-south extent of the frame; east-west follows "
                         "from the panel shape (see LON_PER_LAT)")
    ap.add_argument("--fps", type=float, default=6.0,
                    help="faster than the daily GIFs — this one holds five days")
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--colors", type=int, default=192)
    ap.add_argument("--marker-scale", type=float, default=1.2)
    ap.add_argument("--vmax-q", type=float, default=1.5,
                    help="top of the streamflow ramp, shared with the daily plates")
    ap.add_argument("--basemap", default="street",
                    choices=(*basemap.PROVIDERS, "none"),
                    help="navigation tiles under the data; 'none' falls back to "
                         "the plain county fill the daily plates use")
    ap.add_argument("--basemap-fade", type=float, default=0.42,
                    help="blend the tiles toward white (0 = full colour). Some "
                         "fade keeps the map a reference layer rather than a "
                         "competitor to the rainfall field on top of it")
    ap.add_argument("--basemap-zoom", type=int, default=0,
                    help="0 picks the zoom from the rendered panel width")
    ap.add_argument("--precip-alpha", type=float, default=0.0,
                    help="0 = 0.55 over tiles (so the street map reads through "
                         f"the rainfall), {PRECIP_ALPHA} without them")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", choices=("gif", "plate"), help="render just one")
    ap.add_argument("--show-alerts", action="store_true",
                    help="also draw every OTHER alerting bridge in the frame, "
                         "with the class and severity keys. Off by default: "
                         "these figures are about one structure")
    args = ap.parse_args()

    cfg = load_config()
    row = cfg[cfg["bridge_id"].astype(str) == args.bridge]
    if row.empty:
        raise SystemExit(f"{args.bridge} is not in the monitored configuration")
    br = row.iloc[0]
    comid = int(br["comid"])
    carries = args.carries or CARRIES.get(args.bridge)
    label = str(args.bridge)
    name = f"{args.bridge} — {carries}" if carries else f"Bridge {args.bridge}"
    extent = bridge_extent(br, args.span_mi, name)
    extent["place"] = place_of(args.bridge)
    la, lo = extent["lat"], extent["lon"]
    log.info("%s at %.5f, %.5f -> COMID %d; frame lat %.3f..%.3f, lon %.3f..%.3f",
             args.bridge, br["lat"], br["lon"], comid, la[0], la[1], lo[0], lo[1])

    # One fetch per PRODUCT, not one per run, and each at the zoom its own output
    # resolution can show. The plate renders at 200 dpi and the animation at 90,
    # so a single set of tiles either starves the plate or hands the GIF ~3x the
    # pixels it can draw — and downsampling tiles by 3 turns every place name
    # into mush, which is the one thing the basemap is here to provide. Tiles are
    # cached on disk, so the second zoom is a few requests, once, ever.
    def tiles(dpi: int):
        if args.basemap == "none":
            return None
        z = args.basemap_zoom or basemap.zoom_for(lo, int(PANEL_W_IN * dpi))
        try:
            return basemap.fetch(la, lo, zoom=z, provider=args.basemap,
                                 fade=args.basemap_fade, workers=args.workers)
        except Exception as e:  # noqa: BLE001
            log.warning("basemap unavailable (%s) — falling back to county fill", e)
            return None

    precip_alpha = args.precip_alpha or (0.55 if args.basemap != "none"
                                         else PRECIP_ALPHA)

    hours = period_hours(args.days)
    ev = load_events()
    counties, flow, roads = load_counties(), load_flowlines(), load_roads()
    cnamed = load_counties_named()
    q100 = (cfg.dropna(subset=["comid"]).drop_duplicates("comid")
            .set_index("comid")["Q100_cfs"])

    inframe = ev[ev["day"].isin(args.days)
                 & ev["lat"].between(la[0], la[1])
                 & ev["lon"].between(lo[0], lo[1])]
    alerts = (inframe.sort_values("severity_rp", ascending=False)
              .groupby("bridge_id")
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   map_class=("map_class", "first"),
                   severity_rp=("severity_rp", "max"),
                   first_hour=("valid_hour", "min")).reset_index())
    log.info("%d bridge(s) alerting in the frame over %d day(s)%s",
             len(alerts), len(args.days),
             "" if args.show_alerts else " — not drawn (--show-alerts)")
    de = alerts if args.show_alerts else None

    # The subject's OWN alert, which drives the ring colour and the animation's
    # status line whether or not the rest of the fleet is drawn. A bridge that
    # never fired simply has none, and the ring stays quiet for the whole loop.
    own = alerts[alerts["bridge_id"].astype(str) == args.bridge]
    subj = own.iloc[0] if len(own) else None
    if subj is None:
        log.info("%s never alerted in this window", args.bridge)
    else:
        log.info("%s alerted %d-yr (%s) at %s", args.bridge,
                 int(subj["severity_rp"]), subj["map_class"],
                 subj["first_hour"].tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z"))

    mr, mlats, mlons = load_mrms(hours, la, lo, args.workers)
    have = np.isfinite(mr).any(axis=(1, 2))
    acc, nwin = peak_accum(mr, have)
    peak = peak_flow(hours, args.workers)
    ratio = (ratio_series(peak["q_aa_cms"], q100)
             if not peak.empty and "q_aa_cms" in peak.columns else None)

    # The period's numbers, read off the cell and the reach the bridge is on.
    # They are stated for BOTH NWM products, not just the one the panel draws:
    # the open-loop run is what fired the trigger, and on this reach the two
    # disagree by more than a factor of two — which is the finding.
    ri = int(np.abs(mlats - float(br["lat"])).argmin())
    ci = int(np.abs(mlons - float(br["lon"])).argmin())
    rain = float(acc[ri, ci])
    rv = RIVER_OF.get(comid)
    peaks = []
    for col, lbl in (("q_ol_cms", "open-loop"), ("q_aa_cms", "A&A")):
        if peak.empty or col not in peak.columns or comid not in peak.index:
            continue
        v = float(peak.loc[comid, col]) * CFS
        peaks.append(f"{lbl} {v:,.0f} cfs = {v / float(br['Q100_cfs']):.2f}× Q100")
        log.info("  %-9s period peak on COMID %d: %s cfs (%.2f x Q100)",
                 lbl, comid, f"{v:,.0f}", v / float(br["Q100_cfs"]))

    # Three lines for the plate's right-hand header column, one per story row.
    # The reach line names the creek the structure actually spans rather than
    # saying "not the stream under the deck" — e07 gives us the river name, so
    # the caveat can be specific about what the monitor is NOT watching.
    facts = [
        f"Peak {ACCUM_HOURS}-h rain at the bridge {rain:.2f} in   ·   "
        f"frame max {float(np.nanmax(acc)):.2f} in",
        f"Reach peak: {'   ·   '.join(peaks)}" if peaks else "",
        f"Monitored reach: COMID {comid}" + (f" — {rv}" if rv else "")
        + (f", not {carries.split(' over ')[-1]}" if carries and " over " in carries
           else ""),
    ]
    log.info("peak %d-h rain: %.2f in max over the frame, %.2f in at the bridge, "
             "from %d complete windows", ACCUM_HOURS, float(np.nanmax(acc)), rain, nwin)
    if ratio is not None:
        inbox = flow[flow["lat"].between(la[0], la[1])
                     & flow["lon"].between(lo[0], lo[1])]["comid"].unique()
        r = ratio.reindex(inbox).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            log.info("period peak flow ratio in frame: median %.2f, p99 %.2f, "
                     "max %.2f (ramp tops out at %.2f)", r.median(),
                     r.quantile(.99), r.max(), args.vmax_q)

    failed_txt = (f"Bridge failed {pd.Timestamp(args.failed):%A %d %B %Y}"
                  if args.failed else "")
    reach_label = facts[2]

    # Row 4 of the plate: what the monitor said, and when, relative to the
    # failure on row 3. With the fleet markers gone this is the only place the
    # trigger appears on the still, and the two dates together are the finding.
    if subj is not None:
        cls = CLASS_STYLE.get(subj["map_class"], CLASS_STYLE["unknown"])[2]
        first = subj["first_hour"].tz_convert(TZ)
        # Abbreviated weekday/month, unlike the failure row above it: spelled
        # out, this line runs to 0.53 of the canvas and collides with the reach
        # line sharing row 4. Measured, not guessed.
        alert_txt = (f"Monitor alert fired {first:%a %d %b %H:%M %Z} — "
                     f"{int(subj['severity_rp'])}-yr, "
                     f"{cls.replace('Flow — ', '')}")
    else:
        alert_txt = "This bridge never fired an alert in the window"

    if args.only != "plate":
        # Abbreviated: the animation's subtitle also carries the bridge name and
        # the status, and the spelled-out date pushes the three past the canvas.
        gif_failed = (f"failed {pd.Timestamp(args.failed):%a %d %b %Y}"
                      if args.failed else "")
        bm = tiles(args.dpi)
        vmax_p = max(0.15, float(np.nanpercentile(mr[np.isfinite(mr)], 99.9)))
        log.info("rainfall ramp tops at %.2f in/h (99.9th pct over the frame)", vmax_p)
        nwm_h = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ts, d in zip(hours, ex.map(load_nwm_hour, hours)):
                if d is not None:
                    nwm_h[ts] = d
        frames = [frame(i, ts, hours, args.days, extent, mr, mlats, mlons, nwm_h,
                        q100, de, br, label, args.failed, vmax_p, args.vmax_q,
                        args.dpi, args.colors, args.marker_scale, flow, counties,
                        roads, cnamed, gif_failed, reach_label, bm, precip_alpha,
                        subj)
                  for i, ts in enumerate(hours)]
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       loop=0, duration=int(1000 / args.fps), optimize=True)
        key = ep_key(f"bridge/{args.bridge}_episode.gif")
        write_bytes(buf.getvalue(), bucket(), key, content_type="image/gif")
        log.info("wrote s3://%s/%s — %d frames at %.1f fps (%.0f s loop, %.1f MB)",
                 bucket(), key, len(frames), args.fps, len(frames) / args.fps,
                 len(buf.getvalue()) / 1e6)

    if args.only != "gif":
        bm = tiles(PLATE_DPI)
        fig = plate(acc, mlats, mlons, ratio, extent, de, br, label, args.days,
                    flow, counties, roads, cnamed, args.marker_scale, args.vmax_q,
                    facts, failed_txt, alert_txt, args.failed, bm, precip_alpha)
        for ext, ctype in (("png", "image/png"), ("pdf", "application/pdf")):
            buf = io.BytesIO()
            fig.savefig(buf, format=ext, dpi=PLATE_DPI, facecolor=SURFACE)
            key = ep_key(f"bridge/{args.bridge}_peak_period.{ext}")
            write_bytes(buf.getvalue(), bucket(), key, content_type=ctype)
            log.info("wrote s3://%s/%s (%.0f KB)", bucket(), key,
                     len(buf.getvalue()) / 1024)
        plt.close(fig)


if __name__ == "__main__":
    main()
