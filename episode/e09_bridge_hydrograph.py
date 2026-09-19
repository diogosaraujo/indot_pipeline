"""e09 — single-bridge NWM hydrograph with the retro-LP3 return-period tiers.

Built for the post-mortem question "what did the monitor see on this reach while
the bridge was failing?". One panel, one COMID, both NWM products:

  open-loop (analysis_assim_no_da)  the gauge-free run that drives the trigger
  A&A       (analysis_assim)        the data-assimilated run shown alongside it

plus horizontal Q10 / Q50 / Q100 from the retrospective LP3 fit (script 04c),
so the reader can see at a glance how far the modelled flow ever got from the
level that would have fired an alert.

Hours come from the episode store (episode/nwm/{YYYYMMDDHH}.parquet) where it
has them and straight from noaa-nwm-pds where it does not — the store was built
for 12-16 Aug and stops at 2026081703Z, so anything past that is fetched live
for this one COMID rather than by re-running e01 over all 6.9 k of them.

Sized 13.333 x 7.5 in (true 16:9 slide) with 12 pt as the smallest type, so it
drops into the deck at 1:1 and nothing shrinks.

Writes  s3://<bucket>/<prefix>episode/bridge/<bridge>_hydrograph.{png,pdf}

Usage:
    python episode/e09_bridge_hydrograph.py --bridge 29-00250
    python episode/e09_bridge_hydrograph.py --bridge 29-00250 \
        --start 2026-08-12 --end 2026-08-17 --failed 2026-08-15
"""
from __future__ import annotations

import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (HAIRLINE, INK, INK2, MUTED, SURFACE, TIER_C, TZ, bucket,  # noqa: E402
                    ep_key, load_config, load_nwm_hour)
from monitor_common import config, nwm  # noqa: E402
from monitor_common.s3io import write_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("episode.e09")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)

CFS = config.CFS_PER_CMS
OL_C, AA_C = "#14608c", "#7b4ea8"       # same two hues e08 uses for the pair
FAIL_C = "#a8402a"

# Type scale. 12 pt is the floor (tick labels); everything above it is larger.
# See the slide-sizing convention: the canvas is the slide, so points map 1:1.
FS_TICK, FS_AX, FS_LEG, FS_NOTE = 12, 14, 13, 12
FS_SUB, FS_TITLE = 14, 20


# ── series ───────────────────────────────────────────────────────────────────

def utc_hours(start: str, end: str) -> list[pd.Timestamp]:
    """Every UTC hour from local midnight on `start` to local midnight after `end`."""
    t0 = pd.Timestamp(f"{start} 00:00", tz=TZ).tz_convert("UTC")
    t1 = (pd.Timestamp(f"{end} 00:00", tz=TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    return list(pd.date_range(t0, t1, freq="h"))


def from_store(ts: pd.Timestamp, comid: int):
    d = load_nwm_hour(ts)
    if d is None or comid not in d.index:
        return None
    r = d.loc[comid]
    return float(r.get("q_ol_cms", np.nan)), float(r.get("q_aa_cms", np.nan))


def from_noaa(ts: pd.Timestamp, comid: int):
    """Live read of the two channel_rt products for a single reach."""
    ids = np.asarray([comid], dtype=np.int64)
    out = []
    for product in (config.NWM_PRODUCT_TRIGGER, config.NWM_PRODUCT_DISPLAY):
        d = nwm.read_comids(ts, product, ids)
        out.append(np.nan if d is None else float(d["streamflow_cms"].iloc[0]))
    if not np.isfinite(out).any():
        return None
    return out[0], out[1]


def series(comid: int, hours: list[pd.Timestamp], workers: int = 8) -> pd.DataFrame:
    """Open-loop and A&A streamflow (cfs) per hour, store first then NOAA.

    Missing hours stay missing — a gap in the line is the honest rendering of an
    hour NWM never published, and interpolating across it would invent a value
    the trigger never saw.
    """
    rows: dict[pd.Timestamp, tuple] = {}

    def grab(ts, fn):
        return ts, fn(ts, comid)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(grab, t, from_store) for t in hours]):
            ts, got = f.result()
            if got is not None:
                rows[ts] = got
    n_store = len(rows)
    missing = [t for t in hours if t not in rows]
    log.info("episode store supplied %d/%d hours; fetching %d from %s",
             n_store, len(hours), len(missing), config.NWM_BUCKET)

    if missing:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(grab, t, from_noaa) for t in missing]):
                ts, got = f.result()
                if got is not None:
                    rows[ts] = got

    df = pd.DataFrame([(t, *v) for t, v in sorted(rows.items())],
                      columns=["utc", "ol_cms", "aa_cms"]).set_index("utc")
    log.info("assembled %d/%d hours (%d from store, %d live, %d unavailable)",
             len(df), len(hours), n_store, len(df) - n_store, len(hours) - len(df))
    return (df * CFS).rename(columns={"ol_cms": "ol", "aa_cms": "aa"})


# ── figure ───────────────────────────────────────────────────────────────────

def draw(br: pd.Series, q: pd.DataFrame, failed: str | None, hours) -> plt.Figure:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)
    ax = fig.add_axes([0.075, 0.115, 0.845, 0.645])
    ax.set_facecolor(SURFACE)

    t0 = hours[0].tz_convert(TZ)
    t1 = hours[-1].tz_convert(TZ)
    t = q.index.tz_convert(TZ)

    if failed:
        d0 = pd.Timestamp(f"{failed} 00:00", tz=TZ)
        ax.axvspan(d0, d0 + pd.Timedelta(days=1), color=FAIL_C, alpha=0.065,
                   zorder=1, lw=0)
        ax.annotate(f"bridge failed\n{d0:%a %d %b}", xy=(d0 + pd.Timedelta(hours=12), 0.965),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=FS_NOTE, color=FAIL_C, fontweight="bold")

    ax.plot(t, q["ol"], color=OL_C, lw=2.6, zorder=5,
            label="NWM open-loop  (analysis_assim_no_da — drives the trigger)")
    ax.plot(t, q["aa"], color=AA_C, lw=2.2, ls="-.", zorder=5,
            label="NWM A&A  (analysis_assim — with data assimilation)")

    # Return-period tiers. All three are drawn even where the flow never comes
    # near them: the distance between the trace and the lowest line IS the
    # finding on this reach, so cropping the axis to the data would erase it.
    for rp in (10, 50, 100):
        v = float(br[f"Q{rp}_cfs"])
        if not np.isfinite(v):
            continue
        ax.axhline(v, color=TIER_C[rp], ls="--", lw=2.0, zorder=3)
        ax.annotate(f"Q{rp} = {v:,.0f} cfs", xy=(0.997, v),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=FS_NOTE, color=TIER_C[rp], fontweight="bold")

    for col, c, lbl in (("ol", OL_C, "open-loop"), ("aa", AA_C, "A&A")):
        s = q[col].dropna()
        if s.empty:
            continue
        pk = s.idxmax().tz_convert(TZ)
        ax.plot([pk], [s.max()], marker="o", ms=8, mfc="white", mec=c, mew=2.2, zorder=6)
        ax.annotate(f"{lbl} peak {s.max():,.0f} cfs\n{pk:%a %d %b %H:%M} ET",
                    xy=(pk, s.max()), xytext=(14, 18), textcoords="offset points",
                    fontsize=FS_NOTE, color=INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=HAIRLINE))

    ax.set_xlim(t0, t1)
    ax.set_ylim(0, max(float(br["Q100_cfs"]) * 1.14, q.max().max() * 1.25))
    ax.set_ylabel("Streamflow (cfs)", fontsize=FS_AX, color=INK)
    ax.set_xlabel(f"Local time ({t0:%Z})", fontsize=FS_AX, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=t0.tz))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6, tz=t0.tz))
    ax.tick_params(labelsize=FS_TICK, colors=INK2)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.grid(axis="x", ls=":", alpha=0.25)
    for s in ax.spines.values():
        s.set_color(HAIRLINE)

    # Legend lives ABOVE the axes, not in a corner: on a 13.3 in canvas a
    # two-line 13 pt box is wide enough to sit on the open-loop peak no matter
    # which corner it is parked in.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
              fontsize=FS_LEG, handlelength=2.6, columnspacing=2.4)

    fig.text(0.075, 0.965,
             f"Bridge {br['bridge_id']} — NWM streamflow on COMID {int(br['comid'])}",
             fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
    fig.text(0.075, 0.905,
             f"{t0:%d %B} – {t1:%d %B %Y}  ·  {br['lat']:.5f}, {br['lon']:.5f}  ·  "
             f"fires at Q{config.FIRE_RP_SCOUR if br['scour_critical'] else config.FIRE_RP_OTHER}"
             f"  ·  {'scour-critical' if br['scour_critical'] else 'not scour-critical'}",
             fontsize=FS_SUB, color=INK2, va="top")
    fig.text(0.075, 0.855,
             "Q10 / Q50 / Q100 are the retrospective LP3 quantiles fitted on this "
             "reach (04c), not on the bridge's own stream.",
             fontsize=FS_NOTE, color=MUTED, va="top")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="29-00250")
    ap.add_argument("--start", default="2026-08-12")
    ap.add_argument("--end", default="2026-08-17", help="last LOCAL day, inclusive")
    ap.add_argument("--failed", default="2026-08-15", help="local day the bridge failed")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    row = cfg[cfg["bridge_id"].astype(str) == args.bridge]
    if row.empty:
        raise SystemExit(f"{args.bridge} is not in the monitored configuration")
    br = row.iloc[0]
    comid = int(br["comid"])
    log.info("%s -> COMID %d at %.5f, %.5f", args.bridge, comid, br["lat"], br["lon"])
    log.info("  retro-LP3  Q10 %s  Q50 %s  Q100 %s cfs",
             *(f"{br[f'Q{rp}_cfs']:,.0f}" for rp in (10, 50, 100)))

    hours = utc_hours(args.start, args.end)
    q = series(comid, hours, args.workers)
    if q.empty:
        raise SystemExit("no NWM hours available for this reach in the window")
    for col, lbl in (("ol", "open-loop"), ("aa", "A&A")):
        s = q[col].dropna()
        if s.empty:
            continue
        log.info("  %-9s peak %9s cfs at %s  = %.0f%% of Q10",
                 lbl, f"{s.max():,.0f}",
                 s.idxmax().tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z"),
                 100 * s.max() / float(br["Q10_cfs"]))

    fig = draw(br, q, args.failed, hours)
    for ext, ctype in (("png", "image/png"), ("pdf", "application/pdf")):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=200, facecolor=SURFACE)
        key = ep_key(f"bridge/{args.bridge}_hydrograph.{ext}")
        write_bytes(buf.getvalue(), bucket(), key, content_type=ctype)
        log.info("wrote s3://%s/%s (%.0f KB)", bucket(), key, len(buf.getvalue()) / 1024)
    plt.close(fig)


if __name__ == "__main__":
    main()
