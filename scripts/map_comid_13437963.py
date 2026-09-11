#!/usr/bin/env python3
"""Map the 12 permanently-firing bridges against the reach they were assigned to.

The defect is spatial, so it should be looked at spatially: twelve bridges spread
over ~25 km were all assigned to COMID 13437963, a single 3.61 km reach, and the
flow trigger judges every one of them against that reach's streamflow.

Draws, on the monitor's own flowline/county assets:
  * the assigned reach, thick, in the "problem" hue
  * the reach that is actually NEAREST to each bridge, where that differs
  * a dashed leader from every bridge to its assigned reach, so the snap
    distance is a length on the page rather than a number in a table
  * unit Q100 per reach (cfs/mi2), which is what exposes the bad LP3 fit

Usage:  python scripts/map_comid_13437963.py [--out results/comid_13437963_map.png]
"""
from __future__ import annotations

import argparse
import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import boto3  # noqa: E402

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
TARGET = 13437963
R_KM = 6371.0

# Validated light categorical steps: orange = the reach at fault, blue = the
# reach that should have been chosen. Text is darkened separately for contrast.
C_BAD, C_ALT = "#eb6834", "#2a78d6"
T_BAD, T_ALT = "#a3421a", "#1c5cab"
INK, INK2, MUTED = "#14181f", "#4d5866", "#8b8a80"
QUIET, COUNTY = "#c9ccc6", "#9aa0a6"


def s3_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}{key}")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def km(alat, alon, blat, blon):
    return np.hypot(np.radians(blat - alat),
                    np.radians(blon - alon) * np.cos(np.radians(alat))) * R_KM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/comid_13437963_map.png")
    ap.add_argument("--pad", type=float, default=0.055, help="degrees of margin")
    args = ap.parse_args()

    cfg = s3_parquet("monitor/bridge_monitor_config.parquet")
    fl = s3_parquet("monitor/assets/flowlines.parquet")
    counties = s3_parquet("monitor/assets/in_counties.parquet")
    lp3 = s3_parquet("monitor/precompute/bridge_comid_lp3.parquet").set_index("comid")
    area = s3_parquet("monitor/precompute/bridge_watershed_area.parquet").set_index("comid")["area_mi2"]

    b = cfg[pd.to_numeric(cfg["comid"], errors="coerce") == TARGET].copy()
    seg = fl[fl["comid"] == TARGET]
    print(f"{len(b)} bridges on COMID {TARGET}; reach has {len(seg)} vertices")

    # provisional extent, to find the alternatives; final extent set after
    lat0 = min(b.lat.min(), seg.lat.min()) - args.pad
    lat1 = max(b.lat.max(), seg.lat.max()) + args.pad
    lon0 = min(b.lon.min(), seg.lon.min()) - args.pad
    lon1 = max(b.lon.max(), seg.lon.max()) + args.pad
    near = fl[(fl.lat.between(lat0 - .3, lat1 + .3)) & (fl.lon.between(lon0 - .3, lon1 + .3))]

    # nearest reach per bridge
    rows = []
    for _, r in b.iterrows():
        d_assigned = km(r.lat, r.lon, seg.lat.to_numpy(), seg.lon.to_numpy()).min()
        d = km(r.lat, r.lon, near.lat.to_numpy(), near.lon.to_numpy())
        i = int(np.argmin(d))
        rows.append({"bridge_id": r.bridge_id, "lat": r.lat, "lon": r.lon,
                     "d_assigned": d_assigned, "nearest": int(near.comid.iloc[i]),
                     "d_nearest": d[i]})
    bb = pd.DataFrame(rows).sort_values("d_assigned")
    alt = sorted(set(bb.nearest) - {TARGET})
    print("alternative nearest reaches:", alt)

    # An alternative reach drawn off-frame argues nothing, so grow the extent to
    # hold every reach the map references, then re-pad.
    ref = fl[fl.comid.isin([TARGET, *alt])]
    lat0 = min(b.lat.min(), ref.lat.min()) - args.pad
    lat1 = max(b.lat.max(), ref.lat.max()) + args.pad
    lon0 = min(b.lon.min(), ref.lon.min()) - args.pad
    lon1 = max(b.lon.max(), ref.lon.max()) + args.pad
    near = fl[(fl.lat.between(lat0, lat1)) & (fl.lon.between(lon0, lon1))]

    bb = bb.reset_index(drop=True)
    bb["n"] = bb.index + 1

    fig = plt.figure(figsize=(15.0, 10.5), facecolor="white")
    ax = fig.add_axes([0.035, 0.065, 0.655, 0.855])

    for _, ring in counties.groupby("part_id"):
        ax.fill(ring.lon, ring.lat, facecolor="#f6f6f3", edgecolor=COUNTY,
                linewidth=0.9, zorder=1)

    for (cid, _pid), g in near.groupby(["comid", "part_id"], sort=False):
        if cid == TARGET or cid in alt:
            continue
        ax.plot(g.lon, g.lat, color=QUIET, lw=0.7, zorder=2, solid_capstyle="round")

    for cid in alt:
        for _pid, g in fl[fl.comid == cid].groupby("part_id"):
            ax.plot(g.lon, g.lat, color=C_ALT, lw=3.2, zorder=4, solid_capstyle="round")

    for _pid, g in seg.groupby("part_id"):
        ax.plot(g.lon, g.lat, color=C_BAD, lw=5.5, zorder=5, solid_capstyle="round")

    # leaders: bridge -> closest point on its ASSIGNED reach. The fan of these
    # converging on one short segment is the whole argument.
    for _, r in bb.iterrows():
        d = km(r.lat, r.lon, seg.lat.to_numpy(), seg.lon.to_numpy())
        j = int(np.argmin(d))
        ax.plot([r.lon, seg.lon.iloc[j]], [r.lat, seg.lat.iloc[j]],
                color=C_BAD, lw=0.9, ls=(0, (4, 3)), alpha=.7, zorder=3)

    # Numbered badges, keyed in the panel. Twelve ID+distance labels collided
    # badly enough to be unreadable; a number is one glyph and never overlaps.
    ax.scatter(bb.lon, bb.lat, s=310, c="white", marker="o",
               edgecolors=INK, linewidths=1.7, zorder=8)
    for _, r in bb.iterrows():
        ax.text(r.lon, r.lat, str(r.n), fontsize=9.2, fontweight="bold",
                color=INK, ha="center", va="center", zorder=9)

    def label_reach(cid, color, dy):
        g = fl[fl.comid == cid]
        a = area.get(cid, np.nan)
        q = lp3.loc[cid]["Q100_cfs"] if cid in lp3.index else np.nan
        u = q / a if a == a and a > 0 else np.nan
        mid = len(g) // 2
        x = float(np.clip(g.lon.iloc[mid], lon0 + (lon1 - lon0) * .17,
                          lon1 - (lon1 - lon0) * .17))
        y = float(np.clip(g.lat.iloc[mid] + dy, lat0 + (lat1 - lat0) * .04,
                          lat1 - (lat1 - lat0) * .04))
        ax.annotate(f"COMID {cid}   {a:,.0f} mi²   Q100 {q:,.0f} cfs   {u:,.1f} cfs/mi²",
                    (g.lon.iloc[mid], g.lat.iloc[mid]), (x, y),
                    fontsize=8.0, color=color, weight="bold", ha="center", zorder=10,
                    arrowprops=dict(arrowstyle="-", color=color, lw=.8, alpha=.6),
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, lw=1.0, alpha=.96))

    label_reach(TARGET, T_BAD, 0.030)
    for i, cid in enumerate(alt):
        label_reach(cid, T_ALT, -0.026 if i % 2 == 0 else 0.024)

    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / np.cos(np.radians((lat0 + lat1) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")

    mid_lat = (lat0 + lat1) / 2
    dlon5 = 5.0 / (R_KM * np.radians(1.0) * np.cos(np.radians(mid_lat)))
    x0, y0 = lon0 + (lon1 - lon0) * .04, lat0 + (lat1 - lat0) * .035
    ax.plot([x0, x0 + dlon5], [y0, y0], color=INK, lw=2.6, zorder=11)
    ax.text(x0 + dlon5 / 2, y0 + (lat1 - lat0) * .011, "5 km", ha="center",
            fontsize=8.5, color=INK, zorder=11)

    fig.text(0.035, 0.968, f"Twelve bridges, one reach — COMID {TARGET}",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.938,
             "Every bridge is judged by the streamflow on the orange reach, "
             "whatever its distance from it.", fontsize=10.5, color=INK2, va="top")

    # ── key ──────────────────────────────────────────────────────────────────
    px = 0.706
    fig.text(px, 0.918, "BRIDGE", fontsize=8, color=MUTED, va="top", family="monospace")
    fig.text(px + 0.185, 0.918, "TO ASSIGNED", fontsize=8, color=MUTED, va="top",
             family="monospace", ha="right")
    fig.text(px + 0.29, 0.918, "NEAREST", fontsize=8, color=MUTED, va="top",
             family="monospace", ha="right")
    y = 0.894
    for _, r in bb.iterrows():
        far = r.d_assigned > 1.0
        fig.text(px, y, f"{r.n:>2}  {r.bridge_id}", fontsize=8.6,
                 color=INK if far else T_ALT, va="top", family="monospace")
        fig.text(px + 0.185, y, f"{r.d_assigned:5.1f} km", fontsize=8.6, ha="right",
                 color=T_BAD if far else INK2, va="top", family="monospace")
        nearer = r.nearest != TARGET
        fig.text(px + 0.29, y, f"{r.nearest}" if nearer else "same", fontsize=8.6,
                 ha="right", color=T_ALT if nearer else MUTED, va="top", family="monospace")
        y -= 0.0208

    y -= 0.012
    for t, c in [
        (f"COMID {TARGET} drains 1,821 mi² but carries a", INK),
        ("Q100 of just 634 cfs — 0.35 cfs/mi², the", T_BAD),
        ("0.3rd percentile of the network (median 193).", T_BAD),
        ("Its 141 area-peers median 48,441 cfs, so this", INK),
        ("reach is 76× low and an ordinary flow reads as", INK),
        ("a 100-year event — every hour since 18 Aug.", INK),
        ("", INK),
        ("Only bridge 1 is within 300 m. Eight have a", INK),
        ("nearer reach (blue) with plausible thresholds;", T_ALT),
        ("four are genuinely nearest to the broken one.", INK),
    ]:
        fig.text(px, y, t, fontsize=8.9, color=c, va="top")
        y -= 0.0198

    handles = [
        Line2D([], [], color=C_BAD, lw=5, label=f"assigned reach ({TARGET})"),
        Line2D([], [], color=C_ALT, lw=3, label="nearer reach for a bridge"),
        Line2D([], [], color=QUIET, lw=1.2, label="other NWM reaches"),
        Line2D([], [], color=C_BAD, lw=1, ls=(0, (4, 3)), label="bridge → assigned reach"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(px - 0.006, y + 0.004),
               frameon=False, fontsize=9.0, labelspacing=0.75)

    fig.text(0.035, 0.026,
             "Reach and county geometry: monitor/assets (NWM flowlines, Census TIGER). "
             "Thresholds: retro-LP3 (p03). Drainage area: NLDI upstream basin (p06).",
             fontsize=7.8, color=MUTED)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(bb[["n", "bridge_id", "d_assigned", "nearest", "d_nearest"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
