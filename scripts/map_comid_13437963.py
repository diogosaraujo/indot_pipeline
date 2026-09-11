#!/usr/bin/env python3
"""Map the 12 permanently-firing bridges on COMID 13437963, over satellite imagery.

The defect is spatial and the imagery is the point: on the aerial you can see the
ditched farmland these bridges actually cross, and see that the NWM flowline
network does not represent it. Twelve bridges spread over ~25 km were all
assigned to one 3.6 km reach, and the flow trigger judges every one of them
against that reach's streamflow.

Basemap tiles are Esri World Imagery (no key required); everything is drawn in
Web Mercator so the vector overlay registers against the tiles. Strokes and text
carry dark casings/halos, without which bright lines vanish over pale fields and
dark lines vanish over water.

Usage:
    python scripts/map_comid_13437963.py
    python scripts/map_comid_13437963.py --comid 13437963 --zoom 14
"""
from __future__ import annotations

import argparse
import io
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

import boto3  # noqa: E402

BUCKET, PREFIX = "indot-bridge-pipeline", "v1/"
R_M = 6378137.0
TILE = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ATTRIB = "Imagery (c) Esri, Maxar, Earthstar Geographics, and the GIS User Community"
CACHE = pathlib.Path(".tilecache")

# Overlay hues chosen for a dark, noisy ground rather than for paper.
C_BAD = "#ff6b35"      # the reach at fault
C_LEAD = "#ffd166"     # bridge -> assigned reach
C_NET = "#7fd4ff"      # the rest of the NWM network
INK, PAPER, MUTED = "#12151a", "#ffffff", "#b9c2cc"


def s3_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}{key}")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


# -- Web Mercator ------------------------------------------------------------
def merc(lon, lat):
    x = R_M * np.radians(np.asarray(lon, float))
    y = R_M * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2))
    return x, y


def deg2tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_bounds_merc(x, y, z):
    world = 2 * math.pi * R_M
    s = world / (2 ** z)
    return (-world / 2 + x * s, world / 2 - (y + 1) * s,
            -world / 2 + (x + 1) * s, world / 2 - y * s)


def fetch_basemap(lon0, lat0, lon1, lat1, z):
    """Stitched Esri imagery covering the box; returns (RGB array, merc extent)."""
    CACHE.mkdir(exist_ok=True)
    x0, y1 = deg2tile(lon0, lat0, z)
    x1, y0 = deg2tile(lon1, lat1, z)
    xs = list(range(int(math.floor(x0)), int(math.floor(x1)) + 1))
    ys = list(range(int(math.floor(y0)), int(math.floor(y1)) + 1))
    keys = [(x, y) for y in ys for x in xs]
    print(f"  basemap: zoom {z}, {len(xs)}x{len(ys)} = {len(keys)} tiles")

    def one(k):
        x, y = k
        f = CACHE / f"{z}_{y}_{x}.jpg"
        if not f.exists():
            r = requests.get(TILE.format(z=z, y=y, x=x), timeout=40,
                             headers={"User-Agent": "indot-bridge-pipeline/1.0"})
            r.raise_for_status()
            f.write_bytes(r.content)
        return k, Image.open(f).convert("RGB")

    with ThreadPoolExecutor(max_workers=12) as ex:
        tiles = dict(ex.map(one, keys))

    canvas = Image.new("RGB", (256 * len(xs), 256 * len(ys)))
    for (x, y), im in tiles.items():
        canvas.paste(im, ((x - xs[0]) * 256, (y - ys[0]) * 256))
    ext = (tile_bounds_merc(xs[0], ys[0], z)[0], tile_bounds_merc(xs[-1], ys[-1], z)[2],
           tile_bounds_merc(xs[0], ys[-1], z)[1], tile_bounds_merc(xs[0], ys[0], z)[3])
    return np.asarray(canvas), ext


def km(alat, alon, blat, blon):
    return np.hypot(np.radians(blat - alat),
                    np.radians(blon - alon) * np.cos(np.radians(alat))) * 6371.0


def halo(lw=2.6, fg=INK):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comid", type=int, default=13437963)
    ap.add_argument("--out", default="results/comid_13437963_satellite.png")
    ap.add_argument("--zoom", type=int, default=14)
    ap.add_argument("--pad", type=float, default=0.022, help="degrees of margin")
    args = ap.parse_args()
    TARGET = args.comid

    cfg = s3_parquet("monitor/bridge_monitor_config.parquet")
    fl = s3_parquet("monitor/assets/flowlines.parquet")
    lp3 = s3_parquet("monitor/precompute/bridge_comid_lp3.parquet").set_index("comid")
    area = s3_parquet("monitor/precompute/bridge_watershed_area.parquet").set_index("comid")["area_mi2"]

    b = cfg[pd.to_numeric(cfg["comid"], errors="coerce") == TARGET].copy()
    seg = fl[fl["comid"] == TARGET]
    print(f"{len(b)} bridges on COMID {TARGET}; reach has {len(seg)} vertices")

    # Focus: only the offending reach and the bridges that answer to it.
    lat0 = min(b.lat.min(), seg.lat.min()) - args.pad
    lat1 = max(b.lat.max(), seg.lat.max()) + args.pad
    lon0 = min(b.lon.min(), seg.lon.min()) - args.pad * 1.25
    lon1 = max(b.lon.max(), seg.lon.max()) + args.pad * 1.25

    rows = []
    for _, r in b.iterrows():
        d = km(r.lat, r.lon, seg.lat.to_numpy(), seg.lon.to_numpy())
        rows.append({"bridge_id": r.bridge_id, "lat": r.lat, "lon": r.lon,
                     "d": d.min(), "j": int(np.argmin(d))})
    bb = pd.DataFrame(rows).sort_values("d").reset_index(drop=True)
    bb["n"] = bb.index + 1

    img, ext = fetch_basemap(lon0, lat0, lon1, lat1, args.zoom)

    xmin, ymin = merc(lon0, lat0)
    xmax, ymax = merc(lon1, lat1)

    # Size the canvas from the DATA aspect. An equal-aspect image in a box of the
    # wrong shape just shrinks and leaves a band of dead figure, which on a dark
    # ground looks like a rendering fault rather than a margin.
    AX_W, AX_L, AX_B, AX_T = 0.655, 0.028, 0.075, 0.862   # top leaves the title room
    FIG_W = 16.0
    aspect = (xmax - xmin) / (ymax - ymin)
    ax_h_frac = AX_T - AX_B
    fig_h = (FIG_W * AX_W / aspect) / ax_h_frac
    fig_h = float(np.clip(fig_h, 6.0, 13.0))
    print(f"  extent aspect {aspect:.2f} -> figure {FIG_W:.1f} x {fig_h:.1f} in")

    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="#0e1116")
    ax = fig.add_axes([AX_L, AX_B, AX_W, ax_h_frac])
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", zorder=0)

    # the wider NWM network, thin -- the contrast with the visible ditches is
    # itself the finding
    near = fl[(fl.lat.between(lat0, lat1)) & (fl.lon.between(lon0, lon1))]
    for (cid, _p), g in near.groupby(["comid", "part_id"], sort=False):
        if cid == TARGET:
            continue
        gx, gy = merc(g.lon.to_numpy(), g.lat.to_numpy())
        ax.plot(gx, gy, color=C_NET, lw=1.0, alpha=.7, zorder=2, solid_capstyle="round")

    for _, r in bb.iterrows():
        lx, ly = merc([r.lon, seg.lon.iloc[int(r.j)]], [r.lat, seg.lat.iloc[int(r.j)]])
        ax.plot(lx, ly, color=C_LEAD, lw=1.5, ls=(0, (5, 3)), alpha=.95, zorder=4,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="#000000")])

    for _p, g in seg.groupby("part_id"):
        gx, gy = merc(g.lon.to_numpy(), g.lat.to_numpy())
        ax.plot(gx, gy, color=INK, lw=9.5, zorder=5, solid_capstyle="round")
        ax.plot(gx, gy, color=C_BAD, lw=5.5, zorder=6, solid_capstyle="round")

    bx, by = merc(bb.lon.to_numpy(), bb.lat.to_numpy())
    ax.scatter(bx, by, s=330, c=PAPER, marker="o", edgecolors=INK,
               linewidths=2.0, zorder=8)
    for i, r in bb.iterrows():
        ax.text(bx[i], by[i], str(r.n), fontsize=9.6, fontweight="bold",
                color=INK, ha="center", va="center", zorder=9)

    a = float(area.get(TARGET, np.nan))
    q = float(lp3.loc[TARGET]["Q100_cfs"]) if TARGET in lp3.index else float("nan")
    mid = len(seg) // 2
    mx, my = merc(seg.lon.iloc[mid], seg.lat.iloc[mid])
    ax.annotate(f"COMID {TARGET}\n{a:,.0f} mi2   Q100 {q:,.0f} cfs   {q/a:,.2f} cfs/mi2",
                (float(mx), float(my)), (float(mx), float(my) + (ymax - ymin) * 0.11),
                fontsize=9.4, color=PAPER, weight="bold", ha="center", zorder=11,
                arrowprops=dict(arrowstyle="-", color=C_BAD, lw=1.6),
                bbox=dict(boxstyle="round,pad=0.38", fc="#12151a", ec=C_BAD, lw=1.4, alpha=.92))

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2c333d")

    # scale bar (mercator metres are inflated by sec(lat))
    midlat = (lat0 + lat1) / 2
    bar_m = 5000 / math.cos(math.radians(midlat))
    sx, sy = xmin + (xmax - xmin) * .04, ymin + (ymax - ymin) * .055
    ax.plot([sx, sx + bar_m], [sy, sy], color=PAPER, lw=3.4, zorder=12,
            path_effects=halo(5.5), solid_capstyle="butt")
    ax.text(sx + bar_m / 2, sy + (ymax - ymin) * .016, "5 km", ha="center",
            fontsize=9, color=PAPER, zorder=12, path_effects=halo())
    ax.text(xmax - (xmax - xmin) * .006, ymin + (ymax - ymin) * .012, ATTRIB,
            fontsize=6.6, color="#dfe5ec", ha="right", zorder=12, path_effects=halo(2.0))

    # -- panel ---------------------------------------------------------------
    fig.text(0.028, 0.978, f"Twelve bridges answer to COMID {TARGET}",
             fontsize=21, fontweight="bold", color=PAPER, va="top")
    fig.text(0.028, 0.930,
             "Each dashed leader runs from a bridge to the reach whose streamflow "
             "decides its alarm.", fontsize=11, color=MUTED, va="top")

    px = 0.700
    rs = 8.0 / fig_h          # row pitch, so the key fits any canvas
    fig.text(px, 0.930, "BRIDGE", fontsize=8, color=MUTED, va="top", family="monospace")
    fig.text(px + 0.275, 0.930, "TO REACH", fontsize=8, color=MUTED, va="top",
             family="monospace", ha="right")
    y = 0.905
    for _, r in bb.iterrows():
        far = r.d > 1.0
        fig.text(px, y, f"{r.n:>2}  {r.bridge_id}", fontsize=9.2,
                 color=PAPER if far else "#9fe8c0", va="top", family="monospace")
        fig.text(px + 0.275, y, f"{r.d:5.1f} km", fontsize=9.2, ha="right",
                 color=C_BAD if far else MUTED, va="top", family="monospace")
        y -= 0.0225 * rs
    y -= 0.016 * rs
    for t, c in [
        (f"{a:,.0f} mi2 of drainage, but a Q100 of only", PAPER),
        (f"{q:,.0f} cfs = {q/a:,.2f} cfs/mi2, the 0.3rd", C_BAD),
        ("percentile of the network (median 193).", C_BAD),
        ("Its 141 area-peers median 48,441 cfs, so an", PAPER),
        ("ordinary flow reads as a 100-year event -", PAPER),
        ("every hour since 18 Aug.", PAPER),
        ("", PAPER),
        ("Only bridge 1 is within 300 m. On the imagery", PAPER),
        ("the ditches these bridges actually cross are", C_NET),
        ("visible, and absent from the NWM network.", C_NET),
    ]:
        fig.text(px, y, t, fontsize=9.3, color=c, va="top")
        y -= 0.0212 * rs

    handles = [
        Line2D([], [], color=C_BAD, lw=5, label=f"assigned reach ({TARGET})"),
        Line2D([], [], color=C_NET, lw=1.6, label="other NWM reaches"),
        Line2D([], [], color=C_LEAD, lw=1.6, ls=(0, (5, 3)), label="bridge to assigned reach"),
        Line2D([], [], color=PAPER, marker="o", ls="none", ms=9,
               markeredgecolor=INK, label="bridge (12)"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(px - 0.006, y),
                     frameon=False, fontsize=9.4, labelspacing=0.8)
    for t in leg.get_texts():
        t.set_color(MUTED)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(bb[["n", "bridge_id", "d"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
