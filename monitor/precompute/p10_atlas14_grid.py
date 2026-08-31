"""p10 — gridded Atlas-14 24-h depths, co-registered to the monitor's MRMS grid.

The daily summary shows a 24-h MRMS accumulation next to the ARI that
accumulation represents. Turning depth into ARI needs an Atlas-14 depth curve at
EVERY grid cell, and p02 only has curves at the ~16k bridge cells (110 m dedupe)
— dense along waterways, empty elsewhere. Interpolating those would invent a
surface in exactly the places no bridge sits.

So take the published surface instead. NOAA Atlas 14 ships 30 arc-second ASCII
grids per duration and ARI. Indiana sits in VOLUME 2, "Ohio River Basin and
Surrounding States" (directory `orb`) — NOT Volume 8 "Midwestern States"
(`mw`), whose name is misleading: mw covers CO/KS/NE/MO and its data ends near
91 W, well west of the state. Sampling mw over Indiana silently returns NODATA
almost everywhere and plausible-looking values along the western edge, which is
why the cross-check below is not optional.

These grids are STATIC — Atlas 14 is a fixed publication, not a rolling
product, so this runs once and never again. (Its successor, NOAA Atlas 15, will
add nonstationarity; adopting it would be a deliberate re-run, not silent drift.)

Grids are sampled NEAREST onto the exact axes of a stored MRMS state grid rather
than onto a constructed axis, so the ARI raster and the accumulation raster are
cell-for-cell aligned by construction and no render-time regridding is needed.
Nearest also matches how the point trigger reads Atlas-14 (mrms.sample_from_grid),
so the map and the alarm cannot disagree about which cell a bridge is in.

Writes  s3://<bucket>/<prefix>monitor/assets/atlas14_grid_24h.npz
    ari    (n,)          average recurrence intervals, years, ascending
    depth  (n, nlat, nlon)  24-h depth in INCHES, NaN outside the Atlas-14 domain
    lats   (nlat,)        descending, identical to the MRMS state grid
    lons   (nlon,)        ascending, identical to the MRMS state grid
"""
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from common import config, pre_key
from monitor_common import state
from monitor_common.s3io import read_parquet, write_bytes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute.p10")

BASE = "https://hdsc.nws.noaa.gov/pub/hdsc/data/orb"   # Volume 2 — contains Indiana
VOLUME = "orb"
DURATION_H = 24
# The full published curve. p02 fetches the same ten RPs per point, so the
# gridded and point paths interpolate over an identical support.
ARIS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
# Atlas-14 ASCII grids are integer inches x 1000.
SCALE = 1000.0
# NODATA is NOT constant across volumes — mw declares -9, orb declares -999 —
# so it is read from each header rather than assumed. Any non-positive depth is
# masked regardless, since a zero or negative design depth is not a real value
# and would otherwise sail through as a plausible 0.00 in.

OUT_KEY_NAME = "atlas14_grid"


def _fetch_asc(rp: int, timeout: float) -> tuple[dict, list[bytes]]:
    """Download one ARI zip and return (header, data lines) without full parse."""
    url = f"{BASE}/{VOLUME}{rp}yr{DURATION_H}ha.zip"
    log.info("  fetching %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        blob = r.read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    name = next(n for n in z.namelist() if n.lower().endswith(".asc"))
    with z.open(name) as f:
        head = {}
        for _ in range(6):
            k, v = f.readline().decode().split()
            head[k.lower()] = float(v)
        lines = f.read().splitlines()
    return head, lines


def _axes(head: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cell-CENTRE lat (descending, row order) and lon (ascending) axes."""
    n_r, n_c = int(head["nrows"]), int(head["ncols"])
    cs = head["cellsize"]
    lon = head["xllcorner"] + (np.arange(n_c) + 0.5) * cs
    lat = head["yllcorner"] + (np.arange(n_r)[::-1] + 0.5) * cs   # row 0 = north
    return lat, lon


def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Index into `src` of the nearest value for each of `dst`. src ascending."""
    i = np.searchsorted(src, dst).clip(1, len(src) - 1)
    left = np.abs(dst - src[i - 1]) <= np.abs(src[i] - dst)
    return np.where(left, i - 1, i)


def _sample(head: dict, lines: list[bytes],
            lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Nearest-sample one Atlas-14 grid onto (lats desc, lons asc), in inches.

    Only the rows and columns the target window touches are parsed — the full
    Midwest grid is 3239x1934 (6.3 M numbers) and Indiana needs ~500 rows of it.
    """
    src_lat, src_lon = _axes(head)                    # lat descending, lon ascending
    r_idx = _nearest(src_lat[::-1], lats)             # against ascending copy
    r_idx = (len(src_lat) - 1) - r_idx                # back to row (descending) order
    c_idx = _nearest(src_lon, lons)

    want = np.unique(r_idx)
    rows: dict[int, np.ndarray] = {}
    for r in want:
        vals = np.array(lines[r].split(), dtype=float)
        if len(vals) != int(head["ncols"]):
            raise ValueError(f"row {r}: {len(vals)} values, header says {head['ncols']}")
        rows[int(r)] = vals[c_idx]
    out = np.vstack([rows[int(r)] for r in r_idx]).astype(np.float32)
    out[out <= 0] = np.nan                      # covers the header NODATA and any 0
    return out / SCALE


PFDS_CSV = ("https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
            "?lat={lat}&lon={lon}&type=pf&data=depth&units=english&series=pds")


def _pfds_point(lat: float, lon: float, timeout: float) -> dict[int, float] | None:
    """Live PFDS 24-h depths by ARI at one point, read from the LABELLED csv.

    The csv names each duration row ("24-hr:, 2.47,..."), so unlike the js
    endpoint that p02 uses there is no positional assumption to get wrong.
    """
    import re
    try:
        with urllib.request.urlopen(PFDS_CSV.format(lat=lat, lon=lon), timeout=timeout) as r:
            text = r.read().decode()
    except Exception as e:  # noqa: BLE001
        log.debug("    PFDS point fetch failed at %.3f,%.3f: %s", lat, lon, e)
        return None
    m = re.search(rf"^{DURATION_H}-hr:,(.+)$", text, re.MULTILINE)
    hdr = re.search(r"by duration for ARI \(years\):,(.+)$", text, re.MULTILINE)
    if not m or not hdr:
        return None
    rps = [int(float(x)) for x in hdr.group(1).split(",")]
    vals = [float(x) for x in m.group(1).split(",")]
    return dict(zip(rps, vals))


def _validate(depth: np.ndarray, lats, lons, aris: list[int],
              n: int, timeout: float) -> None:
    """Spot-check the gridded surface against LIVE PFDS point queries.

    Deliberately not checked against p02's stored table: that table is built
    from the unlabelled js endpoint by positional assumption, so agreeing with
    it would prove only that two paths share an assumption. The labelled csv is
    independent ground truth for units and georeferencing both.
    """
    # Sample away from the domain edge. Volume 2 stops at the Indiana/Michigan
    # line, and PFDS transparently answers from Volume 8 north of it — so an
    # edge cell compares two different volumes and disagrees by ~6% for a
    # perfectly correct grid. Requiring a finite 3x3 neighbourhood keeps the
    # check pointed at the surface rather than at its boundary.
    rng = np.random.default_rng(0)
    ok3 = np.zeros_like(depth[0], dtype=bool)
    ok3[1:-1, 1:-1] = True
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ok3[1:-1, 1:-1] &= np.isfinite(depth[0][1 + di:len(lats) - 1 + di,
                                                    1 + dj:len(lons) - 1 + dj])
    cand = np.argwhere(ok3)
    if len(cand) == 0:
        log.warning("  no interior cells to spot-check")
        return
    pick = cand[rng.choice(len(cand), min(n, len(cand)), replace=False)]
    ii, jj = pick[:, 0], pick[:, 1]
    devs: list[float] = []
    log.info("  live PFDS spot-check (%d interior points, %d-h duration):", len(ii), DURATION_H)
    for i, j in zip(ii, jj):
        la, lo = float(lats[i]), float(lons[j])
        pt = _pfds_point(round(la, 4), round(lo, 4), timeout)
        if not pt:
            continue
        for rp in (10, 100):
            if rp not in pt or rp not in aris:
                continue
            g = float(depth[aris.index(rp)][i, j])
            if not np.isfinite(g) or pt[rp] <= 0:
                continue
            d = (g - pt[rp]) / pt[rp] * 100
            devs.append(d)
            log.info("    %8.3f,%9.3f  %4d-yr  grid %5.2f  pfds %5.2f  %+6.2f%%",
                     la, lo, rp, g, pt[rp], d)
    if devs:
        a = np.abs(devs)
        log.info("  |dev| median %.2f%%  max %.2f%%  (n=%d)",
                 float(np.median(a)), float(a.max()), len(devs))
        if a.max() > 2.0:
            log.warning("  GRID DISAGREES WITH PFDS BY >2%% — check volume, units, georeferencing")
    else:
        log.warning("  no PFDS points returned — cross-check inconclusive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--check-points", type=int, default=8,
                    help="live PFDS points to spot-check the grid against")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch, sample and validate but do not write to S3")
    args = ap.parse_args()

    # Take the target axes from a real stored grid so the ARI raster and the
    # accumulation raster are aligned by construction, not by agreement.
    hours = sorted(state.existing_hours("grid"))
    if not hours:
        raise SystemExit("no stored MRMS grid slices — run the poller first")
    got = state.read_grid(hours[-1])
    if got is None:
        raise SystemExit(f"could not read grid slice {hours[-1]}")
    _, lats, lons = got
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    log.info("Target grid from %s: %d x %d  lat %.3f..%.3f  lon %.3f..%.3f",
             hours[-1], len(lats), len(lons), lats[0], lats[-1], lons[0], lons[-1])

    stack = []
    for rp in ARIS:
        head, lines = _fetch_asc(rp, args.timeout)
        g = _sample(head, lines, lats, lons)
        finite = int(np.isfinite(g).sum())
        log.info("    %4d-yr: %.2f–%.2f in  (%d/%d cells in domain)",
                 rp, float(np.nanmin(g)), float(np.nanmax(g)), finite, g.size)
        stack.append(g)
    depth = np.stack(stack)

    # COVERAGE GATE. Picking the wrong Atlas-14 volume does not error — it
    # returns a grid whose domain simply misses Indiana, and the summary would
    # then render a nearly blank ARI map that looks like "no rain anywhere".
    # Fail loudly instead.
    cov = float(np.isfinite(depth[0]).mean())
    log.info("Domain coverage over the target grid: %.1f%%", cov * 100)
    if cov < 0.95:
        raise SystemExit(
            f"Atlas-14 volume '{VOLUME}' covers only {cov*100:.1f}% of the Indiana "
            f"grid — wrong volume for this state (Indiana is Volume 2, 'orb')")

    # Monotonic in ARI at every cell, or the ARI interpolation is meaningless.
    d = np.diff(depth, axis=0)
    with np.errstate(invalid="ignore"):
        worst = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=0)
    bad = int(np.sum(np.isfinite(worst) & (worst < 0)))
    log.info("Monotonicity: %d of %d cells decrease with ARI", bad, depth.shape[1] * depth.shape[2])

    _validate(depth, lats, lons, ARIS, args.check_points, args.timeout)

    if args.dry_run:
        log.info("dry run — nothing written")
        return
    buf = io.BytesIO()
    np.savez_compressed(buf, ari=np.array(ARIS), depth=depth,
                        lats=lats.astype(np.float32), lons=lons.astype(np.float32))
    b = config.bucket_prefix()[0]
    key = config.keys()[OUT_KEY_NAME]
    write_bytes(buf.getvalue(), b, key, content_type="application/octet-stream")
    log.info("Wrote s3://%s/%s (%.1f MB)", b, key, len(buf.getvalue()) / 1e6)


if __name__ == "__main__":
    main()
