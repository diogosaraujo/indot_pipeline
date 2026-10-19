"""ArcGIS layers showing what the nearest-pixel MRMS trigger actually samples.

The precip trigger reads ONE MRMS cell — the one nearest the bridge — and
compares its trailing Tc-hour accumulation against an Atlas-14 depth. The basin
draining to the bridge is much larger than that cell, so any rain that falls on
the basin but not on that cell is invisible to the trigger.

For bridge 08-00114 (Rattlesnake Creek, COMID 20100310):
    basin                78.9 km2
    MRMS cell            ~0.93 km2 at this latitude (0.01 deg)
    cells over the basin 110 intersecting, 82 more than half inside
    cells sampled        1   (~1.2% of the basin area)
    trigger              trailing 16 h >= P50 5.17 in (non-scour)

Layers written (EPSG:4326, ready for ArcGIS)
────────────────────────────────────────────
    watershed_<id>.shp      the delineated basin
    mrms_pixels_<id>.shp    one polygon per MRMS cell touching the basin, with
                            COV_FRAC (fraction of the cell inside the basin) and
                            SAMPLED (1 on the single cell the trigger reads)
    bridge_<id>.shp         the bridge point
    mrms_<id>.asc + .prj    the same cells as a raster, on the true MRMS grid

Raster values are COV_FRAC by default. With --event they become the real
trailing-accumulation depth in inches, which is the version that shows the
spatial gradient the trigger throws away.

--event needs cfgrib to read MRMS GRIB2 and so has to run where the pipeline
env is (EC2, ~/miniforge3/envs/indot). Without it the script needs only
geopandas and runs anywhere, including this Windows box.

ESRI ASCII rather than GeoTIFF on purpose: it needs no rasterio, ArcGIS reads it
natively, and the MRMS grid is a regular 0.01 deg lattice in geographic
coordinates, which is exactly what the format assumes.

Usage
    python scripts/mrms_pixel_vs_watershed.py --bridge 08-00114
    python scripts/mrms_pixel_vs_watershed.py --bridge 08-00114 \
        --event-end "2026-06-09 12:00"          # needs cfgrib; adds real depths
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mrms_px")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "monitor"))

CELL = 0.01                       # MRMS grid spacing, degrees
WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
    '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)


def monitor_io():
    from monitor_common import config as mcfg
    from monitor_common.s3io import read_bytes, read_parquet
    b, p = mcfg.bucket_prefix()
    return b, p, read_bytes, read_parquet


def mrms_axes(bucket, prefix, read_bytes) -> tuple[np.ndarray, np.ndarray]:
    """The real MRMS lat/lon cell centres, taken from a stored grid slice.

    Using a stored slice rather than reconstructing the lattice keeps the cells
    aligned with the ones the trigger actually samples.
    """
    import io
    from monitor_common.s3io import list_keys
    keys = sorted(list_keys(bucket, f"{prefix}monitor/state/mrms_grid/"))
    if not keys:
        raise SystemExit("no mrms_grid slices in S3 to take the grid axes from")
    z = np.load(io.BytesIO(read_bytes(bucket, keys[-1])))
    return z["lats"], z["lons"]          # lats descending, lons ascending


def accumulation(end: pd.Timestamp, hours: int, lats, lons) -> np.ndarray | None:
    """Trailing `hours` MRMS sum on the Indiana grid, in inches. Needs cfgrib."""
    try:
        from monitor_common import mrms
    except Exception as e:                                   # noqa: BLE001
        log.error("cannot import monitor_common.mrms (%s)", e)
        return None
    tot = np.zeros((len(lats), len(lons)), dtype="float64")
    got = 0
    for k in range(hours):
        ts = end - pd.Timedelta(hours=k)
        try:
            g = mrms.read_grid(ts)
        except Exception as e:                               # noqa: BLE001
            log.warning("  %s unreadable (%s)", ts, e); continue
        if g is None:
            log.warning("  %s missing", ts); continue
        arr, gl, go = g
        ri = np.searchsorted(-gl, -lats); ci = np.searchsorted(go, lons)
        ri = np.clip(ri, 0, len(gl) - 1); ci = np.clip(ci, 0, len(go) - 1)
        tot += np.nan_to_num(arr[np.ix_(ri, ci)], nan=0.0)
        got += 1
    log.info("  accumulated %d of %d hours ending %s", got, hours, end)
    return tot if got else None


def write_asc(path: pathlib.Path, vals: np.ndarray, lats, lons, nodata=-9999.0):
    """ESRI ASCII Grid. Rows run north->south, matching MRMS lat ordering."""
    with path.open("w") as f:
        f.write(f"ncols {len(lons)}\nnrows {len(lats)}\n")
        f.write(f"xllcorner {lons[0] - CELL / 2:.6f}\n")
        f.write(f"yllcorner {lats[-1] - CELL / 2:.6f}\n")
        f.write(f"cellsize {CELL}\nNODATA_value {nodata:g}\n")
        for row in vals:
            f.write(" ".join("%.4f" % v if np.isfinite(v) else "%g" % nodata
                             for v in row) + "\n")
    path.with_suffix(".prj").write_text(WGS84_WKT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridge", default="08-00114")
    ap.add_argument("--event-end", default=None,
                    help="UTC end of the trailing accumulation window; needs cfgrib")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    bucket, prefix, read_bytes, read_parquet = monitor_io()
    cfg = read_parquet(bucket, f"{prefix}monitor/bridge_monitor_config.parquet")
    cfg["bridge_id"] = cfg["bridge_id"].astype(str).str.strip()
    row = cfg[cfg.bridge_id == args.bridge]
    if row.empty:
        raise SystemExit(f"bridge {args.bridge} not in the monitor config")
    row = row.iloc[0]
    comid = int(row["comid"]); blat, blon = float(row["lat"]), float(row["lon"])
    tc_hr = int(row["tc_dur_hr"]); scour = bool(row["scour_critical"])
    thresh = float(row["P10"] if scour else row["P50"])
    log.info("%s  COMID %d  Tc window %d h  gate P%s = %.2f in",
             args.bridge, comid, tc_hr, "10" if scour else "50", thresh)

    import io
    ws = gpd.read_file(io.BytesIO(read_bytes(
        bucket, f"{prefix}monitor/precompute/watersheds/{comid}.geojson")))
    ws = ws.set_crs(4326, allow_override=True)
    basin = ws.geometry.union_all()
    area_km2 = gpd.GeoSeries([basin], crs=4326).to_crs(26916).area.iloc[0] / 1e6
    log.info("basin %.1f km2", area_km2)

    lats, lons = mrms_axes(bucket, prefix, read_bytes)
    b = basin.bounds
    ri = np.where((lats >= b[1] - CELL) & (lats <= b[3] + CELL))[0]
    ci = np.where((lons >= b[0] - CELL) & (lons <= b[2] + CELL))[0]
    sub_lats, sub_lons = lats[ri], lons[ci]

    acc = None
    if args.event_end:
        acc = accumulation(pd.Timestamp(args.event_end, tz="UTC"), tc_hr, lats, lons)
        if acc is not None:
            acc = acc[np.ix_(ri, ci)]

    # the one cell the trigger reads
    si = int(np.argmin(np.abs(sub_lats - blat)))
    sj = int(np.argmin(np.abs(sub_lons - blon)))

    half = CELL / 2
    recs, cov_grid = [], np.full((len(sub_lats), len(sub_lons)), np.nan)
    for i, la in enumerate(sub_lats):
        for j, lo in enumerate(sub_lons):
            cell = box(lo - half, la - half, lo + half, la + half)
            if not cell.intersects(basin):
                continue
            frac = cell.intersection(basin).area / cell.area
            cov_grid[i, j] = frac
            r = {"ROW": i, "COL": j, "LAT": round(float(la), 4),
                 "LON": round(float(lo), 4), "COV_FRAC": round(float(frac), 4),
                 "SAMPLED": int(i == si and j == sj), "geometry": cell}
            if acc is not None:
                r["ACCUM_IN"] = round(float(acc[i, j]), 4)
            recs.append(r)
    px = gpd.GeoDataFrame(recs, crs=4326)
    log.info("MRMS cells touching the basin: %d  (>50%% inside: %d)",
             len(px), int((px.COV_FRAC > 0.5).sum()))
    log.info("cells sampled by the trigger: %d  -> %.1f%% of the basin area",
             int(px.SAMPLED.sum()), 100.0 * 0.93 / area_km2)

    out = pathlib.Path(args.out_dir or (REPO / "results" / f"mrms_vs_watershed_{args.bridge}"))
    out.mkdir(parents=True, exist_ok=True)
    tag = args.bridge.replace(" ", "_")
    gpd.GeoDataFrame(geometry=[basin], crs=4326).assign(
        BRIDGE_ID=args.bridge, COMID=comid, AREA_KM2=round(area_km2, 2),
        TC_HR=tc_hr, GATE_IN=round(thresh, 3)).to_file(out / f"watershed_{tag}.shp")
    px.to_file(out / f"mrms_pixels_{tag}.shp")
    gpd.GeoDataFrame(geometry=[Point(blon, blat)], crs=4326).assign(
        BRIDGE_ID=args.bridge, COMID=comid, TC_HR=tc_hr,
        GATE_IN=round(thresh, 3)).to_file(out / f"bridge_{tag}.shp")

    vals = acc if acc is not None else cov_grid
    vals = np.where(np.isnan(cov_grid), np.nan, vals)      # basin cells only
    write_asc(out / f"mrms_{tag}.asc", vals, sub_lats, sub_lons)
    log.info("wrote %s", out)
    log.info("  raster values = %s",
             "trailing accumulation, inches" if acc is not None else "coverage fraction")


if __name__ == "__main__":
    main()
