"""04b_regression_flows.py

Fill Q10, Q25, Q50, Q100, Q200, Q500 for stations where the USGS Gage
Statistics Service returned no data (source=None in the script 04 output).

Regional regression equations from:
    Knipe, D. and Rao, A.R. (2005). "Estimation of Peak Discharges of
    Indiana Streams", FHWA/IN/JTRP-2005/1, Joint Transportation Research
    Program, Purdue University. (8 hydrologic regions, 223 Indiana gages)

Equation forms (power law):
    Q_T = C × DA^a1 × Slope^a2                      (Regions 1, 2, 3, 5, 6)
    Q_T = C × DA^a1 × Slope^a2 × (%U + 1)^a3        (Region 4 only)
    Q_T = C × DA^a1 × Slope^a2 × (%W + 1)^a3        (Region 7 only)
    Q_T = C × DA^a1 × (%W + 1)^a2                   (Region 8 only, no slope)

Variables:
    DA    — contributing drainage area (mi²)
    Slope — 10-85 main-channel slope (ft/mi)
    %W    — % basin covered by water/wetlands (NLCD 2019 classes 11+90+95)
    %U    — % basin covered by urban land (NLCD 2019 classes 22+23+24)

    DA, Slope, %U, and %W are all read from the basin_characteristics.parquet
    produced by script 03b.  If slope is missing for a non-Region-8 station,
    a fallback of 1.0 ft/mi is used with a warning.  If %U or %W are missing
    they default to 0 (conservative / rural assumption) with a warning.

Covers return periods: 10, 25, 50, 100, 200, 500 yr.
Q2 and Q5 are NOT produced (not in Rao 2005 equations); those columns
remain null and do not affect script 08 which only uses Q10 and Q50.

Reads:
    s3://<bucket>/<prefix>flow_stats/per_gauge_flow_stats.parquet
    s3://<bucket>/<prefix>stations/indiana_streamflow_sites.parquet
    s3://<bucket>/<prefix>watersheds/basin_characteristics.parquet  (script 03b)

Writes:
    s3://<bucket>/<prefix>flow_stats/per_gauge_flow_stats.parquet  (updated)
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from utils import load_config, s3_client, write_parquet_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("04b_regression")

# ── Regional regression coefficients ─────────────────────────────────────────
# Source: Knipe & Rao (2005) FHWA/IN/JTRP-2005/1, Tables 4.4–4.11
# Format per cell: (C, a1_DA, a2_slope_or_%W, a3_optional)
#   Region 8: equation uses (%W+1) in place of slope → a2 = exp for %W+1
#   Region 7: standard slope + (%W+1)^a3
#   Region 4: standard slope + (%U+1)^a3
#   Others:   C × DA^a1 × Slope^a2
REGION_COEFF: dict[int, dict[int, tuple]] = {
    1: {
        10:  (47.8,  0.802, 0.535, None),
        25:  (55.3,  0.805, 0.561, None),
        50:  (61.4,  0.805, 0.573, None),
        100: (67.5,  0.805, 0.585, None),
        200: (74.3,  0.803, 0.592, None),
        500: (83.9,  0.800, 0.599, None),
    },
    2: {
        10:  (69.6,  0.798, 0.473, None),
        25:  (102.4, 0.777, 0.441, None),
        50:  (133.1, 0.762, 0.417, None),
        100: (169.5, 0.748, 0.394, None),
        200: (213.3, 0.734, 0.371, None),
        500: (283.3, 0.716, 0.341, None),
    },
    3: {
        10:  (74.6,  0.889, 0.416, None),
        25:  (91.5,  0.891, 0.425, None),
        50:  (104.5, 0.894, 0.430, None),
        100: (116.8, 0.898, 0.434, None),
        200: (132.5, 0.898, 0.434, None),
        500: (152.1, 0.902, 0.437, None),
    },
    4: {
        10:  (31.1,  0.820, 0.681, 0.080),
        25:  (37.7,  0.820, 0.698, 0.079),
        50:  (42.9,  0.819, 0.707, 0.077),
        100: (48.4,  0.816, 0.712, 0.075),
        200: (52.7,  0.816, 0.722, 0.074),
        500: (58.7,  0.815, 0.731, 0.073),
    },
    5: {
        10:  (35.8,  0.776, 0.368, None),
        25:  (45.6,  0.764, 0.356, None),
        50:  (53.1,  0.756, 0.347, None),
        100: (60.8,  0.748, 0.338, None),
        200: (68.7,  0.742, 0.330, None),
        500: (79.5,  0.734, 0.319, None),
    },
    6: {
        10:  (22.4,  0.732, 0.776, None),
        25:  (27.9,  0.709, 0.858, None),
        50:  (31.5,  0.696, 0.917, None),
        100: (34.6,  0.687, 0.974, None),
        200: (37.3,  0.681, 1.029, None),
        500: (40.3,  0.675, 1.098, None),
    },
    7: {
        10:  (65.0,  0.873, 0.372, -0.795),
        25:  (89.0,  0.858, 0.361, -0.801),
        50:  (108.4, 0.849, 0.354, -0.803),
        100: (129.3, 0.839, 0.347, -0.803),
        200: (151.1, 0.831, 0.343, -0.802),
        500: (182.2, 0.821, 0.336, -0.800),
    },
    8: {
        10:  (106.0, 0.835, -0.733, None),
        25:  (118.2, 0.839, -0.719, None),
        50:  (126.5, 0.842, -0.707, None),
        100: (134.2, 0.843, -0.695, None),
        200: (141.1, 0.845, -0.683, None),
        500: (149.8, 0.846, -0.667, None),
    },
}

RETURN_PERIODS = [10, 25, 50, 100, 200, 500]
Q_COLS = {10: "Q10", 25: "Q25", 50: "Q50", 100: "Q100", 200: "Q200", 500: "Q500"}


# ── Region assignment ─────────────────────────────────────────────────────────

def assign_region(lat: float, lon: float) -> int:
    """Approximate Rao 2005 region for a station coordinate.

    Boundaries are approximate lat/lon boxes derived from the county map in
    Figure 3.3 of Knipe & Rao (2005). Errors near region borders are
    acceptable given the equations already carry 24–45% standard error.
    """
    # Region 8: far northwest (Lake/Porter/Newton/Jasper/Pulaski — Lake Michigan)
    if lat > 41.0 and lon < -86.7:
        return 8
    # Region 7: northern tier (lake-dominated, St. Joseph/Elkhart/LaPorte/Starke)
    if lat > 41.0:
        return 7
    # Region 1: northeast (Allen/Noble/DeKalb/Whitley/Huntington/Wabash/Miami/Cass)
    if lat > 40.2 and lon > -85.8:
        return 1
    # Region 2: east-central (Adams/Wells/Blackford/Grant/Delaware/Madison/Randolph/Jay)
    if lat > 39.7 and lon > -85.5:
        return 2
    # Region 3: west (Vermillion/Parke/Vigo/Clay/Sullivan/Knox/Gibson/Posey)
    if lon < -87.0 and lat > 38.5:
        return 3
    # Region 4: central (large middle band including Indianapolis metro)
    if lat > 39.5:
        return 4
    # Region 5: south-central (Monroe/Lawrence/Martin/Orange/Dubois/Washington/Brown)
    if lon < -86.2 and lat > 38.0:
        return 5
    # Region 6: southeast (Ripley/Jefferson/Jennings/Dearborn/Clark/Floyd/Scott)
    return 6


# ── Regression evaluation ─────────────────────────────────────────────────────

def apply_regression(
    region: int,
    rp: int,
    da: float,
    slope: float,
    pct_w: float = 0.0,
    pct_u: float = 0.0,
) -> Optional[float]:
    """Return peak flow Q_T (cfs) for given region and return period.

    pct_w and pct_u are percentages (0–100).
    """
    coeffs = REGION_COEFF.get(region, {}).get(rp)
    if coeffs is None:
        return None
    C, a1, a2, a3 = coeffs

    if region == 8:
        q = C * (da ** a1) * ((pct_w + 1) ** a2)
    elif region == 7:
        q = C * (da ** a1) * (slope ** a2) * ((pct_w + 1) ** a3)
    elif region == 4:
        q = C * (da ** a1) * (slope ** a2) * ((pct_u + 1) ** a3)
    else:
        q = C * (da ** a1) * (slope ** a2)

    return round(q, 1) if q > 0 else None


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


# ── Per-station processor ─────────────────────────────────────────────────────

def process_site(
    site_no: str,
    lat: float,
    lon: float,
    basin_char: dict,
) -> dict:
    """Compute Rao (2005) regression flows for one station.

    basin_char is the row from basin_characteristics.parquet as a plain dict.
    """
    region = assign_region(lat, lon)
    out: dict = {"site_no": site_no, "region": region}

    da_raw = basin_char.get("drain_area_mi2")
    if da_raw is None or pd.isna(da_raw):
        log.warning("%s: no drainage area in basin_characteristics — skipping", site_no)
        return out
    da = float(da_raw)

    slope_raw = basin_char.get("slope_ft_mi")
    if slope_raw is None or pd.isna(slope_raw):
        if region != 8:
            log.warning("%s: no slope in basin_characteristics — using 1.0 ft/mi", site_no)
        slope = 1.0
    else:
        slope = float(slope_raw)

    pct_u_raw = basin_char.get("pct_u")
    pct_w_raw = basin_char.get("pct_w")

    if region == 4 and (pct_u_raw is None or pd.isna(pct_u_raw)):
        log.error("%s: region 4 requires %%U but pct_u is missing — skipping (rerun 03b)", site_no)
        return out
    if region in (7, 8) and (pct_w_raw is None or pd.isna(pct_w_raw)):
        log.error("%s: region %d requires %%W but pct_w is missing — skipping (rerun 03b)", site_no, region)
        return out

    pct_u = float(pct_u_raw) if pct_u_raw is not None and not pd.isna(pct_u_raw) else 0.0
    pct_w = float(pct_w_raw) if pct_w_raw is not None and not pd.isna(pct_w_raw) else 0.0

    flows = {}
    for rp in RETURN_PERIODS:
        q = apply_regression(region, rp, da, slope, pct_w=pct_w, pct_u=pct_u)
        if q is not None:
            flows[Q_COLS[rp]] = q

    if flows:
        out["source"] = "regression"
        out.update(flows)

    log.debug(
        "%s: region=%d da=%.1f slope=%.2f pct_u=%.1f pct_w=%.1f Q10=%s Q50=%s",
        site_no, region, da, slope, pct_u, pct_w,
        flows.get("Q10"), flows.get("Q50"),
    )
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    bucket = cfg["aws"]["output_bucket"]
    prefix = cfg["aws"]["output_prefix"]

    flow_stats = _read_parquet_s3(bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    flow_stats["site_no"] = flow_stats["site_no"].astype(str)

    inv = _read_parquet_s3(bucket, f"{prefix}stations/indiana_streamflow_sites.parquet")
    inv["site_no"] = inv["site_no"].astype(str)
    inv = inv[["site_no", "dec_lat_va", "dec_long_va"]].copy()

    try:
        basin_chars = _read_parquet_s3(
            bucket, f"{prefix}watersheds/basin_characteristics.parquet"
        )
        basin_chars["site_no"] = basin_chars["site_no"].astype(str)
        basin_char_map: dict[str, dict] = basin_chars.set_index("site_no").to_dict("index")
        log.info("Loaded basin characteristics for %d stations", len(basin_char_map))
    except Exception as e:
        raise RuntimeError(
            f"Could not load basin_characteristics.parquet: {e}\n"
            "Run script 03b first to generate basin characteristics."
        ) from e

    no_source      = flow_stats["source"].isna()
    partial_gage   = (
        flow_stats["source"].eq("gage_stats")
        & (flow_stats["Q10"].isna() | flow_stats["Q50"].isna())
    )
    existing_regr  = flow_stats["source"].eq("regression")
    needs_fill     = flow_stats[no_source | partial_gage | existing_regr]["site_no"].tolist()
    log.info(
        "Stations needing regression fill: %d (%d source=None, %d partial gage_stats)",
        len(needs_fill), int(no_source.sum()), int(partial_gage.sum()),
    )
    if not needs_fill:
        log.info("No gaps to fill. Exiting.")
        return

    targets = pd.DataFrame({"site_no": needs_fill}).merge(inv, on="site_no", how="left")
    missing_coords = targets[targets["dec_lat_va"].isna()]
    if len(missing_coords):
        log.warning(
            "Skipping %d stations with no coordinates: %s",
            len(missing_coords), missing_coords["site_no"].tolist(),
        )
    targets = targets.dropna(subset=["dec_lat_va", "dec_long_va"])
    log.info("Processing %d stations via regression", len(targets))

    max_workers = cfg["streamstats"].get("max_concurrent", 8)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                process_site,
                str(row["site_no"]),
                float(row["dec_lat_va"]),
                float(row["dec_long_va"]),
                basin_char_map.get(str(row["site_no"]), {}),
            ): str(row["site_no"])
            for _, row in targets.iterrows()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            site = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                log.error("Failed %s: %s", site, e)
                continue
            results.append(rec)
            if i % 25 == 0 or i == len(futs):
                log.info(
                    "[%d/%d] %s region=%s Q10=%s Q50=%s",
                    i, len(futs),
                    rec.get("site_no"), rec.get("region"),
                    rec.get("Q10"), rec.get("Q50"),
                )

    reg_df = pd.DataFrame(results)
    if reg_df.empty:
        log.error("No regression results produced.")
        return

    n_filled = int((reg_df.get("source") == "regression").sum()) if "source" in reg_df.columns else 0
    log.info("Regression filled %d / %d stations", n_filled, len(results))

    flow_stats = flow_stats.set_index("site_no")
    for _, row in reg_df.iterrows():
        site = row["site_no"]
        if site not in flow_stats.index:
            continue
        has_gage_stats = flow_stats.at[site, "source"] == "gage_stats"
        for col in ["Q10", "Q25", "Q50", "Q100", "Q200", "Q500"]:
            if col in row and not pd.isna(row[col]):
                if not has_gage_stats or pd.isna(flow_stats.at[site, col]):
                    flow_stats.at[site, col] = row[col]
        if not has_gage_stats:
            if "source" in row and not pd.isna(row.get("source")):
                flow_stats.at[site, "source"] = row["source"]
            if "regression_region" in flow_stats.columns:
                flow_stats.at[site, "regression_region"] = f"Rao2005_R{row.get('region', '?')}"
    flow_stats = flow_stats.reset_index()

    write_parquet_to_s3(flow_stats, bucket, f"{prefix}flow_stats/per_gauge_flow_stats.parquet")
    src_counts = flow_stats["source"].value_counts(dropna=False).to_dict()
    log.info("Done. Source counts: %s", src_counts)
    log.info(
        "Q10 non-null: %d, Q50 non-null: %d (of %d total)",
        flow_stats["Q10"].notna().sum(),
        flow_stats["Q50"].notna().sum(),
        len(flow_stats),
    )


if __name__ == "__main__":
    main()
