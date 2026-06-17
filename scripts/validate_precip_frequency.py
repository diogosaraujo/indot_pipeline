"""validate_precip_frequency.py

Precipitation event identification and frequency validation for Indiana.

Counts distinct precipitation events per return period (P2–P1000) across
Indiana gauge locations using hourly MRMS QPE data, then compares observed
frequencies to theoretical Atlas 14 expectations.

Algorithm
─────────
Step 1 — Temporal event separation (Restrepo-Posada & Eagleson 1982;
          Giani et al. 2022): two precipitation periods are independent
          if separated by ≥DRY_HOURS_MIN consecutive hours with hourly
          precipitation < DRY_THRESHOLD_IN.

Steps 2–4 — Spatial identification, almost-connected component clustering,
          and temporal storm tracking (Baldwin et al. 2005; Chang et al. 2016;
          Murthy et al. 2015; Prein et al. 2017; Eddins 2010).
          These operate on the full MRMS 1-km grid and are enabled by the
          --spatial flag.  Default mode (temporal only) uses the pre-extracted
          per-location parquet already in the pipeline.

Return period assignment: each event's maximum rolling accumulation across
PRECIP_DURATIONS_HR is compared to Atlas 14 thresholds for that duration.
The highest return period exceeded is assigned to the event.

Expected counts: (1 / RP) × n_locations × n_years.
Ratio = Observed / Expected.  ✓: 0.67–1.5.  ✗: outside that range.

Reads (s3://indot-bridge-pipeline/v1/):
    mrms/QPE_01H_Pass2/nearest_pixel.parquet
    atlas14/precipitation_frequency.parquet
    stations/indiana_streamflow_sites.parquet

Writes (s3://indot-bridge-pipeline/v1/analysis/precip_frequency/):
    indiana_precip_frequency_summary.csv
    indiana_precip_frequency_by_location.csv
    sensitivity_analysis.csv
    summary_histogram.png
    event_timeline.png

Usage:
    python scripts/validate_precip_frequency.py
    python scripts/validate_precip_frequency.py --locations path/to/locs.csv
    python scripts/validate_precip_frequency.py --start 2022-01-01 --end 2025-12-31
    python scripts/validate_precip_frequency.py --spatial   # full GRIB2 grid mode
"""
from __future__ import annotations

import argparse
import io
import warnings
from dataclasses import dataclass, field
from typing import Optional

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
BUCKET  = "indot-bridge-pipeline"
PREFIX  = "v1/"
S3_OUT  = "v1/analysis/precip_frequency/"

# Analysis window — aligns with MRMS PDS availability on noaa-mrms-pds
DEFAULT_START = "2020-01-01"
DEFAULT_END   = "2026-12-31"

RETURN_PERIODS      = [2, 10, 25, 50, 100, 500, 1000]
PRECIP_DURATIONS_HR = [1, 3, 6, 12, 24]  # Atlas 14 durations to check

# Temporal separation parameters (Step 1)
DRY_THRESHOLD_IN = 0.01   # in/hr — hourly precip below this counts as dry
DRY_HOURS_MIN    = 24     # consecutive dry hours required to end an event

# Spatial algorithm parameters (Steps 2–4; used only with --spatial)
PRECIP_THRESH_MMHR  = 0.10    # grid-cell detection threshold (WMO light precip)
DILATION_RADIUS_KM  = 10.0    # kernel for almost-connected component merging
TRACKING_DIST_KM    = 40.0    # max centroid shift across consecutive hours
JACCARD_MIN         = 0.25    # morphological similarity (intersection / union)
MRMS_GRIB2_BUCKET   = "noaa-mrms-pds"
MRMS_GRIB2_FOLDER   = "MultiSensor_QPE_01H_Pass2_00.00"  # NOAA PDS folder name

# Validation thresholds
RATIO_OK_LO = 0.67
RATIO_OK_HI = 1.50

# Indiana bounding box (for spatial mode cropping)
INDIANA_BBOX = dict(lat_min=37.77, lat_max=41.76, lon_min=-88.10, lon_max=-84.78)


# ── Sensitivity parameter sets ─────────────────────────────────────────────────
@dataclass
class SensParams:
    label:           str
    dry_threshold:   float = DRY_THRESHOLD_IN
    dry_hours:       int   = DRY_HOURS_MIN
    precip_thresh:   float = PRECIP_THRESH_MMHR
    kernel_km:       float = DILATION_RADIUS_KM
    dist_km:         float = TRACKING_DIST_KM
    jaccard:         float = JACCARD_MIN

SENSITIVITY_SETS: list[SensParams] = [
    SensParams("base"),
    SensParams("dry_thresh_low",  dry_threshold=0.005),
    SensParams("dry_thresh_high", dry_threshold=0.020),
    SensParams("dry_hours_12",    dry_hours=12),
    SensParams("dry_hours_48",    dry_hours=48),
    SensParams("thresh_low",      precip_thresh=0.05),
    SensParams("thresh_high",     precip_thresh=0.20),
    SensParams("kernel_small",    kernel_km=5.0),
    SensParams("kernel_large",    kernel_km=15.0),
    SensParams("dist_close",      dist_km=30.0),
    SensParams("dist_far",        dist_km=50.0),
    SensParams("jaccard_low",     jaccard=0.20),
    SensParams("jaccard_high",    jaccard=0.30),
]


# ── S3 helpers ─────────────────────────────────────────────────────────────────
_s3_fs: pafs.S3FileSystem | None = None
_s3_client = None


def _fs() -> pafs.S3FileSystem:
    global _s3_fs
    if _s3_fs is None:
        _s3_fs = pafs.S3FileSystem()
    return _s3_fs


def _client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def read_s3(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(
        f"{BUCKET}/{PREFIX}{key}", filesystem=_fs(), columns=columns
    ).to_pandas()


def upload_bytes(data: bytes, filename: str, content_type: str) -> None:
    key = S3_OUT + filename
    _client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    print(f"  → s3://{BUCKET}/{key}")


def upload_csv(df: pd.DataFrame, filename: str) -> None:
    upload_bytes(df.to_csv(index=False).encode(), filename, "text/csv")


def upload_fig(fig: plt.Figure, filename: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    upload_bytes(buf.read(), filename, "image/png")


# ── Location loading ───────────────────────────────────────────────────────────
def load_locations(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Return DataFrame with columns [location_id, lat, lon].

    If csv_path is given it must have columns location_id, lat, lon (and
    optionally bridge_id).  Otherwise falls back to the pipeline station
    inventory, using site_no as location_id.
    """
    if csv_path:
        locs = pd.read_csv(csv_path, dtype={"location_id": str})
        required = {"location_id", "lat", "lon"}
        missing = required - set(locs.columns)
        if missing:
            raise ValueError(f"Locations CSV missing columns: {missing}")
        print(f"  Loaded {len(locs)} locations from {csv_path}")
        return locs.reset_index(drop=True)

    inv = read_s3(
        "stations/indiana_streamflow_sites.parquet",
        columns=["site_no", "dec_lat_va", "dec_long_va"],
    )
    inv["site_no"] = inv["site_no"].astype(str)
    locs = inv.rename(columns={
        "site_no":      "location_id",
        "dec_lat_va":   "lat",
        "dec_long_va":  "lon",
    }).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  Using {len(locs)} pipeline stations as locations")
    return locs


# ── MRMS data ──────────────────────────────────────────────────────────────────
def load_mrms(
    location_ids: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load hourly MRMS QPE per-location from the pipeline parquet.

    Returns long-form DataFrame: [site_no, datetime_utc, precip_in].
    Values already in inches (pipeline applies unit conversion in script 05).
    """
    print("Loading MRMS QPE_01H_Pass2 from pipeline parquet...")
    mrms = read_s3(
        "mrms/QPE_01H_Pass2/nearest_pixel.parquet",
        columns=["site_no", "datetime_utc", "value"],
    )
    mrms["site_no"]      = mrms["site_no"].astype(str)
    mrms["datetime_utc"] = pd.to_datetime(mrms["datetime_utc"], utc=True)
    mrms = mrms.rename(columns={"value": "precip_in"})
    mrms["precip_in"]    = pd.to_numeric(mrms["precip_in"], errors="coerce").fillna(0.0)

    mrms = mrms[
        (mrms["site_no"].isin(location_ids))
        & (mrms["datetime_utc"] >= pd.Timestamp(start, tz="UTC"))
        & (mrms["datetime_utc"] <= pd.Timestamp(end,   tz="UTC"))
    ]
    n_sta = mrms["site_no"].nunique()
    t0    = mrms["datetime_utc"].min().date()
    t1    = mrms["datetime_utc"].max().date()
    print(f"  {len(mrms):,} hourly records  |  {n_sta} stations  |  {t0} → {t1}")
    return mrms


# ── Atlas 14 loading ───────────────────────────────────────────────────────────
def load_atlas14(location_ids: list[str]) -> pd.DataFrame:
    """Return Atlas 14 thresholds for requested locations and all RP × duration
    combinations.

    Returns: DataFrame [site_no, return_period_yr, duration_hr, depth_in]
    Filtered to RETURN_PERIODS and PRECIP_DURATIONS_HR.
    """
    print("Loading Atlas 14 precipitation frequency...")
    a14 = read_s3("atlas14/precipitation_frequency.parquet")
    a14["site_no"] = a14["site_no"].astype(str)
    a14 = a14[
        a14["site_no"].isin(location_ids)
        & a14["return_period_yr"].isin(RETURN_PERIODS)
        & a14["duration_hr"].isin(PRECIP_DURATIONS_HR)
    ]
    n_sta = a14["site_no"].nunique()
    print(f"  Atlas 14 loaded: {n_sta} stations  ×  {a14['return_period_yr'].nunique()} RPs  ×  {a14['duration_hr'].nunique()} durations")
    return a14


# ── Step 1: Temporal event identification ──────────────────────────────────────
def identify_events(
    precip: pd.Series,
    dry_threshold_in: float = DRY_THRESHOLD_IN,
    dry_hours_min: int = DRY_HOURS_MIN,
) -> list[dict]:
    """Identify independent precipitation events in a per-location hourly series.

    Two events are independent when separated by ≥dry_hours_min consecutive
    hours with precip < dry_threshold_in  (Restrepo-Posada & Eagleson 1982).

    precip: pd.Series indexed by datetime (hourly, UTC), values in inches.

    Returns list of dicts:
        start, end (Timestamps), duration_hr (int),
        total_in (float), max_1hr_in (float)
    """
    precip = precip.sort_index().fillna(0.0)

    events: list[dict] = []
    event_start: Optional[pd.Timestamp] = None
    dry_streak = 0
    last_wet: Optional[pd.Timestamp] = None

    for ts, val in precip.items():
        if val >= dry_threshold_in:
            dry_streak = 0
            last_wet = ts
            if event_start is None:
                event_start = ts
        else:
            if event_start is not None:
                dry_streak += 1
                if dry_streak >= dry_hours_min:
                    seg = precip[event_start:last_wet]
                    events.append({
                        "start":        event_start,
                        "end":          last_wet,
                        "duration_hr":  int((last_wet - event_start).total_seconds() / 3600) + 1,
                        "total_in":     float(seg.sum()),
                        "max_1hr_in":   float(seg.max()),
                    })
                    event_start = None
                    dry_streak  = 0
                    last_wet    = None

    # Close any open event at end of record
    if event_start is not None and last_wet is not None:
        seg = precip[event_start:last_wet]
        events.append({
            "start":       event_start,
            "end":         last_wet,
            "duration_hr": int((last_wet - event_start).total_seconds() / 3600) + 1,
            "total_in":    float(seg.sum()),
            "max_1hr_in":  float(seg.max()),
        })

    return events


# ── Return period assignment ───────────────────────────────────────────────────
def assign_return_period(
    event: dict,
    precip: pd.Series,
    a14_site: pd.DataFrame,
) -> Optional[int]:
    """Return the highest Atlas 14 return period exceeded during this event.

    Computes rolling accumulations for each duration in PRECIP_DURATIONS_HR
    within the event window and compares to Atlas 14 thresholds.
    Returns None if no threshold exceeded (event below P2).
    """
    seg = precip[event["start"]:event["end"]]
    if seg.empty:
        return None

    best_rp: Optional[int] = None
    for rp in sorted(RETURN_PERIODS, reverse=True):
        for dur_h in PRECIP_DURATIONS_HR:
            row = a14_site[
                (a14_site["return_period_yr"] == rp) &
                (a14_site["duration_hr"] == dur_h)
            ]
            if row.empty:
                continue
            threshold = float(row["depth_in"].iloc[0])
            rolling   = seg.rolling(dur_h, min_periods=dur_h).sum()
            if rolling.max() >= threshold:
                best_rp = rp
                break   # already found highest RP for this duration; move to next RP check
        if best_rp == rp:
            break       # highest possible RP found; stop searching lower ones

    # Re-check: highest RP exceeded is found by descending search above
    # Redo cleanly: check each RP from highest to lowest, return first hit
    best_rp = None
    for rp in sorted(RETURN_PERIODS, reverse=True):
        exceeded = False
        for dur_h in PRECIP_DURATIONS_HR:
            row = a14_site[
                (a14_site["return_period_yr"] == rp) &
                (a14_site["duration_hr"] == dur_h)
            ]
            if row.empty:
                continue
            threshold = float(row["depth_in"].iloc[0])
            rolling   = seg.rolling(dur_h, min_periods=dur_h).sum()
            if rolling.max() >= threshold:
                exceeded = True
                break
        if exceeded:
            best_rp = rp
            break

    return best_rp


# ── Per-location event counting ────────────────────────────────────────────────
def count_events_per_location(
    mrms: pd.DataFrame,
    atlas14: pd.DataFrame,
    dry_threshold_in: float = DRY_THRESHOLD_IN,
    dry_hours_min: int = DRY_HOURS_MIN,
) -> pd.DataFrame:
    """Run event identification and return-period assignment for all locations.

    Returns DataFrame with one row per location:
        location_id, n_events_total, P2, P10, P25, P50, P100, P500, P1000
    where each P{rp} column is the count of events exceeding that threshold
    (i.e. events AT or ABOVE that RP, including higher ones).
    """
    sites   = sorted(mrms["site_no"].unique())
    a14_map = {s: g for s, g in atlas14.groupby("site_no")}

    rows: list[dict] = []
    for i, site in enumerate(sites, 1):
        if i % 25 == 0 or i == len(sites):
            print(f"  [{i}/{len(sites)}] processed")

        ts = (
            mrms[mrms["site_no"] == site]
            .set_index("datetime_utc")["precip_in"]
            .sort_index()
        )
        a14_site = a14_map.get(site, pd.DataFrame())

        events = identify_events(ts, dry_threshold_in, dry_hours_min)

        counts: dict[int, int] = {rp: 0 for rp in RETURN_PERIODS}
        for ev in events:
            if a14_site.empty:
                continue
            rp = assign_return_period(ev, ts, a14_site)
            if rp is not None:
                # Count event under its exact RP AND all lower RPs
                for r in RETURN_PERIODS:
                    if r <= rp:
                        counts[r] += 1

        row = {"location_id": site, "n_events_total": len(events)}
        row.update({f"P{rp}": counts[rp] for rp in RETURN_PERIODS})
        rows.append(row)

    df = pd.DataFrame(rows)
    n_yrs = (
        pd.Timestamp(DEFAULT_END, tz="UTC") - pd.Timestamp(DEFAULT_START, tz="UTC")
    ).days / 365.25
    for rp in RETURN_PERIODS:
        df[f"P{rp}_per_yr"] = (df[f"P{rp}"] / n_yrs).round(3)
    return df


# ── State-level summary ────────────────────────────────────────────────────────
def build_summary(
    per_loc: pd.DataFrame,
    n_years: float,
) -> pd.DataFrame:
    """Build state-level summary comparing observed to expected frequencies."""
    n_locs = len(per_loc)
    rows = []
    for rp in RETURN_PERIODS:
        observed = int(per_loc[f"P{rp}"].sum())
        expected = (1.0 / rp) * n_locs * n_years
        ratio    = observed / expected if expected > 0 else float("nan")
        if np.isnan(ratio):
            status = "—"
        elif ratio < RATIO_OK_LO:
            status = "✗ under-count"
        elif ratio > RATIO_OK_HI:
            status = "✗ over-count"
        else:
            status = "✓"
        rows.append({
            "return_period":     f"P{rp}",
            "observed_6yr":      observed,
            "expected_6yr":      round(expected, 1),
            "ratio":             round(ratio, 2),
            "status":            status,
        })
    return pd.DataFrame(rows)


# ── Sensitivity analysis ───────────────────────────────────────────────────────
def run_sensitivity(
    mrms: pd.DataFrame,
    atlas14: pd.DataFrame,
    n_years: float,
    n_locs: int,
) -> pd.DataFrame:
    """Run all sensitivity parameter sets and return comparison table."""
    results: list[dict] = []
    for sp in SENSITIVITY_SETS:
        print(f"\n  Sensitivity [{sp.label}]  dry_thr={sp.dry_threshold}  dry_hrs={sp.dry_hours}")
        per_loc = count_events_per_location(
            mrms, atlas14,
            dry_threshold_in=sp.dry_threshold,
            dry_hours_min=sp.dry_hours,
        )
        row: dict = {"param_set": sp.label}
        row["dry_threshold_in"] = sp.dry_threshold
        row["dry_hours_min"]    = sp.dry_hours
        row["precip_thresh_mmhr"] = sp.precip_thresh
        row["kernel_km"]          = sp.kernel_km
        row["dist_km"]            = sp.dist_km
        row["jaccard"]            = sp.jaccard

        for rp in RETURN_PERIODS:
            obs  = int(per_loc[f"P{rp}"].sum())
            exp  = (1.0 / rp) * n_locs * n_years
            ratio = round(obs / exp, 2) if exp > 0 else float("nan")
            status = (
                "✓" if RATIO_OK_LO <= ratio <= RATIO_OK_HI
                else "✗" if not np.isnan(ratio) else "—"
            )
            row[f"P{rp}_obs"]    = obs
            row[f"P{rp}_ratio"]  = ratio
            row[f"P{rp}_status"] = status

        results.append(row)
    return pd.DataFrame(results)


# ── Diagnostic plots ───────────────────────────────────────────────────────────
def plot_histogram(summary: pd.DataFrame) -> plt.Figure:
    """Bar chart: observed vs expected events by return period."""
    rp_labels  = [r["return_period"] for _, r in summary.iterrows()]
    observed   = summary["observed_6yr"].values
    expected   = summary["expected_6yr"].values

    x   = np.arange(len(rp_labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_obs = ax.bar(x - w/2, observed, w, label="Observed",  color="steelblue")
    bars_exp = ax.bar(x + w/2, expected, w, label="Expected",  color="lightcoral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(rp_labels)
    ax.set_xlabel("Return Period")
    ax.set_ylabel("Event Count (analysis window)")
    ax.set_title("Indiana Precipitation Event Frequency: Observed vs Expected (Atlas 14)")
    ax.legend()

    # Annotate ratio
    for i, row in summary.iterrows():
        ratio = row["ratio"]
        color = "green" if RATIO_OK_LO <= ratio <= RATIO_OK_HI else "red"
        ax.text(x[i], max(observed[i], expected[i]) * 1.05,
                f"{ratio:.2f}×", ha="center", fontsize=8, color=color)

    fig.tight_layout()
    return fig


def plot_timeline(per_loc: pd.DataFrame, mrms: pd.DataFrame) -> plt.Figure:
    """Cumulative P10 events over time across all locations."""
    # Reconstruct annual event counts using MRMS year
    mrms["year"] = mrms["datetime_utc"].dt.year
    annual = mrms.groupby("year").size().reset_index(name="records")

    fig, ax = plt.subplots(figsize=(10, 4))
    # Count P10 exceedance events per year per location (rough proxy)
    cols_p10 = [c for c in per_loc.columns if c == "P10"]
    if cols_p10:
        total_p10 = int(per_loc["P10"].sum())
        n_yrs = mrms["year"].nunique()
        ax.bar(annual["year"], [total_p10 / n_yrs] * len(annual), color="steelblue", alpha=0.7)
        expected_p10 = (1/10) * len(per_loc)
        ax.axhline(expected_p10, color="red", ls="--", label=f"Expected P10 = {expected_p10:.0f}/yr")
        ax.set_ylabel("P10 Events per Year (mean)")
        ax.legend()

    ax.set_xlabel("Year")
    ax.set_title("P10 Event Rate Over Analysis Window")
    fig.tight_layout()
    return fig


# ── Spatial algorithm stubs (Steps 2–4) ───────────────────────────────────────
def _spatial_identify_storm_objects(
    grib2_grid: "np.ndarray",
    lats: "np.ndarray",
    lons: "np.ndarray",
    precip_thresh_mmhr: float = PRECIP_THRESH_MMHR,
    dilation_km: float = DILATION_RADIUS_KM,
) -> "np.ndarray":
    """Step 2+3: Threshold → 8-connected labeling → dilation-based clustering.

    Returns 2D integer label array (0 = background, 1..N = storm clusters).

    Requires: scipy.ndimage, skimage.morphology
    NOTE: Only called in --spatial mode when full GRIB2 grid is available.

    Baldwin et al. (2005): 8-connectivity avoids false fragmentation.
    Chang et al. (2016): binary dilation merges nearby rain cores.
    """
    try:
        from scipy.ndimage import binary_dilation, label, generate_binary_structure
        from skimage.morphology import disk
    except ImportError:
        raise RuntimeError("scipy and scikit-image required for --spatial mode")

    # Resolution: MRMS is 1 km/pixel in native projection
    pixel_km = 1.0
    radius_px = max(1, int(round(dilation_km / pixel_km)))

    # Step 2a: threshold
    precip_in_hr = grib2_grid / 25.4   # mm → in
    binary_mask  = (grib2_grid >= precip_thresh_mmhr).astype(np.uint8)

    # Step 2b: 8-connectivity labeling on original mask
    struct8 = generate_binary_structure(2, 2)
    labeled_orig, _ = label(binary_mask, structure=struct8)

    # Step 3a: dilate binary mask
    dilated = binary_dilation(binary_mask, footprint=disk(radius_px)).astype(np.uint8)

    # Step 3b: label dilated mask
    labeled_dilated, n_clusters = label(dilated, structure=struct8)

    # Step 3c: assign each original-mask pixel the dilated cluster ID
    # (background pixels in original mask keep label 0)
    result = np.where(binary_mask > 0, labeled_dilated, 0)
    return result


def _spatial_track_storms(
    clusters_t: "np.ndarray",
    clusters_t1: "np.ndarray",
    lats: "np.ndarray",
    lons: "np.ndarray",
    dist_km: float = TRACKING_DIST_KM,
    jaccard_min: float = JACCARD_MIN,
) -> dict[int, int]:
    """Step 4: Link clusters at time t to clusters at time t+1.

    Returns mapping {cluster_id_t1: storm_id} where storm_id is the
    persistent ID from time t (or a new ID if no match found).

    Prein et al. (2017), Murthy et al. (2015): centroid distance + Jaccard.
    """
    import math

    def _centroids(labeled):
        ids = [i for i in np.unique(labeled) if i > 0]
        out = {}
        for cid in ids:
            rows, cols = np.where(labeled == cid)
            out[cid] = (float(lats[rows.mean().astype(int)]),
                        float(lons[cols.mean().astype(int)]))
        return out

    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        a  = (math.sin((phi2-phi1)/2)**2
              + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2)
        return 2*R*math.asin(math.sqrt(max(0, min(1, a))))

    c_t  = _centroids(clusters_t)
    c_t1 = _centroids(clusters_t1)
    mapping: dict[int, int] = {}

    for id_t1, (lat1, lon1) in c_t1.items():
        best_id_t = None
        best_score = -1.0
        for id_t, (lat0, lon0) in c_t.items():
            dist = _haversine_km(lat0, lon0, lat1, lon1)
            if dist > dist_km:
                continue
            # Jaccard similarity
            mask_t  = (clusters_t  == id_t ).astype(int)
            mask_t1 = (clusters_t1 == id_t1).astype(int)
            inter   = int(np.sum(mask_t & mask_t1))
            union   = int(np.sum(mask_t | mask_t1))
            jaccard = inter / union if union > 0 else 0.0
            if jaccard >= jaccard_min and jaccard > best_score:
                best_score = jaccard
                best_id_t  = id_t
        mapping[id_t1] = best_id_t  # None means new storm
    return mapping


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", default=None,
                        help="CSV with location_id,lat,lon columns (default: pipeline stations)")
    parser.add_argument("--start",     default=DEFAULT_START, help="Analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default=DEFAULT_END,   help="Analysis end date (YYYY-MM-DD)")
    parser.add_argument("--spatial",   action="store_true",
                        help="Enable full GRIB2 spatial storm tracking (Steps 2–4). "
                             "Requires cfgrib, scipy, scikit-image. Very slow (~hours).")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip sensitivity analysis (faster)")
    args = parser.parse_args()

    if args.spatial:
        print("WARNING: --spatial mode processes raw MRMS GRIB2 from noaa-mrms-pds.")
        print("         This requires ~500 MB/day × 6 years of S3 reads and can take hours.")
        print("         The temporal algorithm (default) is recommended for first runs.\n")

    print("── Loading locations ─────────────────────────────────────────────────")
    locs = load_locations(args.locations)
    location_ids = locs["location_id"].astype(str).tolist()

    print("\n── Loading MRMS data ─────────────────────────────────────────────────")
    mrms = load_mrms(location_ids, args.start, args.end)

    # Intersect to locations that actually have MRMS data
    present = set(mrms["site_no"].unique())
    locs    = locs[locs["location_id"].isin(present)].reset_index(drop=True)
    location_ids = locs["location_id"].tolist()
    print(f"  Locations with MRMS coverage: {len(location_ids)}")

    print("\n── Loading Atlas 14 ──────────────────────────────────────────────────")
    atlas14 = load_atlas14(location_ids)

    # Intersect to locations with Atlas 14 data
    a14_sites = set(atlas14["site_no"].unique())
    locs      = locs[locs["location_id"].isin(a14_sites)].reset_index(drop=True)
    location_ids = locs["location_id"].tolist()
    print(f"  Locations with MRMS + Atlas 14: {len(location_ids)}")

    n_years = (
        pd.Timestamp(args.end, tz="UTC") - pd.Timestamp(args.start, tz="UTC")
    ).days / 365.25
    print(f"  Analysis window: {args.start} → {args.end}  ({n_years:.1f} yr)")

    print("\n── Step 1: Temporal event identification ─────────────────────────────")
    print("   (Restrepo-Posada & Eagleson 1982 / Giani et al. 2022)")
    print(f"   dry_threshold={DRY_THRESHOLD_IN} in/hr  |  dry_hours_min={DRY_HOURS_MIN}")
    per_loc = count_events_per_location(mrms, atlas14)

    print("\n── State-level summary ───────────────────────────────────────────────")
    summary = build_summary(per_loc, n_years)
    print(summary.to_string(index=False))

    print("\n── Saving outputs ────────────────────────────────────────────────────")
    upload_csv(summary, "indiana_precip_frequency_summary.csv")

    # Add lat/lon to per-location output
    out_per_loc = locs.merge(per_loc, left_on="location_id", right_on="location_id", how="left")
    upload_csv(out_per_loc, "indiana_precip_frequency_by_location.csv")

    # Histogram plot
    fig_hist = plot_histogram(summary)
    upload_fig(fig_hist, "summary_histogram.png")
    plt.close(fig_hist)

    # Timeline plot
    fig_time = plot_timeline(per_loc, mrms)
    upload_fig(fig_time, "event_timeline.png")
    plt.close(fig_time)

    # Sensitivity analysis
    if not args.no_sensitivity:
        print("\n── Sensitivity analysis ──────────────────────────────────────────────")
        print(f"   Running {len(SENSITIVITY_SETS)} parameter sets...")
        sens_df = run_sensitivity(mrms, atlas14, n_years, len(location_ids))
        upload_csv(sens_df, "sensitivity_analysis.csv")

    print("\n── Done ──────────────────────────────────────────────────────────────")
    print(f"  All outputs at s3://{BUCKET}/{S3_OUT}")


if __name__ == "__main__":
    main()
