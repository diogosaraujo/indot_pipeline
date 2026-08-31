"""Runtime configuration for the bridge-monitoring system.

Everything is driven by environment variables (set on the Lambda functions).
When run locally or on EC2 (the precompute path) the output bucket/prefix fall
back to the pipeline ``config.yaml`` so both halves share one source of truth.
"""
from __future__ import annotations

import os
from functools import lru_cache

REGION = os.environ.get("AWS_REGION", "us-east-1")

# ── Firing thresholds (return periods, years) ────────────────────────────────
# Per the operating decision: scour-critical bridges fire at the 10-yr; all
# other over-water bridges fire only at the 50-yr and above.  Every PDF still
# shows all three tiers (P10/P50/P100, Q10/Q50/Q100); these only gate alerts.
FIRE_RP_SCOUR = int(os.environ.get("MONITOR_FIRE_RP_SCOUR", "10"))
FIRE_RP_OTHER = int(os.environ.get("MONITOR_FIRE_RP_OTHER", "50"))
SEVERITY_RPS = [10, 50, 100]

# ── Event separation (matches scripts/08_trigger_analysis.py MERGE_GAP_HOURS) ─
DECLUSTER_GAP_HOURS = int(os.environ.get("MONITOR_DECLUSTER_GAP_HOURS", "24"))

# A bridge that stays continuously above threshold is ONE event and alerts once,
# when it first crosses. Keeping a sustained event visible after that is the
# daily summary's job, not a re-alert's — an hourly poller re-firing on a
# multi-day flood produces a stream of near-identical mails that trains the
# reader to ignore the one that is new. (2026-08-31 decision, replacing the
# short-lived REALERT_HOURS re-fire.)

# ── Daily summary ────────────────────────────────────────────────────────────
# Sent every morning regardless of whether anything fired: a summary that only
# arrives on bad days cannot distinguish "nothing happened" from "the monitor
# is dead", which is exactly the ambiguity that hid a 13-day outage.
DAILY_TZ = os.environ.get("MONITOR_DAILY_TZ", "America/New_York")
DAILY_SEND_HOUR = int(os.environ.get("MONITOR_DAILY_SEND_HOUR", "6"))    # local
# The window is the previous LOCAL calendar day, 00:00–24:00, so "yesterday"
# means the same thing to the reader as it does to the report.
DAILY_WINDOW_HOURS = int(os.environ.get("MONITOR_DAILY_WINDOW_HOURS", "24"))

# ── Rolling state window ─────────────────────────────────────────────────────
STATE_HOURS = int(os.environ.get("MONITOR_STATE_HOURS", "48"))       # hours kept
BACKFILL_MAX_HOURS = int(os.environ.get("MONITOR_BACKFILL_MAX_HOURS", "6"))
MRMS_SEARCH_BACK = int(os.environ.get("MONITOR_MRMS_SEARCH_BACK", "6"))
NWM_SEARCH_BACK = int(os.environ.get("MONITOR_NWM_SEARCH_BACK", "6"))

# ── Source staleness ─────────────────────────────────────────────────────────
# MRMS and NWM normally publish ~1 h behind valid time. When a NODD mirror falls
# behind, the poller keeps succeeding while silently evaluating old conditions —
# that is the failure this threshold makes visible. 3 h allows the usual latency
# plus a missed cycle without crying wolf.
STALE_WARN_HOURS = float(os.environ.get("MONITOR_STALE_WARN_HOURS", "3"))

# ── Email (Amazon SES) ───────────────────────────────────────────────────────
ALERT_SENDER = os.environ.get("MONITOR_ALERT_SENDER", "")            # SES-verified identity
ALERT_RECIPIENTS = [e.strip() for e in
                    os.environ.get("MONITOR_ALERT_RECIPIENTS", "").split(",") if e.strip()]

# ── Fan-out ──────────────────────────────────────────────────────────────────
ALERTER_FUNCTION = os.environ.get("MONITOR_ALERTER_FUNCTION", "indot-bridge-alerter")

# ── Public NOAA sources (anonymous S3, us-east-1) ────────────────────────────
MRMS_BUCKET = "noaa-mrms-pds"
MRMS_FOLDER = "MultiSensor_QPE_01H_Pass2_00.00"
NWM_BUCKET = "noaa-nwm-pds"
NWM_PRODUCT_TRIGGER = "analysis_assim_no_da"   # open-loop (gauge-free) -> drives the flow trigger
NWM_PRODUCT_DISPLAY = "analysis_assim"          # A&A (with DA) -> shown on the PDF

CFS_PER_CMS = 35.3146667

# ── Bridge dataset column names (from bridge_coverage_flags.parquet) ─────────
ASSET_COL = "Asset Name"
LAT_COL = "(16) Latitude:"
LON_COL = "(17) Longitude:"
SCOUR_COL = "scour_critical"
WATERWAY_COL = "over_waterway"


@lru_cache(maxsize=1)
def bucket_prefix() -> tuple[str, str]:
    """(output_bucket, output_prefix) from env, else config.yaml."""
    b = os.environ.get("MONITOR_BUCKET")
    if b:
        return b, os.environ.get("MONITOR_PREFIX", "v1/")
    import pathlib

    import yaml
    here = pathlib.Path(__file__).resolve()
    for cand in ("config.yaml", *[p / "config.yaml" for p in here.parents]):
        try:
            with open(cand) as f:
                cfg = yaml.safe_load(f)
            return cfg["aws"]["output_bucket"], cfg["aws"]["output_prefix"]
        except (FileNotFoundError, OSError):
            continue
    raise RuntimeError("Set MONITOR_BUCKET (and MONITOR_PREFIX) or provide config.yaml")


def keys() -> dict:
    """Canonical S3 keys used by the monitor, resolved against the output bucket."""
    b, p = bucket_prefix()
    return {
        "bucket": b,
        "prefix": p,
        "config": f"{p}monitor/bridge_monitor_config.parquet",
        "state_mrms": f"{p}monitor/state/mrms/",     # + {YYYYMMDDHH}.parquet
        "state_grid": f"{p}monitor/state/mrms_grid/",  # + {YYYYMMDDHH}.npz (IN subset)
        "state_nwm": f"{p}monitor/state/nwm/",       # + {YYYYMMDDHH}.parquet
        "alert_state": f"{p}monitor/alert_state.parquet",
        "alerts": f"{p}monitor/alerts/",             # archived PDFs
        "pending": f"{p}monitor/alerts/pending/",    # + {YYYYMMDDHH}.parquet (poller -> alerter)
        "daily": f"{p}monitor/daily/",               # + {YYYYMMDD}.pdf archived summaries
        "counties": f"{p}monitor/assets/in_counties.parquet",   # digest-map outlines (p07)
        "health": f"{p}monitor/health.json",         # last-known source-staleness state
        "flowlines": f"{p}monitor/assets/flowlines.parquet",   # river network (e02)
        "places": f"{p}monitor/assets/bridge_places.parquet",  # county/city/river (e07)
        # Gridded Atlas-14 24-h depths on the MRMS grid (p10). Static product.
        "atlas14_grid": f"{p}monitor/assets/atlas14_grid_24h.npz",
    }
