# Bridge flood-alert monitor

Real-time, hourly monitoring of Indiana bridges-over-water. Each hour it pulls
the newest MRMS precipitation and NWM streamflow, checks every bridge against
its precomputed thresholds, and emails a PDF alert (maps + 24-h time series)
whenever a bridge's precip **or** streamflow trigger fires — de-duplicated on a
24-hour event separation so no bridge re-alerts within a day.

This is the operational sibling of the trigger study in `scripts/`. It reuses
that study's exact methods (Kirpich Tc accumulation, Atlas-14 depths, NWM
retrospective LP3 quantiles, `group_wet_events` 24-h declustering).

---

## What fires an alert

| Bridge class | Fires when… | Severity shown |
|---|---|---|
| **Scour-critical** (760) | trailing `round(Tc)`-h MRMS ≥ **P10** *or* NWM open-loop flow ≥ **Q10** | 10 / 50 / 100-yr |
| **All other over-water** (≈16,150) | ≥ **P50/Q50** (the 10-yr never fires these) | 50 / 100-yr |

- **Precip trigger** — trailing `round(Kirpich Tc)`-hour MRMS accumulation ≥ the
  Atlas-14 depth at that duration (per `scripts/08c_tc_trigger_analysis.py`).
- **Flow trigger** — NWM **open-loop** (`analysis_assim_no_da`) hourly streamflow
  ≥ the retro-LP3 `Q` from `scripts/04c` (the gauge-free operating point from
  `scripts/08f`). The PDF also shows **A&A** (`analysis_assim`, with DA) for context.
- **Event separation** — an alert is raised only when the current wet hour is
  **> 24 h** after the previous wet hour for that (bridge, trigger); this mirrors
  `group_wet_events` (`MERGE_GAP_HOURS = 24`). Nowcast only — no forecast.

Every PDF shows all of P10/P50/P100 and Q10/Q50/Q100 regardless of which tier fired.

---

## Architecture — one poller, not one Lambda per bridge

```
EventBridge (hourly, :55)
        │
        ▼
  POLLER Lambda  ── fetches ONCE:  MRMS grib (~0.7 MB) + NWM open-loop & A&A (~12.6 MB each)
        │            samples all ~16.9k bridges + all bridge COMIDs
        │            appends per-hour state slices, evaluates triggers, declusters
        │
        ├── async invoke ──►  ALERTER Lambda  (once per firing bridge)
        │                         builds PDF (2 maps + 2 time series), archives to S3,
        │                         emails via SES
        ▼
   alert_state.parquet (24-h dedup)     state/{mrms,nwm}/{YYYYMMDDHH}.parquet (48-h rolling)
```

One poller reads each CONUS file **once** and evaluates every bridge — 16,914
per-bridge Lambdas would each re-download the same 13 MB NWM file (~220 GB/hr of
redundant reads). Firing is rare, so PDF generation fans out only when needed.

### Files

```
monitor/
  monitor_common/         # baked into the Lambda image (self-contained)
    config.py             #   env-driven settings + S3 key layout
    s3io.py grib.py       #   S3 helpers; MRMS grid helpers (copied from utils.py)
    mrms.py nwm.py        #   live readers (point sampling; per-COMID streamflow+velocity)
    state.py              #   rolling per-hour slices + 24-h alert-dedup state
    triggers.py           #   trigger eval + 24-h declustering  ← core logic
    catalog.py figure.py  #   config loader; alert-PDF builder
  lambda_poller.py        # hourly fan-in handler
  lambda_alerter.py       # per-bridge PDF + SES handler
  precompute/             # RUN ONCE on EC2 (full conda env) → bridge_monitor_config.parquet
    p01_bridge_comid_tc.py    #   COMID + Kirpich Tc (NLDI + 3DEP)
    p02_atlas14.py            #   Atlas-14 depths (PFDS), deduped per ~110 m cell
    p03_retro_lp3.py          #   retro-LP3 Q10/Q50/Q100 per COMID (04c machinery)
    p04_assemble_config.py    #   merge → monitor/bridge_monitor_config.parquet
  Dockerfile env-lambda.yml
  deploy/  00_ses_setup 01_iam 02_build_push 03_deploy_lambdas 04_schedule
```

### S3 layout (under `s3://<bucket>/<prefix>`)

```
monitor/bridge_monitor_config.parquet     # the one table the Lambdas read
monitor/precompute/*.parquet              # intermediate precompute outputs
monitor/state/mrms/{YYYYMMDDHH}.parquet   # rolling 48-h slices (auto-pruned)
monitor/state/nwm/{YYYYMMDDHH}.parquet
monitor/alert_state.parquet               # last wet/alert hour per (bridge,trigger)
monitor/alerts/alert_<asset>_<YYYYMMDDHH>.pdf
```

---

## Run it

### 1. Precompute the thresholds (once, on the EC2 box with the conda `indot` env)

```bash
conda activate indot
cd indot_pipeline
python monitor/precompute/p01_bridge_comid_tc.py     # hours  (NLDI + 3DEP; resumable)
python monitor/precompute/p02_atlas14.py             # ~1 hr  (PFDS; deduped, resumable)
python monitor/precompute/p03_retro_lp3.py           # hours  (retrospective Zarr; batched)
python monitor/precompute/p04_assemble_config.py     # seconds → bridge_monitor_config.parquet
```

All four are checkpointed to S3 and safe to re-run (they skip completed work).
Re-run annually (or when the bridge inventory changes) to refresh thresholds.

### 2. Deploy the Lambdas

```bash
cd monitor/deploy
cp config.env.example config.env      # edit: ACCOUNT_ID, bucket, SES sender/recipients
source config.env
./00_ses_setup.sh          # verify SES sender (+ recipients while in the SES sandbox)
./01_iam.sh                # execution role (own bucket R/W, SES, invoke alerter, logs)
./02_build_push.sh         # docker build + push to ECR  (needs Docker + the repo checked out)
./03_deploy_lambdas.sh     # create/update poller + alerter from the image
./04_schedule.sh           # hourly EventBridge rule → poller

# smoke test
aws lambda invoke --function-name "$POLLER_FN" /tmp/out.json && cat /tmp/out.json
```

NOAA source buckets are read **anonymously**, so the IAM role only needs your own
bucket, SES, and invoke permissions.

---

## Cost estimate (us-east-1)

Reads from `noaa-mrms-pds` / `noaa-nwm-pds` are **free and in-region** (the bucket
owner pays; no data-transfer or request charge). Cost is essentially Lambda
GB-seconds. Measured 2026-07-29: MRMS grib ≈ 0.5–0.9 MB, NWM `channel_rt` ≈ 12.6 MB.

| Item | Basis | Per month | Per day |
|---|---|---|---|
| Poller Lambda | 3 GB × ~60 s × 720 runs | ~$2.1 | ~$0.07 |
| Alerter Lambda | 2 GB × 40 s × ~200 alerts | ~$0.3 | ~$0.01 |
| CloudWatch Logs | ~0.7 GB ingest | ~$0.35 | ~$0.01 |
| ECR image (~2.5 GB) | storage | ~$0.25 | ~$0.01 |
| S3 (state + PDFs) | <1 GB + requests | ~$0.05 | — |
| SES + EventBridge | ~200 emails, 720 triggers | ~$0.03 | — |
| **Total** | | **≈ $3–5/mo** (budget **$5–10** for wet months) | **≈ $0.15–0.30/day** |

**One-time precompute:** a few hours on the existing `m5.2xlarge` (~$2–15).

Cost is nearly independent of bridge count: the poller downloads the same two
files whether it evaluates 760 bridges or 16,914.

---

## Operational notes

- **Latency** — MRMS Pass2 (gauge-corrected) and NWM analysis publish ~1 h after
  valid time; the poller searches back up to 6 h for the newest available hour and
  runs at :55. It backfills up to `BACKFILL_MAX_HOURS` missing hours so the
  trailing-Tc accumulation and 24-h series stay complete.
- **State** — per-hour slice files are append-only (no hourly rewrite race) and
  pruned past `STATE_HOURS` (48). `alert_state.parquet` holds only bridges that
  have ever been wet, so it stays tiny.
- **Tuning** — all knobs are env vars on the functions (see `monitor_common/config.py`
  and `deploy/config.env.example`): firing RPs, state window, search-back, SES.
- **Bridges without a COMID / LP3 fit** get precip-only monitoring (their `Q*` are
  null and simply never fire). Bridges without Atlas-14 (fetch failed) get
  flow-only monitoring.
- **SES sandbox** — to email arbitrary recipients, request production access;
  otherwise every recipient must be verified. Domain (DKIM) verification is
  recommended for a real deployment.
