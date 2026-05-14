# Indiana Bridge Inspection Pipeline — Data Acquisition

End-to-end procedure for downloading USGS streamflow records, watershed delineations, StreamStats flood-frequency flows, and MRMS precipitation (point + watershed-aggregated) for every USGS streamgage in Indiana, executed on AWS.

This repository packages the workflow as a restart-safe AWS data pipeline. It is intended for teams building hydrologic context for bridge inspection, event screening, and watershed-based analysis in Indiana.

## Repository summary

- Domain: Indiana hydrology and bridge-screening data acquisition
- Runtime target: AWS EC2 plus S3 in `us-east-1`
- Data sources: USGS NWIS, USGS StreamStats, NOAA MRMS, NOAA National Water Model
- Primary outputs: Parquet, GeoJSON, and Zarr
- Execution style: sequential scripts with idempotent reruns

## What the repository does

- Builds an Indiana gauge inventory
- Pulls full-record instantaneous streamflow for qualifying gauges
- Delineates contributing watersheds with StreamStats
- Retrieves published and regression-based flood-frequency metrics
- Extracts MRMS precipitation at gauge points and across watersheds
- Downloads National Water Model streamflow, velocity, nudge, and stage for each gauge

## Why this repository exists

The project is organized to make a large, multi-source hydrologic acquisition workflow understandable, reproducible, and easy to resume after interruption. Configuration is centralized, each major step has its own script, and outputs are written to durable object storage rather than held locally.

---

## 1. What this pipeline produces

| Output | Format | Approx. size |
|---|---|---|
| Indiana gauge inventory | Parquet, GeoJSON | < 1 MB |
| Instantaneous streamflow time series, full record (per gauge) | Partitioned Parquet | Larger than daily values; depends on sensor interval and period of record |
| Watershed polygons (per gauge) | GeoJSON | 50–200 MB |
| Flood-frequency flows Q2…Q500 (per gauge) | Parquet | < 1 MB |
| MRMS at nearest pixel to each gauge, *per product* | Parquet | 1–3 GB / product |
| MRMS watershed-mean time series, *per product* | Parquet | 1–3 GB / product |
| MRMS pixel-level Zarr per watershed (QPE products only, sparse above lower-limit) | Per-gauge Zarr stores | 5–25 GB / product |
| NWM COMID location table (snap distance from USGS gauge) | Parquet | < 1 MB |
| NWM Retrospective v3.0 — streamflow, velocity, Head/stage per gauge | Parquet | 5–15 GB |
| NWM Analysis & Assimilation — streamflow, velocity, nudge, stage per gauge | Parquet | 2–8 GB |
| NWM Open-Loop A&A — streamflow, velocity, stage per gauge | Parquet | 2–8 GB |
| Aggregate trigger-skill figures (CSI / POD / FAR heatmaps, POD vs FAR scatter, Indiana map) | PNG (S3) | < 5 MB |
| Per-gauge CSI heatmaps | PNG (S3) | < 50 MB total |

NWM time coverage by product:

| Product | Period | Variables |
|---|---|---|
| Retrospective v3.0 | Feb 1979 – Dec 2023, hourly | streamflow, velocity, Head (stage direct) |
| Analysis & Assimilation | ~Sep 2018 – present, hourly | streamflow, velocity, nudge |
| Open-Loop A&A | ~Sep 2018 – present, hourly | streamflow, velocity |

Stage for Retrospective is the `Head` variable (water surface elevation, m NAVD88) output directly by NWM. Stage for A&A and Open-Loop is derived by interpolating the HAND-based Synthetic Rating Curves (SRC) stored in `HYDRO_TBL_1D.nc` — see caveat 8 below.

The MRMS section iterates over every entry in `cfg.mrms.products`. Default config has only `QPE_01H_Pass2` enabled. Uncomment ARI_01H..ARI_24H, QPEFFG_MAX, RQI_24H, and the other QPE variants in `config.yaml` to extract them too.

Time coverage:

- **USGS streamflow:** instantaneous/unit-value discharge period of record at each gauge.
- **MRMS (all products):** **14 October 2020 → present** only. The NOAA MRMS S3 bucket does not hold pre-2020 data. If pre-2020 MRMS is required, see the appendix on Iowa Environmental Mesonet — different code path, no free in-region transfer.
- **StreamStats flow stats:** as published by USGS in the gage-statistics database (typically updated through the most recent peak-flow analysis for each gauge).

---

## 2. Why AWS — and which region

The MRMS public dataset lives at `s3://noaa-mrms-pds`, hosted in **us-east-1**. Reading it from any other region or from outside AWS incurs egress charges and is dramatically slower. Every step in this pipeline should run on an EC2 instance in `us-east-1`, with output written to an S3 bucket also in `us-east-1`. That keeps all transfer free and keeps GRIB read latency in the millisecond range.

**Region: us-east-1 (N. Virginia). Do not deviate from this.**

---

## 3. Tools and dependencies

| Layer | Tool | Purpose |
|---|---|---|
| Compute | EC2 (m5.2xlarge or larger) | Run the pipeline |
| Storage | S3 | Outputs and intermediate artifacts |
| Identity | IAM role attached to EC2 | S3 access without long-lived keys |
| Python env | Miniforge / mamba | Manage native deps (eccodes, GDAL) |
| USGS API | `dataretrieval` | Water Data API streamflow + site metadata |
| StreamStats | `requests` + retries | Watershed + flow stats REST calls |
| Geospatial | `geopandas`, `shapely`, `rasterio`, `rioxarray` | Watershed handling, masking |
| MRMS read | `s3fs`, `xarray`, `cfgrib` (eccodes) | Stream GRIB2 directly from S3 |
| NWM read | `s3fs`, `xarray`, `zarr`, `h5py` | Zarr (retrospective) + HDF5 byte-range (operational) |
| Parallelism | `concurrent.futures` or `dask.distributed` | Multi-process MRMS/NWM extraction |
| Output | `pyarrow`, `zarr` | Parquet for tabular, Zarr for gridded |

The `cfgrib` library wraps ECMWF's `eccodes` — that is a C dependency, easiest installed via `mamba install -c conda-forge eccodes`.

---

## 4. Cost estimate

All figures are us-east-1 on-demand pricing as of late April 2026 and are end-to-end estimates for one full pipeline run with the default config (single MRMS product: `QPE_01H_Pass2`).

| Resource | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| EC2 m5.2xlarge (8 vCPU, 32 GB) — scripts 01–07 | $0.384 / hr | ~24 hours wall-clock | **$9.22** |
| EC2 r5.2xlarge (8 vCPU, 64 GB) — script 08 only | $0.504 / hr | ~1 hour | **$0.50** |
| EBS gp3 root + scratch, 200 GB | $0.08 / GB-mo | ~3 days = 0.1 mo | **$1.60** |
| S3 PUT requests | $0.005 / 1k | ~5,000 PUTs | **$0.03** |
| S3 GET requests (against MRMS) | $0.0004 / 1k | ~50,000 GETs | **$0.02** |
| S3 storage (outputs) | $0.023 / GB-mo | ~50 GB | **$1.15 / month ongoing** |
| Inter-region transfer | n/a | $0 (same region) | **$0** |
| Total one-time run | | | **≈ $11** |
| Ongoing storage of outputs | | | **≈ $1–2 / month** |

**Adding more MRMS products scales the cost roughly linearly with the MRMS step.** The MRMS extraction (steps 5–6) is ~80% of total run-time. Each additional product adds ~15–20 hours of m5.2xlarge time (~$6–8) and a few GB of S3 storage. Enabling all 10 products in the config catalog would put a full run at ~**$60–80** and ongoing storage at ~$5–8/mo.

**Variations:**
- Skip the per-watershed pixel extraction (just keep nearest-pixel + watershed-mean): drops storage substantially and run time by ~40%, total ≈ $6 for the default single-product run.
- Use `m5.4xlarge` to halve wall-clock to ~12 hours: total ≈ $13 (better human turnaround, marginally more $).
- Use Spot instances for the MRMS step (the only step long enough to benefit): roughly 65–70% off the EC2 portion. Net total ≈ $5 for default config. Risk of interruption is real — checkpoint per day-of-data.

The dominant cost driver is **how many hours your EC2 instance is on**, not data volume. Don't leave the instance running after the pipeline finishes.

---

## 5. Prerequisites — AWS CLI

The `.bat` helper scripts in this repository use the **AWS Command Line Interface (AWS CLI v2)**. Install it on your local **Windows** machine before running any of the scripts in section 6.

### Install AWS CLI v2 on Windows

1. Go to the official AWS CLI documentation page and download the **AWS CLI MSI installer for Windows (64-bit)**. Search for "Install or update to the latest version of the AWS CLI" on the AWS docs site to find the current installer link.
2. Run the downloaded `.msi` file and follow the installer prompts.
3. Open a **new** Command Prompt or PowerShell window (the installer updates `PATH` — existing windows won't see it).
4. Verify the installation:

   ```
   aws --version
   ```

   Expected output: `aws-cli/2.x.x Python/3.x.x Windows/...`

### Configure credentials

You need an IAM user with programmatic access. In the AWS Console go to **IAM → Users → *your user* → Security credentials → Create access key**. Then run:

```
aws configure
```

Enter the following when prompted:

| Field | Value |
|---|---|
| AWS Access Key ID | The key ID from the step above |
| AWS Secret Access Key | The matching secret |
| Default region name | `us-east-1` |
| Default output format | `json` |

Credentials are written to `~/.aws/credentials` and `~/.aws/config`. All `.bat` scripts in this repository read them automatically — no further configuration needed.

> **Tip:** Use a dedicated IAM user with `AdministratorAccess` (or a tighter policy scoped to EC2, S3, and IAM) rather than root account credentials.

---

## 6. Step-by-step procedure

### 6.1 One-time AWS setup

Run the following `.bat` scripts **in order** from a Command Prompt or PowerShell window on your local machine. Each script is idempotent — safe to re-run if interrupted.

**a. Create the S3 bucket**

Open `create-s3-bucket.bat` in a text editor and set `BUCKET_NAME` to a globally unique name (e.g. `indot-bridge-pipeline-<your-initials>`). Bucket names must be lowercase and contain only letters, numbers, and hyphens. Then run:

```bat
create-s3-bucket.bat
```

This creates the bucket in `us-east-1` and blocks all public access.

**b. Create the IAM role**

Open `create-iam-role.bat` and set `BUCKET_NAME` to the **exact same name** used above, then run:

```bat
create-iam-role.bat
```

This creates the `EC2-INDOT-Pipeline` IAM role with a trust policy that allows EC2 to assume it, attaches a customer-inline policy scoped to your bucket plus read-only access to `noaa-mrms-pds`, and creates the matching EC2 instance profile. Wait about 15 seconds after this completes before the next step — IAM changes take a moment to propagate.

**c. Launch the EC2 instance**

Open `launch-ec2.bat` and set `KEY_NAME` to an existing EC2 key pair in your account (create one in the AWS Console under **EC2 → Key Pairs** if needed), then run:

```bat
launch-ec2.bat
```

This creates a security group that allows SSH only from your current public IP, launches an `m5.2xlarge` Ubuntu 24.04 instance with a 200 GB gp3 root volume, attaches the `EC2-INDOT-Pipeline` instance profile, and prints the SSH connection command when the instance is ready.

### 6.2 Provision the Python environment

#### Authenticate to GitHub from EC2

Use a **GitHub Personal Access Token (PAT)** to clone over HTTPS — no SSH key setup required on the instance.

1. On GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**
2. Under *Repository access*, select your `indot_pipeline` repo
3. Under *Permissions*, set **Contents → Read-only**
4. Click **Generate token** and copy it (you won't see it again)

#### Connect to the instance

```powershell
# Run this from your local machine (use the IP printed by launch-ec2.bat)
ssh -i C:\Users\daraujo\Downloads\indot-pipeline-key.pem ubuntu@<ec2-ip>
```

#### Clone and set up the environment

```bash
# Clone using the token embedded in the HTTPS URL
git clone https://<TOKEN>@github.com/diogosaraujo/indot_pipeline.git
cd indot_pipeline

# Install miniforge, create the mamba env, and install all dependencies
bash setup_ec2.sh
```

`setup_ec2.sh` installs conda/mamba but does not modify your shell's startup file. After it finishes, run the following commands to activate the environment:

```bash
# Make conda available in the current shell
source ~/miniforge3/etc/profile.d/conda.sh

# Initialize conda for bash (one-time per instance — writes to ~/.bashrc)
~/miniforge3/bin/conda init bash

# Initialize mamba for bash (one-time per instance — required before mamba activate works)
mamba shell init --shell bash --root-prefix=~/miniforge3

# Reload shell to apply the initialization
source ~/.bashrc

# Activate the project environment
mamba activate indot
```

On future reconnections, `mamba activate indot` is all you need — the shell init only needs to run once per instance.

> If you prefer to copy a local checkout rather than clone from GitHub, you can `scp` it from your local machine before SSHing in:
> ```bash
> scp -r indot_pipeline/ ubuntu@<ec2-ip>:~/
> ```

### 6.3 Configure pipeline parameters

Edit `config.yaml` to set your S3 bucket name, output prefix, MRMS product variant, and date range. Defaults are sensible for this project.

### 6.4 USGS Water Data API token (required for script 02)

Without a token the USGS API applies a strict rate limit that causes 429 errors across the 297-site inventory. Register for a free token at the USGS Water Data portal (`api.waterdata.usgs.gov`) under **My Account → API Tokens**.

Once you have the token, set it on EC2 before running script 02:

```bash
# Current session only
export API_USGS_PAT=your_token_here

# Persist across reconnections
echo 'export API_USGS_PAT=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

The token is injected as a Bearer Authorization header into every `dataretrieval` request automatically — no other changes needed.

### 6.5 Run the pipeline scripts in order

Each script is independent and idempotent — re-running picks up where it left off.

```bash
python scripts/01_get_indiana_stations.py        # ~1 minute
python scripts/02_download_streamflow.py         # network-bound; instantaneous records can be large
python scripts/03_delineate_watersheds.py        # ~6-10 hours, NLDI rate-limited
python scripts/04_get_flow_statistics.py         # ~2 hours, StreamStats rate-limited
python scripts/05_extract_mrms_nearest.py        # ~4-6 hours, parallelized
python scripts/06_extract_mrms_watershed.py      # ~10-16 hours, parallelized
python scripts/07_extract_atlas14.py             # ~1 hour, NOAA PFDS rate-limited
python scripts/04b_regression_flows.py           # ~2 minutes, fills missing Q10/Q50 via Rao 2005
```

Scripts 08 and 09 require a separate step each — see sections 6.6 and 6.7 below.

```bash
python scripts/10_download_nwm.py                # ~1 hr (retrospective) + ~4 hrs each for A&A and Open-Loop
```

Steps 03 and 04 cannot be meaningfully sped up by adding cores — StreamStats limits you to 4 concurrent requests. Steps 05 and 06 *do* scale with CPU; bigger instance = faster. Steps 02, 03, and 04 can run concurrently in three terminals if you want to overlap them.

### 6.6 Run script 08 — instance resize required

Script 08 (`08_trigger_analysis.py`) loads the full MRMS and USGS streamflow records into memory simultaneously. Peak RAM exceeds what the standard `m5.2xlarge` (32 GB) can hold, so it must be run on a larger instance. Follow this procedure from your **local Windows machine**:

**Step 1 — Upsize the instance**

```bat
upsize-ec2.bat
```

This stops the running `indot-pipeline` instance, changes its type to `r5.2xlarge` (64 GB RAM), and restarts it. It prints the new SSH connection command when the instance is ready (~3–4 minutes).

**Step 2 — SSH in and run script 08**

```powershell
ssh -i C:\Users\daraujo\Downloads\indot-pipeline-key.pem ubuntu@<new-ip>
```

```bash
cd indot_pipeline
source ~/miniforge3/etc/profile.d/conda.sh
mamba activate indot
python scripts/08_trigger_analysis.py            # ~45-60 minutes
```

**Step 3 — Downsize the instance back**

Once script 08 finishes, exit the SSH session and run from your local machine:

```bat
downsize-ec2.bat
```

This stops the instance, restores it to `m5.2xlarge`, and restarts it. The instance returns to its normal cost tier (~$0.384/hr vs $0.504/hr for the r5.2xlarge).

> **Cost note:** the upsize window costs approximately $0.50 in EC2 time for a single script 08 run. Do not leave the instance running as an r5.2xlarge after the script finishes.

### 6.7 Run script 09 — figures

Script 09 (`09_figures.py`) reads `analysis/trigger_analysis.parquet` produced by script 08 and writes all figures directly to S3 — no display or local file system needed. Run it on the same instance used for scripts 01–07 (the `m5.2xlarge` is sufficient; the script is CPU-light).

```bash
python scripts/09_figures.py                     # ~5-10 minutes
```

Figures are written to `s3://<bucket>/<prefix>analysis/figures/` and `analysis/figures/stations/`. Download them locally with:

```bash
aws s3 sync s3://indot-bridge-pipeline-<your-id>/v1/analysis/figures/ ./figures/
```

**Aggregate figures** (one set per flow threshold Q10 / Q50):

| File | Description |
|---|---|
| `csi_heatmap_Q{rp}.png` | CSI by duration × precip return period (pooled counts across all stations) |
| `pod_heatmap_Q{rp}.png` | POD by duration × precip return period (pooled counts) |
| `far_heatmap_Q{rp}.png` | FAR by duration × precip return period (pooled counts) |
| `pod_vs_far_Q{rp}.png` | POD vs FAR scatter coloured by accumulation duration (pooled counts) |
| `best_csi_per_station.png` | Bar chart of best achievable CSI per gauge |
| `best_combo_per_station.png` | Horizontal bar chart — best (duration / precip RP / flow threshold) per gauge |
| `map_best_csi.png` | Indiana map of gauges coloured by best CSI |

**Per-gauge figures** (`analysis/figures/stations/`):

| File | Description |
|---|---|
| `{site_no}_csi.png` | CSI heatmap for Q10 and Q50 side by side |

> **Note on pooled vs per-station metrics:** The aggregate heatmaps (CSI / POD / FAR) are computed from globally pooled TP/FP/FN/TN counts across all stations, not station averages. This avoids giving equal weight to short-record gauges and gauges with few events. The per-station bar charts and map still use each station's individual best-CSI value.

### 6.8 Cost-saving teardown

When step 06 finishes:

```bash
# Confirm everything is in S3
aws s3 ls s3://indot-bridge-pipeline-<your-id>/ --recursive --summarize
# Stop or terminate the instance
```

Stopping preserves the EBS volume (cheap, ~$16/month for 200 GB). **Terminating** deletes it. If you're done with this dataset for now, terminate.

---

## 7. Important caveats specific to this project

1. **MRMS pre-2020 data is not on AWS.** The retrospective period of record here is roughly 2020-10-14 onward. For your INDOT framework that's a meaningful constraint: you can only attribute condition-rating drops to MRMS-derived events for inspection cycles whose inter-inspection windows fall entirely after that date. Earlier transitions can still be cross-referenced with USGS gauge data.

2. **StreamStats' published Q-statistics are not all there.** The Gage Statistics Service returns whatever has been published for that gauge, which varies. Expect coverage on the order of 50–80% of Indiana streamgages for full Q2–Q500 sets. Smaller and newer gauges often have only a partial set, or none. Script 04 records gracefully which gauges are missing which return periods rather than failing.

3. **The legacy StreamStats API is in sunset.** USGS announced deprecation for January 30, 2026. The current scripts still isolate the legacy `streamstatsservices` calls to one function per step, so migrating to the new `ss-delineate` / `ss-hydro` endpoints should be a focused follow-up.

4. **`dataretrieval` is also in transition.** As of its January 2026 release, the package's `waterdata` module wraps USGS's modernized Water Data APIs and is now used by this pipeline for station inventory and instantaneous streamflow retrieval. Legacy `nwis` still exists in the package, but steps 01 and 02 no longer depend on it.

8. **NWM SRC file path must be confirmed before stage is available for operational products.** The HAND-based Synthetic Rating Curves live in `HYDRO_TBL_1D.nc`, which is bundled with the NWM domain files on `noaa-nwm-pds`. Before running script 10, verify the exact key with:

   ```bash
   aws s3 ls s3://noaa-nwm-pds/nwm.20240101/domain/ --no-sign-request
   ```

   Then set `nwm.src_s3_key` in `config.yaml` (e.g. `nwm.20240101/domain/HYDRO_TBL_1D.nc`). Until that key is set, `stage_m` is written as NaN for A&A and Open-Loop — Retrospective `stage_m` comes from the `Head` variable and is unaffected.

9. **NWM operational archive depth.** The `noaa-nwm-pds` bucket holds NWM operational output going back to approximately September 2018 (NWM v2.0). Earlier files may be missing or in a different format. Script 10 skips missing hourly files without failing so the parquet is simply sparse for gaps in the archive.

10. **NWM script 10 runtime.** The retrospective uses the public Zarr store and completes in ~1 hour. The two operational products (A&A and Open-Loop) process ~67,000 hourly NetCDF files each using `h5py` byte-range reads; expect ~3–5 hours per product with 16 workers. Run both on the same `m5.2xlarge` used for MRMS — no resize needed.

5. **MRMS product catalog and units.** `cfg.mrms.products` is a list — steps 05 and 06 loop over it. The default config enables only `QPE_01H_Pass2` (gauge-bias-corrected hourly accumulation, the standard reference for hydrologic studies). The other supported products are commented in:
   - `QPE_01H_Pass1`, `QPE_03H_Pass2`, `QPE_24H_Pass2` — additional QPE accumulation windows.
   - `ARI_01H..ARI_24H` — Average Recurrence Interval, in years. Climatology-aware: an ARI value of 25 means *this rainfall is a 25-year event at this location*. Likely the most useful single trigger metric for the INDOT comparison because it normalizes for spatial variation in climatological intensity (a fixed 2.5"/24h means very different things in northern vs. southern Indiana).
   - `QPEFFG_MAX` — QPE as a percentage of Flash Flood Guidance, another physically-motivated trigger.
   - `RQI_24H` — Radar Quality Index. Required for QC of any QPE-based trigger; tells you which pixels had reliable radar coverage on a given day.

   QPE products are stored in **inches** by default (`mrms.units: in`) to match HNTB's existing data lake and INDOT's 2.5"/24h trigger convention. Set `mrms.units: mm` for native units. ARI/QPEFFG/RQI are not affected by the units toggle.

6. **Watershed-mean vs. all-pixels-in-watershed.** Step 06 produces both, per product:
   - `watershed_mean.parquet` — area-weighted mean over **all** pixels in the watershed (no lower-limit filtering — this matters; the mean is statistically faithful regardless of how dry the storm was). This is the daily-driver for trigger comparison.
   - `per_watershed_zarr/{site_no}.zarr/` — pixel-level time series, **sparse**: only pixels at or above `product.lower_limit` are stored. For QPE_01H_Pass2 this is 0.1", which is the same threshold HNTB's existing pipeline uses. Per-pixel Zarr is generated only for QPE-like products even when `per_pixel_zarr: true` globally — for ARI/RQI/QPEFFG, the watershed mean carries essentially all the useful signal.

   If storage cost is a concern, set `mrms.per_pixel_zarr: false` to skip pixel-level output.

---

## 8. Files in this project

```
indot_pipeline/
├── README.md                              <- you are here
├── config.yaml                            <- bucket names, dates, product (AWS run)
├── config_gcp.yaml                        <- GCP project, bucket, NWM SRC key (GCP run)
├── requirements.txt                       <- pip-installable deps
├── environment.yml                        <- mamba/conda env (preferred)
├── setup_ec2.sh                           <- one-shot EC2 provisioning
├── setup_gcp_infra.sh                     <- gcloud CLI: create GCS bucket, service account, VM
├── setup_gcp_vm.sh                        <- on-VM: install miniforge, mamba env, AWS CLI
├── create-s3-bucket.bat                   <- AWS CLI: create output S3 bucket
├── create-iam-role.bat                    <- AWS CLI: create EC2 IAM role + instance profile
├── launch-ec2.bat                         <- AWS CLI: launch EC2 instance (m5.2xlarge)
├── upsize-ec2.bat                         <- AWS CLI: stop instance, resize to r5.2xlarge (64 GB), restart
├── downsize-ec2.bat                       <- AWS CLI: stop instance, restore to m5.2xlarge, restart
└── scripts/
    ├── utils.py                           <- shared helpers
    ├── 01_get_indiana_stations.py         <- Water Data API station inventory → S3
    ├── 01_get_stations_gcp.py             <- Water Data API station inventory → GCS
    ├── 02_download_streamflow.py          <- instantaneous/unit values, full record
    ├── 03_delineate_watersheds.py         <- NLDI watershed delineation
    ├── 04_get_flow_statistics.py          <- gage stats Q2..Q500
    ├── 05_extract_mrms_nearest.py         <- MRMS at gauge point
    ├── 06_extract_mrms_watershed.py       <- MRMS over watershed polygon
    ├── 07_extract_atlas14.py              <- NOAA Atlas 14 precipitation frequency
    ├── 04b_regression_flows.py            <- Rao 2005 regression fill for missing Q10/Q50
    ├── 08_trigger_analysis.py             <- precipitation trigger vs. streamflow analysis
    ├── 09_figures.py                      <- summary figures (heatmaps, maps, per-gauge CSI)
    ├── 10_download_nwm.py                 <- NWM retrospective + A&A + Open-Loop per gauge → S3
    └── 10_download_nwm_gcp.py             <- NWM retrospective + A&A + Open-Loop per gauge → GCS
```

---

## 9. Running script 10 on Google Cloud (NWM extraction)

The National Water Model operational archive lives at `gs://national-water-model` on Google Cloud Storage. Running the NWM extraction step on a GCP Compute Engine VM means all the heavy operational reads stay within GCP, where they are fast and free. The retrospective Zarr store (`s3://noaa-nwm-retrospective-3-0-pds`) is on AWS Open Data — reading it from a GCP VM is free under the Open Data Sponsorship Programme.

A single run of both operational products plus the retrospective completes in roughly **8–10 hours** on an `e2-standard-4` VM (~$0.134/hr), keeping total GCP cost well inside the **$300 free credit** new accounts receive.

> **Why not run script 10 on AWS?** The `noaa-nwm-pds` bucket only retains a ~1-year rolling window of operational data. The full archive back to September 2018 is in `gs://national-water-model`. Reading ~67,000 NetCDF files across clouds is slower than reading them in-region; doing it on GCP keeps latency low and costs negligible.

### 9.1 Cost estimate

| Resource | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| GCP e2-standard-4 VM | $0.134 / hr | ~10 hours | **$1.34** |
| GCS output storage (parquets) | $0.020 / GB-mo | ~800 MB for 1 month | **$0.02** |
| GCS → S3 egress (optional push) | $0.09 / GB | ~800 MB | **$0.07** |
| Total | | | **≈ $1.50** |

All costs are well within the $300 free credit. Credit expires 90 days after account creation.

### 9.2 Prerequisites — GCP account and gcloud CLI

**GCP account**

1. Create a GCP account at `cloud.google.com`. New accounts receive **$300 free credit**, valid for 90 days.
2. In the [Google Cloud Console](https://console.cloud.google.com/projectcreate), create a new project and note the **Project ID** (e.g. `my-project-123456`). Project IDs are permanent and globally unique.
3. Enable billing on the project (required to use Compute Engine, even with free credits).

**gcloud CLI (on your local Windows machine)**

1. Search for "Google Cloud CLI installer" on `cloud.google.com/sdk/docs/install` and download the Windows installer (`.exe`).
2. Run the installer. When prompted, leave "Run `gcloud init`" checked.
3. Open a **new** PowerShell window and verify:
   ```
   gcloud --version
   ```
4. Authenticate and set your project:
   ```
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

### 9.3 Fill in config_gcp.yaml

Open `config_gcp.yaml` in the project root and fill in:

| Key | Example value | Notes |
|---|---|---|
| `gcp.project` | `my-project-123456` | Your GCP project ID |
| `gcp.output_bucket` | `indot-nwm` | Globally unique GCS bucket name |
| `gcp.nwm_src_key` | `nwm.20240101/domain/HYDRO_TBL_1D.nc` | Path to the SRC file — see below |
| `aws.output_bucket` | `indot-bridge-pipeline` | Optional: S3 bucket to push outputs to |
| `aws.access_key_id` | *(your key)* | Optional: leave blank to use `AWS_ACCESS_KEY_ID` env var |
| `aws.secret_access_key` | *(your secret)* | Optional: leave blank to use `AWS_SECRET_ACCESS_KEY` env var |

**Finding `nwm_src_key`** — run this from any machine that has `gsutil` or the gcloud CLI:

```bash
gsutil ls "gs://national-water-model/nwm.*/domain/HYDRO_TBL_1D.nc" | sort | tail -1
```

Copy the output path and remove the `gs://national-water-model/` prefix. Example result: `nwm.20240101/domain/HYDRO_TBL_1D.nc`.

Until this key is set, `stage_m` will be written as NaN for A&A and Open-Loop. Retrospective `stage_m` comes from the `Head` variable directly and is not affected.

### 9.4 Create GCP resources

Open `setup_gcp_infra.sh`, set `PROJECT_ID` and `BUCKET_NAME` at the top to match your `config_gcp.yaml`, then run from your local machine:

```bash
bash setup_gcp_infra.sh
```

This script (idempotent — safe to re-run):
- Enables the Compute, Storage, and IAM APIs on your project
- Creates the GCS output bucket in `us-central1`
- Creates a service account (`indot-nwm-sa`) with `Storage Object Admin` on the output bucket
- Launches a Compute Engine VM (`indot-nwm-vm`, `e2-standard-4`, `us-central1-a`, 50 GB SSD) with the service account attached

Wait about 60 seconds for the VM to finish booting, then connect:

```bash
gcloud compute ssh indot-nwm-vm --zone=us-central1-a --project=YOUR_PROJECT_ID
```

### 9.5 Copy the repo to the VM

From your **local machine** (in the directory that contains `indot_pipeline/`):

```bash
# Option A: copy a local checkout (no GitHub access required)
gcloud compute scp --recurse indot_pipeline/ indot-nwm-vm:~/ \
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

```bash
# Option B: clone from GitHub (run from inside the SSH session)
git clone https://YOUR_PAT@github.com/diogosaraujo/indot_pipeline.git
```

For option B, create a fine-grained GitHub Personal Access Token with read-only access to the repo (Settings → Developer settings → Personal access tokens → Fine-grained tokens).

### 9.6 Provision the Python environment on the VM

From inside the SSH session, in the project root:

```bash
bash setup_gcp_vm.sh
```

When it finishes, initialize the shell and activate the environment:

```bash
~/miniforge3/bin/conda init bash
mamba shell init --shell bash --root-prefix=~/miniforge3
source ~/.bashrc
mamba activate indot
```

Set the USGS API token (same requirement as the AWS path — see section 6.4):

```bash
export API_USGS_PAT=your_token_here
echo 'export API_USGS_PAT=your_token_here' >> ~/.bashrc
```

On future SSH reconnections, `mamba activate indot` is all you need.

### 9.7 Run the GCP scripts

```bash
# ~1 minute — writes station inventory to GCS
python scripts/01_get_stations_gcp.py

# ~8-10 hours — retrospective + A&A + Open-Loop
python scripts/10_download_nwm_gcp.py
```

Script 10 logs progress to stdout and automatically skips the retrospective if `nwm/retrospective.parquet` already exists in the GCS bucket. It is safe to interrupt and resume — the retrospective is the longest step and benefits most from this.

Expected runtimes:

| Step | Approx. time |
|---|---|
| Station inventory (script 01) | ~1 minute |
| COMID lookup via NLDI | ~5 minutes |
| Retrospective Zarr load | ~30–60 minutes |
| A&A extraction (~67k files) | ~3–4 hours |
| Open-Loop extraction (~67k files) | ~3–4 hours |

### 9.8 Transfer outputs to AWS S3

**Option A — Automatic push during extraction (recommended)**

Fill in the `aws` section of `config_gcp.yaml` before running script 10. The script pushes each finished parquet to your S3 bucket immediately after writing it to GCS. Total egress is ~800 MB (~$0.07).

**Option B — Manual transfer after the run**

From the VM (with AWS credentials configured via `aws configure` or environment variables):

```bash
# Download all NWM parquets from GCS to the VM
gsutil -m cp "gs://YOUR_GCS_BUCKET/v1/nwm/*.parquet" ./nwm_outputs/

# Upload to S3
aws s3 cp --recursive ./nwm_outputs/ "s3://YOUR_S3_BUCKET/v1/nwm/"
```

### 9.9 Teardown

Stop the VM when done (the service account and bucket stay in place):

```bash
gcloud compute instances stop indot-nwm-vm \
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

A stopped VM accrues no compute charges but keeps its disk (~$0.006/hr for 50 GB SSD). To eliminate all ongoing cost, delete the VM:

```bash
gcloud compute instances delete indot-nwm-vm \
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

Once outputs are confirmed in S3, delete the GCS bucket too:

```bash
gsutil rm -r gs://YOUR_GCS_BUCKET
```

> **Billing alert:** Set a budget alert so you are notified before credits run out.
> In the Google Cloud Console go to **Billing → Budgets & alerts → Create budget**, set the amount to $5, and configure email notifications at 80% and 100%. You will receive an email well before this workload's ~$1.50 cost is exceeded.
