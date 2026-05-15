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
| NWM Retrospective v3.0 — streamflow, velocity per gauge | Parquet | 5–15 GB |
| NWM Analysis & Assimilation — streamflow, velocity, nudge per gauge | Parquet | 2–8 GB |
| NWM Open-Loop A&A — streamflow, velocity per gauge | Parquet | 2–8 GB |
| NWM stage (stage_m added to all three products) | Parquet (overwrites above) | no additional size |
| Aggregate trigger-skill figures (CSI / POD / FAR heatmaps, POD vs FAR scatter, Indiana map) | PNG (S3) | < 5 MB |
| Per-gauge CSI heatmaps | PNG (S3) | < 50 MB total |

NWM time coverage by product:

| Product | Period | Variables |
|---|---|---|
| Retrospective v3.0 | Feb 1979 – Dec 2023, hourly | streamflow, velocity |
| Analysis & Assimilation | ~Sep 2018 – present, hourly | streamflow, velocity, nudge |
| Open-Loop A&A | ~Sep 2018 – present, hourly | streamflow, velocity |

Script 10 stores only streamflow and velocity (plus nudge for A&A). Script 11 then adds `stage_m` to all three parquets by interpolating HAND-based Synthetic Rating Curves from `HYDRO_TBL_1D.nc` on `noaa-nwm-pds` — see caveat 8 below.

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
python scripts/11_derive_stage.py                # ~10-20 min — adds stage_m to all three NWM parquets
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

8. **`nwm.src_s3_key` must be set before running script 11.** Stage derivation is handled by script 11, which reads `HYDRO_TBL_1D.nc` from `noaa-nwm-pds`. Before running it, verify the exact key with:

   ```bash
   aws s3 ls s3://noaa-nwm-pds/nwm.20240101/domain/ --no-sign-request
   ```

   Then set `nwm.src_s3_key` in `config.yaml` (e.g. `nwm.20240101/domain/HYDRO_TBL_1D.nc`). Script 10 no longer writes `stage_m` at all — all three NWM parquets contain only `streamflow_cms` and `velocity_ms` (plus `nudge_cms` for A&A) until script 11 is run.

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
├── setup_ec2.sh                           <- on-VM: install miniforge, mamba env (runs on EC2)
├── setup-gcp-infra.ps1                    <- gcloud CLI: create GCS bucket, service account, VM (runs locally)
├── setup_gcp_vm.sh                        <- on-VM: install miniforge, mamba env, AWS CLI (runs on GCP VM)
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
    ├── 10_download_nwm.py                 <- NWM retrospective + A&A + Open-Loop (streamflow + velocity) → S3
    ├── 10_download_nwm_gcp.py             <- NWM A&A + Open-Loop (streamflow + velocity) → GCS
    └── 11_derive_stage.py                 <- adds stage_m to all three NWM parquets via SRC interpolation
```

---

## 9. Running script 10 on Google Cloud (NWM operational extraction)

The full NWM operational archive (Analysis & Assimilation + Open-Loop, Sep 2018 → present) lives at `gs://national-water-model` on Google Cloud Storage. Running the extraction on a GCP Compute Engine VM keeps all the heavy NetCDF reads within GCP — fast and free of egress charges. The retrospective (Feb 1979 – Dec 2023) is handled separately by the AWS script (`10_download_nwm.py`), which reads the public Zarr store on `noaa-nwm-retrospective-3-0-pds`.

After the GCP run finishes and its parquets land in S3, `11_derive_stage.py` is run on the AWS EC2 instance to add `stage_m` to all three products uniformly.

> **Why not run the operational extraction on AWS?** The `noaa-nwm-pds` bucket only keeps a rolling ~17-month window of files. The complete archive back to September 2018 exists only in `gs://national-water-model`. Running in-region on GCP avoids cross-cloud reads for ~130,000 hourly files and keeps egress cost near zero.

### 9.1 Cost estimate

| Resource | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| GCP `e2-standard-4` VM | $0.134 / hr | ~8 hours | **$1.07** |
| GCS output storage (2 parquets) | $0.020 / GB-mo | ~600 MB for 1 month | **$0.01** |
| GCS → S3 egress (optional push) | $0.09 / GB | ~600 MB | **$0.05** |
| **Total** | | | **≈ $1.15** |

All costs are well within the $300 free credit that new GCP accounts receive. Credit expires 90 days after account creation.

### 9.2 Prerequisites — GCP account and gcloud CLI

**GCP account**

1. Create a GCP account at `cloud.google.com`. New accounts receive **$300 free credit**, valid for 90 days.
2. In the [Google Cloud Console](https://console.cloud.google.com/projectcreate), create a new project and note the **Project ID** (e.g. `positive-harbor-496218-u6`). Project IDs are permanent and globally unique.
3. Enable billing on the project (required to use Compute Engine, even with free credits).

**gcloud CLI (on your local Windows machine)**

1. Go to `cloud.google.com/sdk/docs/install` and download the Windows installer (`.exe`).
2. Run the installer. When prompted, leave "Run `gcloud init`" checked.
3. Open a **new** PowerShell window (the installer updates `PATH` — existing windows won't see it) and verify:
   ```powershell
   gcloud --version
   ```
4. Authenticate and set your project:
   ```powershell
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

   Verify it took:
   ```powershell
   gcloud config get-value project
   ```

### 9.3 Fill in config_gcp.yaml

Open `config_gcp.yaml` in the project root and set the following fields:

| Key | Example value | Notes |
|---|---|---|
| `gcp.project` | `positive-harbor-496218-u6` | Your GCP project ID |
| `gcp.output_bucket` | `indot-nwm` | Globally unique GCS bucket name — must be lowercase |
| `gcp.output_prefix` | `v1/` | Key prefix for all outputs in GCS |
| `aws.output_bucket` | `indot-bridge-pipeline` | S3 bucket to push parquets to after each product finishes |
| `aws.output_prefix` | `v1/` | Must match the prefix used by the AWS pipeline |
| `aws.region` | `us-east-1` | AWS region where the S3 bucket lives |
| `aws.access_key_id` | *(your key)* | Leave blank to use the `AWS_ACCESS_KEY_ID` environment variable |
| `aws.secret_access_key` | *(your secret)* | Leave blank to use `AWS_SECRET_ACCESS_KEY` env var |

Setting the `aws.output_bucket` is recommended — the script automatically pushes each finished parquet to S3 as soon as it completes, so if the VM is stopped mid-run you won't lose any already-finished products.

Leave `aws.access_key_id` and `aws.secret_access_key` blank and configure them via `aws configure` on the VM instead if you prefer not to store credentials in the YAML file.

### 9.4 Create GCP resources

Open `setup-gcp-infra.ps1` in the project root and set the two variables at the top of the file:

```powershell
$PROJECT_ID  = "positive-harbor-496218-u6"
$BUCKET_NAME = "indot-nwm"
```

`BUCKET_NAME` must match `gcp.output_bucket` in `config_gcp.yaml`. The script auto-detects your `gcloud` installation so it works from any PowerShell window. Run from your **local machine**:

```powershell
.\setup-gcp-infra.ps1
```

If PowerShell blocks execution with a script policy error, run once to allow local scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The script is idempotent — safe to re-run if interrupted. It will:

1. Set the active gcloud project to `PROJECT_ID`
2. Enable the Compute Engine, Cloud Storage, and IAM APIs
3. Create the GCS output bucket in `us-central1` with public access blocked
4. Create a service account `indot-nwm-sa` and grant it `Storage Object Admin` on the bucket
5. Launch VM `indot-nwm-vm` (`e2-standard-4`, 4 vCPU / 16 GB RAM, 50 GB SSD, `us-central1-a`) with the service account attached

When it finishes it prints the exact SSH command to use. Verify the VM is running:

```powershell
gcloud compute instances list --project=YOUR_PROJECT_ID
```

### 9.5 Connect to the VM

The VM uses OS Login, so no key pair file is needed. Connect directly with:

```powershell
gcloud compute ssh indot-nwm-vm --zone=us-central1-a --project=YOUR_PROJECT_ID
```

If this is the first time connecting, gcloud will generate an SSH key pair automatically and upload your public key to the VM. Wait ~60 seconds after `setup_gcp_infra.sh` finishes before connecting to give the VM time to finish booting.

### 9.6 Copy the repo to the VM

Run this from your **local machine**, in the directory that contains `indot_pipeline/`:

```powershell
gcloud compute scp --recurse indot_pipeline/ indot-nwm-vm:~/ `
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

Alternatively, clone from GitHub inside the SSH session (requires a fine-grained Personal Access Token with read-only access to the repo — Settings → Developer settings → Personal access tokens → Fine-grained tokens):

```bash
git clone https://YOUR_PAT@github.com/diogosaraujo/indot_pipeline.git
```

### 9.7 Provision the Python environment

From inside the SSH session, change to the project root and run the setup script:

```bash
cd indot_pipeline
bash setup_gcp_vm.sh
```

This will (skipping any step already done if re-run):
1. Download and install Miniforge to `~/miniforge3`
2. Create the `indot` conda/mamba environment from `environment.yml`
3. Install `google-cloud-storage` into the environment
4. Install the AWS CLI v2 (used for optional manual S3 transfer)

When it finishes, the script has already run the shell init steps. Activate the environment for the current session:

```bash
source ~/.bashrc && mamba activate indot
```

On future SSH reconnections, `mamba activate indot` is all you need.

If you plan to use the S3 push via AWS credentials (rather than embedding them in `config_gcp.yaml`), configure them now:

```bash
aws configure
# Enter: access key ID, secret key, region (us-east-1), output format (json)
```

### 9.8 Run the GCP scripts

Script 01 builds the station inventory and writes it to GCS. Script 10 reads it, resolves COMIDs via NLDI, then extracts the two operational products.

```bash
# ~1 minute
python scripts/01_get_stations_gcp.py

# ~6-8 hours total
python scripts/10_download_nwm_gcp.py
```

**Recommended: run script 10 inside a persistent session** so it survives SSH disconnection:

```bash
# Using tmux (pre-installed on Ubuntu)
tmux new -s nwm
mamba activate indot
python scripts/10_download_nwm_gcp.py

# Detach: Ctrl-B then D
# Reattach later: tmux attach -t nwm
```

Script 10 logs progress to stdout. If `aws.output_bucket` is set in `config_gcp.yaml`, each parquet is pushed to S3 immediately after it finishes writing to GCS, so partial runs are not lost.

Expected runtimes:

| Step | Approx. time |
|---|---|
| Station inventory (script 01) | ~1 minute |
| COMID lookup via NLDI | ~5 minutes |
| A&A extraction (~67,000 files, Sep 2018 → present) | ~3–4 hours |
| Open-Loop extraction (~67,000 files, Sep 2018 → present) | ~3–4 hours |

To confirm the run succeeded, check the output files in GCS:

```bash
gsutil ls -l gs://YOUR_GCS_BUCKET/v1/nwm/
```

You should see `comid_locations.parquet`, `analysis_assim.parquet`, and `open_loop.parquet`.

### 9.9 Transfer outputs to AWS S3

**Option A — Automatic push (recommended, configured before the run)**

If `aws.output_bucket` is filled in `config_gcp.yaml`, the script pushes each parquet to S3 automatically. No manual step needed — skip to section 9.10.

**Option B — Manual transfer after the run**

From inside the SSH session, with AWS credentials configured (see section 9.7):

```bash
# Copy parquets from GCS to the VM's local disk
mkdir -p ~/nwm_outputs
gsutil -m cp "gs://YOUR_GCS_BUCKET/v1/nwm/*.parquet" ~/nwm_outputs/

# Push to S3
aws s3 cp --recursive ~/nwm_outputs/ "s3://YOUR_S3_BUCKET/v1/nwm/"
```

Verify the files arrived in S3:

```bash
aws s3 ls s3://YOUR_S3_BUCKET/v1/nwm/
```

Expected files: `comid_locations.parquet`, `analysis_assim.parquet`, `open_loop.parquet`.

### 9.10 Run script 11 on AWS to add stage

Once the operational parquets are in S3 (and after `10_download_nwm.py` has also run on AWS to produce `retrospective.parquet`), SSH to your EC2 instance and run script 11:

```bash
# Confirm all three source parquets exist
aws s3 ls s3://YOUR_S3_BUCKET/v1/nwm/

# Set the SRC key in config.yaml first if you haven't yet:
# nwm:
#   src_s3_key: "nwm.20240101/domain/HYDRO_TBL_1D.nc"
# Find the current path with:
aws s3 ls s3://noaa-nwm-pds/ --no-sign-request | grep "nwm\." | sort | tail -5
# Then:
aws s3 ls s3://noaa-nwm-pds/nwm.YYYYMMDD/domain/ --no-sign-request | grep HYDRO_TBL

python scripts/11_derive_stage.py   # ~10-20 min
```

Script 11 overwrites each parquet in place, adding a `stage_m` column derived by interpolating HAND-based Synthetic Rating Curves from `HYDRO_TBL_1D.nc` on `noaa-nwm-pds`.

### 9.11 Teardown

Stop the GCP VM when done (service account and GCS bucket remain):

```bash
gcloud compute instances stop indot-nwm-vm \
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

A stopped VM accrues no compute charges but its SSD disk continues to cost ~$0.17/month for 50 GB. To eliminate all ongoing cost, delete the VM:

```bash
gcloud compute instances delete indot-nwm-vm \
    --zone=us-central1-a --project=YOUR_PROJECT_ID
```

Once all outputs are confirmed in S3, delete the GCS bucket:

```bash
gsutil rm -r gs://YOUR_GCS_BUCKET
```

> **Billing alert:** Set a budget alert so you are notified before credits run out.
> In the Google Cloud Console go to **Billing → Budgets & alerts → Create budget**, set the amount to **$5**, and configure email notifications at 80% and 100%. You will receive an email well before this workload's ~$1.15 cost is exceeded.
