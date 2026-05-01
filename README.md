# Indiana Bridge Inspection Pipeline — Data Acquisition

End-to-end procedure for downloading USGS streamflow records, watershed delineations, StreamStats flood-frequency flows, and MRMS precipitation (point + watershed-aggregated) for every USGS streamgage in Indiana, executed on AWS.

This repository packages the workflow as a restart-safe AWS data pipeline. It is intended for teams building hydrologic context for bridge inspection, event screening, and watershed-based analysis in Indiana.

## Repository summary

- Domain: Indiana hydrology and bridge-screening data acquisition
- Runtime target: AWS EC2 plus S3 in `us-east-1`
- Data sources: USGS NWIS, USGS StreamStats, NOAA MRMS
- Primary outputs: Parquet, GeoJSON, and Zarr
- Execution style: sequential scripts with idempotent reruns

## What the repository does

- Builds an Indiana gauge inventory
- Pulls full-record instantaneous streamflow for qualifying gauges
- Delineates contributing watersheds with StreamStats
- Retrieves published and regression-based flood-frequency metrics
- Extracts MRMS precipitation at gauge points and across watersheds

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
| Parallelism | `concurrent.futures` or `dask.distributed` | Multi-process MRMS extraction |
| Output | `pyarrow`, `zarr` | Parquet for tabular, Zarr for gridded |

The `cfgrib` library wraps ECMWF's `eccodes` — that is a C dependency, easiest installed via `mamba install -c conda-forge eccodes`.

---

## 4. Cost estimate

All figures are us-east-1 on-demand pricing as of late April 2026 and are end-to-end estimates for one full pipeline run with the default config (single MRMS product: `QPE_01H_Pass2`).

| Resource | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| EC2 m5.2xlarge (8 vCPU, 32 GB) | $0.384 / hr | ~24 hours wall-clock | **$9.22** |
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

## 5. Step-by-step procedure

### 5.1 One-time AWS setup

1. Create an S3 bucket in us-east-1, e.g. `indot-bridge-pipeline-<your-id>`. Block public access.
2. Create an IAM role for EC2 (`EC2-INDOT-Pipeline`) with two managed policies:
   - `AmazonS3FullAccess` *scoped* to your bucket via a customer-inline policy, not the AWS-managed one.
   - Read-only S3 access to `arn:aws:s3:::noaa-mrms-pds` (it's public so this is more about keeping principles tidy than authorization).
3. Launch an EC2 instance:
   - AMI: Ubuntu 24.04 LTS (any recent LTS is fine)
   - Type: `m5.2xlarge`
   - Storage: 200 GB gp3 root volume
   - Region/AZ: us-east-1, any AZ
   - Attach the IAM role above
   - Open inbound port 22 only to your IP

### 5.2 Provision the Python environment

SSH in and run `setup_ec2.sh` from this repo (it installs miniforge, creates a `mamba` env named `indot`, and installs all dependencies including eccodes).

```bash
scp -r indot_pipeline/ ubuntu@<ec2-ip>:~/
ssh ubuntu@<ec2-ip>
cd ~/indot_pipeline
bash setup_ec2.sh
mamba activate indot
```

### 5.3 Configure pipeline parameters

Edit `config.yaml` to set your S3 bucket name, output prefix, MRMS product variant, and date range. Defaults are sensible for this project.

### 5.4 Run the pipeline scripts in order

Each script is independent and idempotent — re-running picks up where it left off.

```bash
python scripts/01_get_indiana_stations.py        # ~1 minute
python scripts/02_download_streamflow.py         # network-bound; instantaneous records can be large
python scripts/03_delineate_watersheds.py        # ~6-10 hours, StreamStats rate-limited
python scripts/04_get_flow_statistics.py         # ~2 hours, StreamStats rate-limited
python scripts/05_extract_mrms_nearest.py        # ~4-6 hours, parallelized
python scripts/06_extract_mrms_watershed.py      # ~10-16 hours, parallelized
```

Steps 03 and 04 cannot be meaningfully sped up by adding cores — StreamStats limits you to 4 concurrent requests. Steps 05 and 06 *do* scale with CPU; bigger instance = faster. Steps 02, 03, and 04 can run concurrently in three terminals if you want to overlap them.

### 5.5 Cost-saving teardown

When step 06 finishes:

```bash
# Confirm everything is in S3
aws s3 ls s3://indot-bridge-pipeline-<your-id>/ --recursive --summarize
# Stop or terminate the instance
```

Stopping preserves the EBS volume (cheap, ~$16/month for 200 GB). **Terminating** deletes it. If you're done with this dataset for now, terminate.

---

## 6. Important caveats specific to this project

1. **MRMS pre-2020 data is not on AWS.** The retrospective period of record here is roughly 2020-10-14 onward. For your INDOT framework that's a meaningful constraint: you can only attribute condition-rating drops to MRMS-derived events for inspection cycles whose inter-inspection windows fall entirely after that date. Earlier transitions can still be cross-referenced with USGS gauge data.

2. **StreamStats' published Q-statistics are not all there.** The Gage Statistics Service returns whatever has been published for that gauge, which varies. Expect coverage on the order of 50–80% of Indiana streamgages for full Q2–Q500 sets. Smaller and newer gauges often have only a partial set, or none. Script 04 records gracefully which gauges are missing which return periods rather than failing.

3. **The legacy StreamStats API is in sunset.** USGS announced deprecation for January 30, 2026. The current scripts still isolate the legacy `streamstatsservices` calls to one function per step, so migrating to the new `ss-delineate` / `ss-hydro` endpoints should be a focused follow-up.

4. **`dataretrieval` is also in transition.** As of its January 2026 release, the package's `waterdata` module wraps USGS's modernized Water Data APIs and is now used by this pipeline for station inventory and instantaneous streamflow retrieval. Legacy `nwis` still exists in the package, but steps 01 and 02 no longer depend on it.

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

## 7. Files in this project

```
indot_pipeline/
├── README.md                              <- you are here
├── config.yaml                            <- bucket names, dates, product
├── requirements.txt                       <- pip-installable deps
├── environment.yml                        <- mamba/conda env (preferred)
├── setup_ec2.sh                           <- one-shot EC2 provisioning
└── scripts/
    ├── utils.py                           <- shared helpers
    ├── 01_get_indiana_stations.py         <- Water Data API station inventory
    ├── 02_download_streamflow.py          <- instantaneous/unit values, full record
    ├── 03_delineate_watersheds.py         <- StreamStats watershed.geojson
    ├── 04_get_flow_statistics.py          <- gage stats Q2..Q500
    ├── 05_extract_mrms_nearest.py         <- MRMS at gauge point
    └── 06_extract_mrms_watershed.py       <- MRMS over watershed polygon
```
