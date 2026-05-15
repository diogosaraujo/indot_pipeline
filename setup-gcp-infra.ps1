# setup-gcp-infra.ps1
#
# Creates all Google Cloud resources needed to run the NWM pipeline on GCP:
#   - Enables required APIs
#   - Creates the GCS output bucket
#   - Creates a service account with Storage Object Admin on that bucket
#   - Launches a Compute Engine VM with the service account attached
#
# Run from PowerShell on your local Windows machine.
# Requires: gcloud CLI installed and authenticated (gcloud auth login).

# --- CONFIGURATION ---------------------------------------------------
$PROJECT_ID   = "positive-harbor-496218-u6"
$BUCKET_NAME  = "indot-nwm"
# BUCKET_NAME must be globally unique, all lowercase, and match
# gcp.output_bucket in config_gcp.yaml.
# ---------------------------------------------------------------------

$REGION       = "us-central1"
$ZONE         = "us-central1-a"
$VM_NAME      = "indot-nwm-vm"
$SA_NAME      = "indot-nwm-sa"
$SA_EMAIL     = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
$MACHINE_TYPE = "e2-standard-4"
$DISK_SIZE    = "50GB"

# --- Locate gcloud ---------------------------------------------------
function Find-Gcloud {
    if (Get-Command gcloud -ErrorAction SilentlyContinue) { return "gcloud" }
    $candidates = @(
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:ProgramFiles\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "C:\google-cloud-sdk\bin\gcloud.cmd"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

$gcloud = Find-Gcloud
if (-not $gcloud) {
    Write-Host "ERROR: gcloud not found. Install the Google Cloud SDK from cloud.google.com/sdk and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Using gcloud: $gcloud"

# --- Step 1: Set active project --------------------------------------
Write-Host ""
Write-Host "[1/6] Setting active project: $PROJECT_ID"
& $gcloud config set project $PROJECT_ID
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not set project. Run '& `"$gcloud`" auth login' first." -ForegroundColor Red
    exit 1
}

# --- Step 2: Enable APIs ---------------------------------------------
Write-Host ""
Write-Host "[2/6] Enabling APIs (Compute, Storage, IAM)..."
& $gcloud services enable `
    compute.googleapis.com `
    storage.googleapis.com `
    iam.googleapis.com `
    --project=$PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to enable APIs." -ForegroundColor Red; exit 1 }
Write-Host "    Done."

# --- Step 3: Create GCS bucket ---------------------------------------
Write-Host ""
Write-Host "[3/6] GCS bucket: gs://$BUCKET_NAME"
& $gcloud storage buckets describe "gs://$BUCKET_NAME" --project=$PROJECT_ID 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Creating..."
    & $gcloud storage buckets create "gs://$BUCKET_NAME" `
        --project=$PROJECT_ID `
        --location=$REGION `
        --uniform-bucket-level-access `
        --public-access-prevention
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to create bucket." -ForegroundColor Red; exit 1 }
    Write-Host "    Created."
} else {
    Write-Host "    Already exists - skipping."
}

# --- Step 4: Create service account ----------------------------------
Write-Host ""
Write-Host "[4/6] Service account: $SA_NAME"
& $gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Creating..."
    & $gcloud iam service-accounts create $SA_NAME `
        --display-name="INDOT NWM pipeline" `
        --project=$PROJECT_ID
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to create service account." -ForegroundColor Red; exit 1 }
    Write-Host "    Created."
} else {
    Write-Host "    Already exists - skipping."
}

# --- Step 5: Grant Storage Object Admin ------------------------------
# Applied every run - safe to repeat, ensures correct permissions.
Write-Host ""
Write-Host "[5/6] Granting Storage Object Admin on gs://$BUCKET_NAME..."
& $gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" `
    --member="serviceAccount:$SA_EMAIL" `
    --role="roles/storage.objectAdmin"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to set IAM binding." -ForegroundColor Red; exit 1 }
Write-Host "    Done."

# --- Step 6: Launch VM -----------------------------------------------
Write-Host ""
Write-Host "[6/6] VM: $VM_NAME ($ZONE, $MACHINE_TYPE, $DISK_SIZE SSD)"
& $gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT_ID 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Creating..."
    & $gcloud compute instances create $VM_NAME `
        --project=$PROJECT_ID `
        --zone=$ZONE `
        --machine-type=$MACHINE_TYPE `
        --service-account=$SA_EMAIL `
        --scopes=cloud-platform `
        --image-family=ubuntu-2204-lts `
        --image-project=ubuntu-os-cloud `
        --boot-disk-size=$DISK_SIZE `
        --boot-disk-type=pd-ssd `
        --metadata=enable-oslogin=TRUE
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to create VM." -ForegroundColor Red; exit 1 }
    Write-Host "    Created."
} else {
    Write-Host "    Already exists - skipping."
}

Write-Host ""
Write-Host "========================================="
Write-Host " All GCP resources are ready!"
Write-Host " Project : $PROJECT_ID"
Write-Host " Bucket  : gs://$BUCKET_NAME"
Write-Host " VM      : $VM_NAME ($ZONE)"
Write-Host ""
Write-Host " Wait ~60 seconds for the VM to boot, then connect:"
Write-Host " & `"$gcloud`" compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID"
Write-Host ""
Write-Host " Next steps:"
Write-Host "   1. Copy the repo to the VM  (README section 9.6)"
Write-Host "   2. SSH in and run: bash setup_gcp_vm.sh"
Write-Host "   3. Run the pipeline scripts  (README section 9.8)"
Write-Host "========================================="
