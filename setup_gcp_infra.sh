#!/usr/bin/env bash
# setup_gcp_infra.sh
#
# Creates all Google Cloud resources needed to run scripts 01 and 10 on GCP:
#   - Enables required APIs
#   - Creates a GCS output bucket
#   - Creates a service account with Storage Object Admin on that bucket
#   - Launches a Compute Engine VM with the service account attached
#
# Run from your local machine (not from the VM).
# Prerequisites: gcloud CLI installed and authenticated (see README section 9.2).

set -euo pipefail

# ─────────────── FILL THESE IN BEFORE RUNNING ─────────────────────────────────
PROJECT_ID=""     # your GCP project ID (e.g. "my-project-123456")
BUCKET_NAME=""    # globally unique GCS bucket name (e.g. "indot-nwm-abc123")
# ──────────────────────────────────────────────────────────────────────────────

REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="indot-nwm-vm"
SA_NAME="indot-nwm-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "$PROJECT_ID" || -z "$BUCKET_NAME" ]]; then
    echo "ERROR: Set PROJECT_ID and BUCKET_NAME at the top of this script before running."
    exit 1
fi

echo "==> Setting active project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "==> Enabling APIs (Compute, Storage, IAM)..."
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    iam.googleapis.com \
    --project="$PROJECT_ID"

# ── GCS output bucket ──────────────────────────────────────────────────────────
echo "==> GCS bucket: gs://$BUCKET_NAME"
if gsutil ls "gs://$BUCKET_NAME" &>/dev/null; then
    echo "    Already exists — skipping."
else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET_NAME"
    # Block public access so outputs are private by default
    gsutil pap set enforced "gs://$BUCKET_NAME"
    echo "    Created."
fi

# ── Service account ────────────────────────────────────────────────────────────
echo "==> Service account: $SA_NAME"
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    echo "    Already exists — skipping."
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="INDOT NWM pipeline" \
        --project="$PROJECT_ID"
    echo "    Created."
fi

echo "==> Granting Storage Object Admin on output bucket..."
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://$BUCKET_NAME"
echo "    Done."

# ── Compute Engine VM ──────────────────────────────────────────────────────────
echo "==> VM: $VM_NAME ($ZONE, e2-standard-4, 50 GB SSD)"
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    echo "    Already exists — skipping."
else
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="e2-standard-4" \
        --service-account="$SA_EMAIL" \
        --scopes="cloud-platform" \
        --image-family="ubuntu-2204-lts" \
        --image-project="ubuntu-os-cloud" \
        --boot-disk-size="50GB" \
        --boot-disk-type="pd-ssd" \
        --metadata="enable-oslogin=TRUE"
    echo "    Created."
fi

echo ""
echo "==> All resources ready."
echo ""
echo "    Connect to the VM with:"
echo "      gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "    Then copy the repo to the VM and run setup_gcp_vm.sh."
echo "    See README section 9 for the full step-by-step procedure."
