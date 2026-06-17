#!/bin/bash
# ec2_setup.sh
#
# Run this once after SSH-ing into a fresh EC2 instance to install all
# dependencies needed for the INDOT pipeline precipitation and streamflow
# analysis scripts.
#
# Recommended instance type: c5.4xlarge (16 vCPU, 32 GB) — us-east-1
#   On-demand cost: ~$0.68/hr.
#   SPOT price is typically 30-50% lower (~$0.25-0.40/hr).
#
# Usage:
#   scp -i key.pem scripts/ec2_setup.sh ec2-user@<IP>:~/
#   ssh -i key.pem ec2-user@<IP>
#   bash ec2_setup.sh

set -e

echo "=== System packages ==="
sudo yum update -y -q
# eccodes: required by cfgrib to read GRIB2 files
sudo yum install -y git tmux htop eccodes-devel 2>/dev/null || \
  sudo apt-get install -y git tmux htop libeccodes-dev 2>/dev/null || true

echo "=== Python environment ==="
pip install --quiet --upgrade pip
pip install --quiet \
    boto3 \
    s3fs \
    cfgrib \
    eccodes \
    xarray \
    scipy \
    scikit-image \
    pyarrow \
    pandas \
    numpy \
    matplotlib

echo "=== Clone / pull pipeline repo ==="
REPO_DIR="$HOME/indot_pipeline"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    # Replace with your actual repo URL
    git clone https://github.com/YOUR_ORG/indot_pipeline.git "$REPO_DIR"
fi

echo "=== AWS credentials check ==="
aws sts get-caller-identity || {
    echo "ERROR: AWS credentials not configured."
    echo "Run 'aws configure' or attach an IAM role to this instance."
    exit 1
}

echo ""
echo "=== Setup complete ==="
echo ""
echo "  cd $REPO_DIR"
echo ""
echo "  Example — run storm event analysis in a persistent tmux session:"
echo "    tmux new-session -d -s job"
echo "    tmux send-keys -t job 'python scripts/validate_storm_events.py 2>&1 | tee ~/run.log' Enter"
echo "    tmux attach -t job"
echo ""
echo "  Detach without stopping the job: Ctrl-B then D"
echo "  Reattach after disconnect:       tmux attach -t job"
echo "  Tail log remotely:               tail -f ~/run.log"
