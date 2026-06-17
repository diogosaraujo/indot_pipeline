#!/bin/bash
# ec2_setup.sh
#
# Run this once after SSH-ing into a fresh EC2 instance to install
# dependencies and launch the spatial precipitation analysis in a
# persistent tmux session.
#
# Recommended instance type: c5.4xlarge (16 vCPU, 32 GB) — us-east-1
#   On-demand cost: ~$0.68/hr.  Expected runtime: 15-25 hr → ~$10-17 total.
#   SPOT price is typically 30-50% lower (~$0.25-0.40/hr).
#
# Usage:
#   scp -i key.pem scripts/ec2_setup.sh ec2-user@<IP>:~/
#   ssh -i key.pem ec2-user@<IP>
#   bash ec2_setup.sh
#
# To reconnect to the running job after disconnection:
#   ssh -i key.pem ec2-user@<IP>
#   tmux attach -t precip
#
# To check progress without attaching:
#   ssh -i key.pem ec2-user@<IP> "tmux capture-pane -pt precip -S -50"

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
    numpy

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

echo "=== Starting tmux session 'precip' ==="
# Kill existing session if it exists (clean restart)
tmux kill-session -t precip 2>/dev/null || true

tmux new-session -d -s precip -x 220 -y 50

# Set logging to file AND terminal so you can tail the log after disconnect
LOG="$HOME/spatial_run.log"

tmux send-keys -t precip "cd $REPO_DIR" Enter
tmux send-keys -t precip \
    "python scripts/validate_storm_events.py 2>&1 | tee $LOG" \
    Enter

echo ""
echo "=== Job launched in tmux session 'precip' ==="
echo ""
echo "  Watch live:        tmux attach -t precip"
echo "  Detach from tmux:  Ctrl-B then D"
echo "  Tail log remotely: tail -f $LOG"
echo "  Check progress:    tmux capture-pane -pt precip -S -30"
echo ""
echo "  Resume after interruption:"
echo "    python scripts/validate_storm_events.py"
echo "    (automatically skips already-checkpointed days)"
echo ""
echo "  Aggregate without reprocessing:"
echo "    python scripts/validate_storm_events.py --aggregate-only"
echo ""
echo "  Force reprocess one day:"
echo "    python scripts/validate_storm_events.py --reprocess-date 2023-06-15"
