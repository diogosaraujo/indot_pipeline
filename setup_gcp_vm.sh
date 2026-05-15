#!/usr/bin/env bash
# setup_gcp_vm.sh
#
# Run this ON the GCP VM (not your local machine) after SSHing in.
# Installs Miniforge, creates the mamba 'indot' environment, adds
# google-cloud-storage, and installs the AWS CLI (for optional S3 push).
#
# Usage (from inside the VM, in the project root):
#   bash setup_gcp_vm.sh

set -euo pipefail

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MINIFORGE_INSTALLER="/tmp/Miniforge3.sh"

# ── Miniforge ─────────────────────────────────────────────────────────────────
echo "==> Miniforge"
if [[ -f ~/miniforge3/bin/mamba ]]; then
    echo "    Already installed — skipping."
else
    echo "    Downloading..."
    wget -q -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"
    bash "$MINIFORGE_INSTALLER" -b -p ~/miniforge3
    rm -f "$MINIFORGE_INSTALLER"
    echo "    Installed."
fi

source ~/miniforge3/etc/profile.d/conda.sh

# ── mamba environment ──────────────────────────────────────────────────────────
echo "==> mamba environment 'indot'"
if conda env list | grep -q "^indot "; then
    echo "    Already exists — skipping."
else
    mamba env create -f environment.yml
    echo "    Created."
fi

# Activate so subsequent pip/install steps land in the right env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate indot

echo "==> Installing google-cloud-storage..."
pip install --quiet google-cloud-storage

# ── AWS CLI v2 (needed only if using the optional S3 push) ────────────────────
echo "==> AWS CLI v2"
if command -v aws &>/dev/null; then
    echo "    Already installed — skipping."
else
    echo "    Installing unzip..."
    sudo apt-get install -y -q unzip
    echo "    Downloading..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/awscliv2.zip /tmp/aws
    echo "    Installed."
fi

echo ""
echo "==> Done."
echo ""
echo "    Initialize your shell for mamba (one-time per VM):"
echo "      ~/miniforge3/bin/conda init bash"
echo "      mamba shell init --shell bash --root-prefix=~/miniforge3"
echo "      source ~/.bashrc"
echo "      mamba activate indot"
echo ""
echo "    Set your USGS API token:"
echo "      export API_USGS_PAT=your_token_here"
echo "      echo 'export API_USGS_PAT=your_token_here' >> ~/.bashrc"
