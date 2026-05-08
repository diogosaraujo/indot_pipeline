#!/usr/bin/env bash
# One-shot provisioning for a fresh Ubuntu 24.04 EC2 instance.
# Installs miniforge, creates the `indot` env, and installs all deps.
set -euo pipefail

# System packages
sudo apt-get update -y
sudo apt-get install -y build-essential wget curl git unzip

# AWS CLI v2 (not available via apt on Ubuntu 24.04)
if ! command -v aws &>/dev/null; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# Miniforge (mamba) — keeps eccodes/GDAL clean
if [[ ! -d "$HOME/miniforge3" ]]; then
  wget -qO /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm /tmp/miniforge.sh
fi

# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate base

# Create the env from environment.yml (use mamba for speed)
mamba env create -f environment.yml || mamba env update -f environment.yml --prune

echo
echo "Setup complete. To activate the environment, run the following commands:"
echo ""
echo "  # Make conda available in the current shell"
echo "  source ~/miniforge3/etc/profile.d/conda.sh"
echo ""
echo "  # Initialize conda for bash (one-time per instance — writes to ~/.bashrc)"
echo "  ~/miniforge3/bin/conda init bash"
echo ""
echo "  # Initialize mamba for the current session"
echo "  eval \"\$(mamba shell hook --shell bash)\""
echo ""
echo "  # Activate the project environment"
echo "  mamba activate indot"
echo ""
echo "On future reconnections, 'mamba activate indot' is all you need."
