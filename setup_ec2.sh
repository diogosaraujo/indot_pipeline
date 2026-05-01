#!/usr/bin/env bash
# One-shot provisioning for a fresh Ubuntu 24.04 EC2 instance.
# Installs miniforge, creates the `indot` env, and installs all deps.
set -euo pipefail

# System packages
sudo apt-get update -y
sudo apt-get install -y build-essential wget curl git unzip awscli

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
echo "Setup complete. Activate with:"
echo "  source ~/miniforge3/etc/profile.d/conda.sh && conda activate indot"
