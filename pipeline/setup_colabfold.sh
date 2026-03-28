#!/bin/bash
# =============================================================================
# setup_colabfold.sh
# Downloads the ColabFold Docker image and the AlphaFold2 model weights.
# This script only needs to be run ONCE.
# =============================================================================

set -e  # Exit immediately if any command fails

# --- Configuration ---
# The directory where the large AI model weights will be stored.
# This directory is PERSISTENT across all batches (we never delete it).
MODELS_DIR="$HOME/Desktop/protein_workspace/models/colabfold_weights"

echo "=============================================="
echo "  ColabFold Setup Script"
echo "=============================================="
echo ""

# --- Step 1: Pull the ColabFold Docker Image ---
echo "[1/3] Pulling the ColabFold Docker image..."
echo "      This may take 10-20 minutes on the first run."
docker pull ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2
echo "      Done."
echo ""

# --- Step 2: Create the weights directory ---
echo "[2/3] Creating weights directory at: $MODELS_DIR"
mkdir -p "$MODELS_DIR"
echo "      Done."
echo ""

# --- Step 3: Download the AlphaFold2 weights ---
# We use the ColabFold Docker container itself to download the weights.
# The weights are mounted into the container and saved to the host.
# [METHODOLOGICAL FLAG] We use AlphaFold2 weights (not AlphaFold3) because
# the RFdiffusion + ProteinMPNN + ColabFold pipeline was validated using AF2.
# Using AF3 would require re-validation of the filtering thresholds.
echo "[3/3] Downloading AlphaFold2 model weights (~3-5 GB)..."
echo "      This may take 20-40 minutes depending on your internet speed."
docker run --rm \
    -v "$MODELS_DIR:/cache" \
    ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
    python3 -m colabfold.download
echo "      Done."
echo ""

echo "=============================================="
echo "  Setup Complete!"
echo "  Weights saved to: $MODELS_DIR"
echo "  You are now ready to run the pipeline."
echo "=============================================="
