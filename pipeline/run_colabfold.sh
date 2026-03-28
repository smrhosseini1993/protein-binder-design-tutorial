#!/bin/bash
# =============================================================================
# run_colabfold.sh
# Runs the ColabFold Docker container to predict 3D structures from sequences.
# =============================================================================

set -e

# --- Configuration ---
# The directory where the AlphaFold2 weights are stored
MODELS_DIR="$HOME/Desktop/protein_workspace/models/colabfold_weights"

# Input and Output directories (passed as arguments or default to current dir)
INPUT_FASTA="${1:-outputs/best_sequences.fasta}"
OUTPUT_DIR="${2:-outputs/colabfold}"

# Ensure input exists
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input FASTA file not found at $INPUT_FASTA"
    echo "Usage: ./run_colabfold.sh <input_fasta> <output_dir>"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get absolute paths for Docker volume mounting
ABS_INPUT_FASTA=$(realpath "$INPUT_FASTA")
ABS_OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
ABS_MODELS_DIR=$(realpath "$MODELS_DIR")

# We need to mount the directory containing the input fasta, not just the file
INPUT_DIR=$(dirname "$ABS_INPUT_FASTA")
FASTA_FILENAME=$(basename "$ABS_INPUT_FASTA")

echo "=============================================="
echo "  Running ColabFold (AlphaFold2)"
echo "=============================================="
echo "Input: $ABS_INPUT_FASTA"
echo "Output: $ABS_OUTPUT_DIR"
echo "Weights: $ABS_MODELS_DIR"
echo ""

# --- Run ColabFold Docker Container ---
# [METHODOLOGICAL FLAG] Why does ColabFold output 5 models per sequence?
# This is hardcoded into AlphaFold2's architecture. DeepMind trained 5 slightly 
# different neural networks to prevent bias and capture structural ensembles. 
# ColabFold runs your sequence through all 5 networks.
#
# --num-recycle 3: Standard number of recycling iterations for AF2.
# --use-gpu-relax: Uses the GPU to run Amber relaxation on the final structures to fix steric clashes.

docker run --rm --gpus all \
    -v "$ABS_MODELS_DIR:/cache" \
    -v "$INPUT_DIR:/input" \
    -v "$ABS_OUTPUT_DIR:/output" \
    ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
    colabfold_batch \
    "/input/$FASTA_FILENAME" \
    "/output" \
    --num-recycle 3 \
    --use-gpu-relax

echo "=============================================="
echo "  ColabFold Complete!"
echo "  Results saved to: $OUTPUT_DIR"
echo "=============================================="
