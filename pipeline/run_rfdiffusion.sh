#!/bin/bash
# run_rfdiffusion.sh
# Runs the RFdiffusion Docker container for the BoNT/A binder design task

# Get the absolute path of the current directory
PIPELINE_DIR=$(pwd)

# Define paths
INPUTS_DIR="${PIPELINE_DIR}/inputs"
OUTPUTS_DIR="${PIPELINE_DIR}/outputs/rfdiffusion"
MODELS_DIR="${PIPELINE_DIR}/models/rfdiffusion_weights"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUTS_DIR"

echo "Starting RFdiffusion for BoNT/A binder design..."
echo "Target: inputs/trimmed.pdb"
echo "Output directory: $OUTPUTS_DIR"

# Run the Docker container
# --gpus all: Pass the T4 GPU to the container
# -u: Run as the current user to avoid root-owned output files
# -v: Mount local directories to the container
docker run -it --rm --gpus all \
  -u $(id -u):$(id -g) \
  -v "${INPUTS_DIR}:/inputs" \
  -v "${OUTPUTS_DIR}:/outputs" \
  -v "${MODELS_DIR}:/models" \
  rosettacommons/rfdiffusion:latest \
  inference.output_prefix=/outputs/bonta_binder \
  inference.model_directory_path=/models \
  inference.input_pdb=/inputs/trimmed.pdb \
  inference.num_designs=10 \
  'contigmap.contigs=[A1-425/0 80-140]' \
  'ppi.hotspot_res=[A12,A13,A14,A15,A17]' \
  denoiser.noise_scale_ca=0 \
  denoiser.noise_scale_frame=0

echo "RFdiffusion completed! Check the outputs/rfdiffusion directory."
