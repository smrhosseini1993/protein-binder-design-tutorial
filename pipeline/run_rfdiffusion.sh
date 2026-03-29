#!/bin/bash
# run_rfdiffusion.sh
# Runs the RFdiffusion Docker container for the BoNT/A binder design task

# Input arguments (with defaults for standalone testing)
INPUT_PDB="${1:-inputs/trimmed.pdb}"
OUTPUTS_DIR="${2:-outputs/rfdiffusion}"
NUM_DESIGNS="${3:-10}"

# Get absolute paths for Docker volume mounting
PIPELINE_DIR=$(pwd)
ABS_INPUT_PDB=$(realpath "$INPUT_PDB")
INPUTS_DIR=$(dirname "$ABS_INPUT_PDB")
PDB_FILENAME=$(basename "$ABS_INPUT_PDB")

mkdir -p "$OUTPUTS_DIR"
ABS_OUTPUTS_DIR=$(realpath "$OUTPUTS_DIR")
MODELS_DIR="${PIPELINE_DIR}/models/rfdiffusion_weights"

echo "Starting RFdiffusion for BoNT/A binder design..."
echo "Target: $ABS_INPUT_PDB"
echo "Output directory: $ABS_OUTPUTS_DIR"
echo "Number of designs: $NUM_DESIGNS"

# Run the Docker container
# --gpus all: Pass the T4 GPU to the container
# -u: Run as the current user to avoid root-owned output files
# hydra.run.dir: Redirects Hydra's log directory to the mounted volume to avoid permission errors
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -v "${INPUTS_DIR}:/inputs" \
  -v "${ABS_OUTPUTS_DIR}:/outputs" \
  -v "${MODELS_DIR}:/models" \
  rosettacommons/rfdiffusion:latest \
  inference.output_prefix=/outputs/bonta_binder \
  inference.model_directory_path=/models \
  inference.input_pdb=/inputs/${PDB_FILENAME} \
  inference.num_designs=${NUM_DESIGNS} \
  'contigmap.contigs=[A1-425/0 80-140]' \
  'ppi.hotspot_res=[A12,A13,A14,A15,A17]' \
  denoiser.noise_scale_ca=0 \
  denoiser.noise_scale_frame=0 \
  inference.schedule_directory_path=/outputs/schedules \
  hydra.run.dir=/outputs/hydra_logs

echo "RFdiffusion completed! Check the $OUTPUTS_DIR directory."
