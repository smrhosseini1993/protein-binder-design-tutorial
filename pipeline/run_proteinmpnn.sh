#!/bin/bash
# run_proteinmpnn.sh
# Runs the ProteinMPNN Docker container to design sequences for the RFdiffusion backbones

INPUTS_DIR="${1:-outputs/rfdiffusion}"
OUTPUTS_DIR="${2:-outputs/proteinmpnn}"

mkdir -p "$OUTPUTS_DIR"
ABS_INPUTS_DIR=$(realpath "$INPUTS_DIR")
ABS_OUTPUTS_DIR=$(realpath "$OUTPUTS_DIR")

echo "Starting ProteinMPNN sequence design..."
echo "Input directory: $ABS_INPUTS_DIR"
echo "Output directory: $ABS_OUTPUTS_DIR"

# Step 1: Parse PDBs into JSONL format required by ProteinMPNN
echo "Parsing PDBs..."
docker run --rm \
  -u $(id -u):$(id -g) \
  --entrypoint python \
  -v "${ABS_INPUTS_DIR}:/inputs" \
  -v "${ABS_OUTPUTS_DIR}:/outputs" \
  rosettacommons/proteinmpnn:latest \
  /app/proteinmpnn/helper_scripts/parse_multiple_chains.py \
  --input_path=/inputs \
  --output_path=/outputs/parsed_pdbs.jsonl

# Step 2: Run ProteinMPNN
echo "Running sequence design..."
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -v "${ABS_INPUTS_DIR}:/inputs" \
  -v "${ABS_OUTPUTS_DIR}:/outputs" \
  rosettacommons/proteinmpnn:latest \
  --jsonl_path /outputs/parsed_pdbs.jsonl \
  --out_folder /outputs \
  --num_seq_per_target 8 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1

echo "ProteinMPNN completed! Check the $OUTPUTS_DIR directory."
