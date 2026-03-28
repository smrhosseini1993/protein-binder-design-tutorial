#!/bin/bash
# run_proteinmpnn.sh
# Runs the ProteinMPNN Docker container to design sequences for the RFdiffusion backbones

PIPELINE_DIR=$(pwd)
INPUTS_DIR="${PIPELINE_DIR}/outputs/rfdiffusion"
OUTPUTS_DIR="${PIPELINE_DIR}/outputs/proteinmpnn"

mkdir -p "$OUTPUTS_DIR"

echo "Starting ProteinMPNN sequence design..."
echo "Input directory: $INPUTS_DIR"
echo "Output directory: $OUTPUTS_DIR"

# Run the Docker container
# Note: ProteinMPNN is lightweight and can run on CPU, but we pass the GPU anyway for speed
docker run -it --rm --gpus all \
  -u $(id -u):$(id -g) \
  -v "${INPUTS_DIR}:/inputs" \
  -v "${OUTPUTS_DIR}:/outputs" \
  rosettacommons/proteinmpnn:latest \
  --pdb_path_chains "A" \
  --out_folder "/outputs" \
  --num_seq_per_target 2 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1

# Note: The above command is a simplified example. 
# For a real run, we need to parse the specific outputs from RFdiffusion.
# A more robust script would loop through the generated PDBs.

echo "ProteinMPNN completed! Check the outputs/proteinmpnn directory."
