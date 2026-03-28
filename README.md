# BoNT/A Binder Design Pipeline

This directory contains the ready-to-run bash scripts for generating protein binders against the BoNT/A target using Dockerized RFdiffusion and ProteinMPNN.

## Prerequisites

1. **Docker**: Must be installed and your user must be in the `docker` group.
2. **NVIDIA Container Toolkit**: Must be installed to allow GPU passthrough (`--gpus all`).
3. **Storage**: At least 10GB of free space for model weights and outputs.

## Step 1: Download Model Weights

Before running the pipeline, you need to download the pre-trained model weights for RFdiffusion.

```bash
chmod +x setup_models.sh
./setup_models.sh
```
This will create a `models/rfdiffusion_weights` directory and download ~5GB of `.pt` files.

## Step 2: Run RFdiffusion (Backbone Generation)

This script runs the RFdiffusion Docker container to generate 10 backbone designs for the BoNT/A target.

```bash
chmod +x run_rfdiffusion.sh
./run_rfdiffusion.sh
```
*   **Input**: `inputs/trimmed.pdb` (The BoNT/A target structure)
*   **Output**: `outputs/rfdiffusion/` (Contains the generated `.pdb` backbones and `.trb` trajectory files)

*Note: The script is currently set to generate 10 designs (`inference.num_designs=10`) for testing. You can edit the script to increase this number later.*

## Step 3: Run ProteinMPNN (Sequence Design)

Once you have the backbones, you need to design the amino acid sequences that will fold into those shapes.

```bash
chmod +x run_proteinmpnn.sh
./run_proteinmpnn.sh
```
*   **Input**: `outputs/rfdiffusion/`
*   **Output**: `outputs/proteinmpnn/` (Contains `.fa` FASTA files with the designed sequences)

## Directory Structure

```text
pipeline/
├── README.md
├── setup_models.sh
├── run_rfdiffusion.sh
├── run_proteinmpnn.sh
├── inputs/
│   └── trimmed.pdb          # Your target structure
├── models/                  # Created by setup_models.sh
│   └── rfdiffusion_weights/
└── outputs/                 # Created by the run scripts
    ├── rfdiffusion/
    └── proteinmpnn/
```
