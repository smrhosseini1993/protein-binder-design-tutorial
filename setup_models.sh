#!/bin/bash
# setup_models.sh
# Downloads the necessary model weights for RFdiffusion

# Define the directory where weights will be stored
WEIGHTS_DIR="./models/rfdiffusion_weights"

echo "Creating weights directory at $WEIGHTS_DIR..."
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "Downloading RFdiffusion weights (this will take a few minutes, ~5GB total)..."

# Download the weights from the Baker Lab's official server
# The -nc flag prevents downloading if the file already exists
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/Inpaint_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/Complex_beta_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/Original_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/59a5af8cce2e9c581b1c9faa7307110f/Active_ckpt.pt
wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/73504c966b078ec36f204813805a933b/Base_epoch8_ckpt.pt

echo "All weights downloaded successfully!"
echo "You are ready to run the pipeline."
