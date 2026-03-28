#!/bin/bash
# =============================================================================
# run_pipeline_batched.sh
# The Master Script for the BoNT/A Protein Binder Design Pipeline.
# Runs RFdiffusion, ProteinMPNN, and ColabFold in batches to manage storage.
# Compresses results and uploads them to Google Drive via rclone Docker.
# =============================================================================

set -e

# --- Configuration ---
TOTAL_BATCHES=10
DESIGNS_PER_BATCH=1000
WORKSPACE="$HOME/Desktop/protein_workspace"
CURRENT_BATCH_DIR="$WORKSPACE/current_batch"
COMPLETED_DIR="$WORKSPACE/completed_batches"
GDRIVE_REMOTE="gdrive:BoNTA_Binder_Project" # Assumes rclone is configured with this remote name

# Ensure directories exist
mkdir -p "$CURRENT_BATCH_DIR"
mkdir -p "$COMPLETED_DIR"

echo "=============================================="
echo "  Starting Batched Protein Design Pipeline"
echo "  Total Batches: $TOTAL_BATCHES"
echo "  Designs per Batch: $DESIGNS_PER_BATCH"
echo "=============================================="

for (( BATCH=1; BATCH<=TOTAL_BATCHES; BATCH++ ))
do
    echo ""
    echo ">>> STARTING BATCH $BATCH / $TOTAL_BATCHES <<<"
    
    # Create batch-specific directories
    RF_OUT="$CURRENT_BATCH_DIR/1_rfdiffusion_out"
    MPNN_OUT="$CURRENT_BATCH_DIR/2_proteinmpnn_out"
    COLAB_OUT="$CURRENT_BATCH_DIR/3_colabfold_out"
    
    mkdir -p "$RF_OUT" "$MPNN_OUT" "$COLAB_OUT"
    
    # ---------------------------------------------------------
    # Step A: RFdiffusion (Backbone Generation)
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Running RFdiffusion..."
    # Note: In a real run, you would pass DESIGNS_PER_BATCH to the script
    # For this tutorial, we assume run_rfdiffusion.sh handles the configuration
    ./run_rfdiffusion.sh "inputs/trimmed.pdb" "$RF_OUT" "$DESIGNS_PER_BATCH"
    
    # ---------------------------------------------------------
    # Step B: ProteinMPNN (Sequence Design)
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Running ProteinMPNN..."
    ./run_proteinmpnn.sh "$RF_OUT" "$MPNN_OUT"
    
    # ---------------------------------------------------------
    # Step C: Sequence Selection
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Extracting best sequences..."
    BEST_SEQ_FASTA="$CURRENT_BATCH_DIR/best_sequences_batch_${BATCH}.fasta"
    python3 extract_best_seq.py --input_dir "$MPNN_OUT/seqs" --output_file "$BEST_SEQ_FASTA"
    
    # ---------------------------------------------------------
    # Step D: ColabFold (Structure Prediction)
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Running ColabFold..."
    ./run_colabfold.sh "$BEST_SEQ_FASTA" "$COLAB_OUT"
    
    # ---------------------------------------------------------
    # Step E: Filtering
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Filtering results..."
    REPORT_CSV="$COMPLETED_DIR/batch_${BATCH}_report.csv"
    python3 filter_results.py --input_dir "$COLAB_OUT" --output_csv "$REPORT_CSV"
    
    # ---------------------------------------------------------
    # Step F: Compression
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Compressing results..."
    ARCHIVE_NAME="batch_${BATCH}_results.tar.gz"
    ARCHIVE_PATH="$COMPLETED_DIR/$ARCHIVE_NAME"
    
    # Compress the entire current_batch directory
    tar -czf "$ARCHIVE_PATH" -C "$WORKSPACE" "current_batch"
    
    # ---------------------------------------------------------
    # Step G: Upload to Google Drive (via Docker rclone)
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Uploading to Google Drive..."
    # We use the rclone docker image. It requires the rclone config file to be mounted.
    # The config file is usually at ~/.config/rclone/rclone.conf
    RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
    
    if [ -f "$RCLONE_CONFIG" ]; then
        docker run --rm \
            -v "$RCLONE_CONFIG:/config/rclone/rclone.conf" \
            -v "$COMPLETED_DIR:/data" \
            rclone/rclone copy "/data/$ARCHIVE_NAME" "$GDRIVE_REMOTE" \
            --progress
            
        docker run --rm \
            -v "$RCLONE_CONFIG:/config/rclone/rclone.conf" \
            -v "$COMPLETED_DIR:/data" \
            rclone/rclone copy "/data/$(basename "$REPORT_CSV")" "$GDRIVE_REMOTE" \
            --progress
            
        echo "[Batch $BATCH] Upload complete."
    else
        echo "WARNING: rclone config not found at $RCLONE_CONFIG."
        echo "Skipping Google Drive upload. Files remain in $COMPLETED_DIR."
    fi
    
    # ---------------------------------------------------------
    # Step H: Cleanup (Crucial for storage management)
    # ---------------------------------------------------------
    echo "[Batch $BATCH] Cleaning up temporary files..."
    rm -rf "$CURRENT_BATCH_DIR"/*
    
    # If upload was successful, we can also delete the local archive to save space
    if [ -f "$RCLONE_CONFIG" ]; then
        rm -f "$ARCHIVE_PATH"
        echo "[Batch $BATCH] Local archive deleted to save space."
    fi
    
    echo ">>> BATCH $BATCH COMPLETE <<<"
    echo ""
done

echo "=============================================="
echo "  PIPELINE FINISHED SUCCESSFULLY!"
echo "  All $TOTAL_BATCHES batches processed."
echo "=============================================="
