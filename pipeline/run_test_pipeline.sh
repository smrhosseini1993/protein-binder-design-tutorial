#!/bin/bash
# =============================================================================
# run_test_pipeline.sh
# End-to-End Test Script for the BoNT/A Protein Binder Design Pipeline.
# Runs exactly 1 batch of 2 designs to verify all tools and Google Drive upload.
# =============================================================================

set -e

# --- Configuration ---
TOTAL_BATCHES=5
DESIGNS_PER_BATCH=10
WORKSPACE="$HOME/Desktop/protein_workspace"
CURRENT_BATCH_DIR="$WORKSPACE/current_batch"
COMPLETED_DIR="$WORKSPACE/completed_batches"
GDRIVE_REMOTE="gdrive:BoNTA_Binder_Project" # Assumes rclone is configured with this remote name

# Ensure directories exist
mkdir -p "$CURRENT_BATCH_DIR"
mkdir -p "$COMPLETED_DIR"

echo "=============================================="
echo "  Starting End-to-End Test Pipeline"
echo "  Total Batches: $TOTAL_BATCHES"
echo "  Designs per Batch: $DESIGNS_PER_BATCH"
echo "=============================================="

for (( BATCH=1; BATCH<=TOTAL_BATCHES; BATCH++ ))
do
    echo ""
    echo ">>> STARTING TEST BATCH $BATCH <<<"
    
    # Create batch-specific directories
    RF_OUT="$CURRENT_BATCH_DIR/1_rfdiffusion_out"
    MPNN_OUT="$CURRENT_BATCH_DIR/2_proteinmpnn_out"
    COLAB_OUT="$CURRENT_BATCH_DIR/3_colabfold_out"
    
    mkdir -p "$RF_OUT" "$MPNN_OUT" "$COLAB_OUT"
    
    # ---------------------------------------------------------
    # Step A: RFdiffusion (Backbone Generation)
    # ---------------------------------------------------------
    echo "[Test Batch] Running RFdiffusion ($DESIGNS_PER_BATCH designs)..."
    # We pass the number of designs to the script
    ./run_rfdiffusion.sh "inputs/trimmed.pdb" "$RF_OUT" "$DESIGNS_PER_BATCH"
    
    # ---------------------------------------------------------
    # Step B: ProteinMPNN (Sequence Design)
    # ---------------------------------------------------------
    echo "[Test Batch] Running ProteinMPNN..."
    ./run_proteinmpnn.sh "$RF_OUT" "$MPNN_OUT"
    
    # ---------------------------------------------------------
    # Step C: Sequence Selection
    # ---------------------------------------------------------
    echo "[Test Batch] Extracting best sequences..."
    BEST_SEQ_FASTA="$CURRENT_BATCH_DIR/best_sequences_test_batch_${BATCH}.fasta"
    python3 extract_best_seq.py --input_dir "$MPNN_OUT/seqs" --output_file "$BEST_SEQ_FASTA"
    
    # ---------------------------------------------------------
    # Step D: ColabFold (Structure Prediction)
    # ---------------------------------------------------------
    echo "[Test Batch] Running ColabFold..."
    ./run_colabfold.sh "$BEST_SEQ_FASTA" "$COLAB_OUT"
    
    # ---------------------------------------------------------
    # Step E: Filtering
    # ---------------------------------------------------------
    echo "[Test Batch] Filtering results..."
    REPORT_CSV="$COMPLETED_DIR/test_report_batch_${BATCH}.csv"
    python3 filter_results.py --input_dir "$COLAB_OUT" --output_csv "$REPORT_CSV"
    
    # ---------------------------------------------------------
    # Step F: Compression
    # ---------------------------------------------------------
    echo "[Test Batch] Compressing results..."
    ARCHIVE_NAME="test_results_batch_${BATCH}.tar.gz"
    ARCHIVE_PATH="$COMPLETED_DIR/$ARCHIVE_NAME"
    
    # Compress the entire current_batch directory
    tar -czf "$ARCHIVE_PATH" -C "$WORKSPACE" "current_batch"
    
    # ---------------------------------------------------------
    # Step G: Upload to Google Drive (via Docker rclone)
    # ---------------------------------------------------------
    echo "[Test Batch] Uploading to Google Drive..."
    RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
    
    if [ -f "$RCLONE_CONFIG" ]; then
        # Mount rclone.conf as read-write (not :ro) so rclone can refresh its token cache
        docker run --rm \
            -v "$RCLONE_CONFIG:/config/rclone/rclone.conf:rw" \
            -v "$COMPLETED_DIR:/data" \
            rclone/rclone copy "/data/$ARCHIVE_NAME" "$GDRIVE_REMOTE" \
            --progress
            
        docker run --rm \
            -v "$RCLONE_CONFIG:/config/rclone/rclone.conf:rw" \
            -v "$COMPLETED_DIR:/data" \
            rclone/rclone copy "/data/$(basename "$REPORT_CSV")" "$GDRIVE_REMOTE" \
            --progress
            
        echo "[Test Batch] Upload complete."
    else
        echo "WARNING: rclone config not found at $RCLONE_CONFIG."
        echo "Skipping Google Drive upload. Files remain in $COMPLETED_DIR."
        echo "Please configure rclone before running the full pipeline."
    fi
    
    # ---------------------------------------------------------
    # Step H: Cleanup
    # ---------------------------------------------------------
    echo "[Test Batch] Cleaning up temporary files..."
    # ColabFold runs as root inside its container, so MSA files are root-owned.
    # We use a lightweight alpine container (also running as root) to delete them.
    docker run --rm \
        -v "$CURRENT_BATCH_DIR:/cleanup" \
        alpine sh -c "rm -rf /cleanup/*"
    
    if [ -f "$RCLONE_CONFIG" ]; then
        rm -f "$ARCHIVE_PATH"
        echo "[Test Batch] Local archive deleted to save space."
    fi
    
    echo ">>> TEST BATCH COMPLETE <<<"
    echo ""
done

echo "=============================================="
echo "  TEST PIPELINE FINISHED!"
echo "  Check your Google Drive for the test_results.tar.gz file."
echo "=============================================="
