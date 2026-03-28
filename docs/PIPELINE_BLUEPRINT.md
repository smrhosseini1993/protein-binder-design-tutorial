# Pipeline Blueprint: BoNT/A Protein Binder Design

This document serves as the master blueprint for the entire protein binder design pipeline. It outlines the methodology, the data flow (how files move between the server, Docker, and Google Drive), the directory structures, and the reasoning behind specific scientific choices.

## 1. The Big Picture: Why Batches?

**Goal:** Generate 10,000 potential protein binders for the BoNT/A target.
**Constraint:** The server has only 24 GB of available storage. Generating all 10,000 designs at once would require >50 GB of space, causing the server to crash.
**Solution:** We will run the pipeline in **10 sequential batches of 1,000 designs**. After each batch, the results are compressed, uploaded to Google Drive, and deleted from the server to free up space for the next batch.

---

## 2. Methodology & Scientific Choices

This section explains the step-by-step scientific process for a single batch (1,000 designs) and the reasoning behind the numbers.

### Step A: Backbone Generation (RFdiffusion)
*   **Action:** RFdiffusion generates 1,000 3D shapes (backbones) that are physically capable of binding to the BoNT/A target.
*   **Output:** 1,000 `.pdb` files (the 3D coordinates) and 1,000 `.trb` files (metadata).
*   **[METHODOLOGICAL FLAG] Why 10,000 total designs?**
    *   *Reasoning:* Protein design is a numbers game. Most AI-generated designs will fail in the real world. Generating 10,000 gives us a statistically significant pool to filter down to the top ~10-50 candidates for actual lab testing.

### Step B: Sequence Design (ProteinMPNN)
*   **Action:** For each of the 1,000 backbones, ProteinMPNN guesses which amino acid sequence will fold into that specific shape.
*   **[METHODOLOGICAL FLAG] Why generate 8 sequences per backbone?**
    *   *Reasoning:* Biology is flexible. Multiple different sequences can fold into the same shape. By asking ProteinMPNN to generate 8 options per backbone, we increase our chances of finding a highly stable sequence. This is a standard empirical practice in the field.
*   **Output:** 1,000 `.fa` (FASTA) files, each containing 8 sequence options.

### Step C: Sequence Selection (Python Script)
*   **Action:** A custom Python script reads the ProteinMPNN outputs. For each backbone, it looks at the 8 generated sequences, picks the single best one based on the "MPNN score" (lowest energy/highest confidence), and discards the other 7.
*   **[METHODOLOGICAL FLAG] Why pick only 1 sequence out of 8?**
    *   *Reasoning:* The next step (ColabFold) is extremely computationally expensive. Folding all 8 sequences for 1,000 backbones (8,000 folds) would take weeks. By picking the single best sequence per backbone, we save massive amounts of time and compute power while retaining the highest quality candidates.
*   **Output:** A single, clean `.fa` file containing exactly 1,000 optimized sequences.

### Step D: Structure Prediction & Validation (ColabFold / AlphaFold2)
*   **Action:** ColabFold takes the 1,000 sequences and predicts how they will actually fold in 3D space, independent of what RFdiffusion originally intended.
*   **[METHODOLOGICAL FLAG] Why does ColabFold output 5 models per sequence?**
    *   *Reasoning:* This is hardcoded into AlphaFold2's architecture. DeepMind trained 5 slightly different neural networks to prevent bias. ColabFold runs your sequence through all 5 networks. If all 5 predict the exact same shape, you have high confidence it will work in reality.
*   **Output:** 5,000 `.pdb` files (1,000 sequences × 5 models) and 5,000 `.json` files containing confidence scores.

### Step E: Filtering (Python Script)
*   **Action:** A script reads the ColabFold `.json` files and filters the designs based on three critical metrics:
    1.  **pLDDT (>80):** Confidence in the overall fold.
    2.  **iPAE (<10):** Confidence that the binder will stick to the target.
    3.  **RMSD (<2.0 Å):** How closely the ColabFold prediction matches the original RFdiffusion backbone.
*   **Output:** A clean `batch_X_report.csv` listing only the designs that passed the filters.

---

## 3. Data Flow: Server, Docker, and Google Drive

Because you are not an admin, we use Docker for *everything* to bypass permission issues.

1.  **The Host Server (Your Environment):** Acts only as a traffic controller. It holds the master bash script and the raw input files.
2.  **The AI Containers (Docker):** The master script spins up isolated Docker containers for RFdiffusion, ProteinMPNN, and ColabFold. The server "mounts" (shares) a specific folder with the container. The container does the heavy lifting using the GPU, saves the output to the shared folder, and then shuts down.
3.  **The Cloud (Google Drive via Docker):** Once a batch is finished and compressed, the master script spins up an `rclone` Docker container. This container takes the compressed `.tar.gz` file, uploads it directly to your Google Drive, and then the server deletes the local copy.

---

## 4. Directory Structures

### A. The Project Repository (What lives on GitHub)
This is the code you clone to your server.
```text
protein-binder-design-tutorial/
├── PIPELINE_BLUEPRINT.md       # This document
├── MASTERCLASS.md              # The educational tutorial
├── DOCKER_MASTERCLASS.md       # The Docker explanation
├── SESSION_LOG.md              # Record of our progress
└── pipeline/                   # The actual executable code
    ├── inputs/
    │   └── trimmed.pdb         # The BoNT/A target
    ├── scripts/
    │   ├── setup_models.sh     # Downloads AI weights
    │   ├── extract_best_seq.py # Step C script
    │   └── filter_results.py   # Step E script
    └── run_master_batched.sh   # The main loop that runs everything
```

### B. The Server Workspace (What happens during runtime)
When you run `run_master_batched.sh` on your server, it creates temporary folders to hold the massive amounts of data.
```text
~/Desktop/protein_workspace/
├── models/                     # 10GB of downloaded AI weights (Persistent)
├── current_batch/              # The active working directory (Wiped after each batch)
│   ├── 1_rfdiffusion_out/      # 1.5 GB of PDBs/TRBs
│   ├── 2_proteinmpnn_out/      # 10 MB of FASTA files
│   └── 3_colabfold_out/        # 10.0 GB of PDBs/JSONs
└── completed_batches/          # Where compressed files wait to be uploaded
    └── batch_1_results.tar.gz  # ~2.5 GB (Uploaded to GDrive, then deleted)
```

### C. Your Google Drive (The Final Destination)
What you will see when you log into Google Drive on your laptop.
```text
My Drive/
└── BoNTA_Binder_Project/
    ├── batch_1_results.tar.gz
    ├── batch_1_report.csv      # The filtered list of best binders
    ├── batch_2_results.tar.gz
    ├── batch_2_report.csv
    ...
    └── batch_10_results.tar.gz
```
