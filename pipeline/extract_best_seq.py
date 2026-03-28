#!/usr/bin/env python3
"""
extract_best_seq.py

This script reads the output FASTA files from ProteinMPNN.
For each backbone, ProteinMPNN generates multiple sequences (default: 8) and saves them in a single .fa file.
This script parses each .fa file, identifies the sequence with the lowest (best) "score" (MPNN score),
and writes that single best sequence to a new, combined FASTA file ready for ColabFold.

[METHODOLOGICAL FLAG] Why pick only 1 sequence out of 8?
The next step (ColabFold) is extremely computationally expensive. Folding all 8 sequences for 1,000 backbones
(8,000 folds) would take weeks. By picking the single best sequence per backbone based on the MPNN score,
we save massive amounts of time and compute power while retaining the highest quality candidates.
"""

import os
import glob
import argparse

def parse_fasta(filepath):
    """Parses a FASTA file and returns a list of (header, sequence) tuples."""
    entries = []
    with open(filepath, 'r') as f:
        header = ""
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq)))
                header = line[1:] # Remove the '>'
                seq = []
            else:
                seq.append(line)
        if header:
            entries.append((header, "".join(seq)))
    return entries

def extract_score(header):
    """
    Extracts the score from the ProteinMPNN FASTA header.
    Example header: T=0.1, sample=1, score=1.2345, global_score=1.2345, seq_recovery=0.0000
    """
    parts = header.split(',')
    for part in parts:
        part = part.strip()
        if part.startswith("score="):
            try:
                return float(part.split('=')[1])
            except ValueError:
                return float('inf') # If parsing fails, assign a terrible score
    return float('inf')

def main():
    parser = argparse.ArgumentParser(description="Extract the best sequence from ProteinMPNN outputs.")
    parser.add_argument("--input_dir", required=True, help="Directory containing ProteinMPNN .fa files")
    parser.add_argument("--output_file", required=True, help="Path to the combined output .fasta file")
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input_dir, "*.fa"))
    if not input_files:
        print(f"Error: No .fa files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} FASTA files from ProteinMPNN.")
    
    best_sequences = []

    for filepath in input_files:
        entries = parse_fasta(filepath)
        if not entries:
            continue
            
        # The first entry in ProteinMPNN output is usually the original (native/poly-G) sequence.
        # The subsequent entries are the generated samples.
        # We want to evaluate the generated samples.
        samples = entries[1:] if len(entries) > 1 else entries
        
        best_entry = None
        best_score = float('inf')
        
        for header, seq in samples:
            score = extract_score(header)
            if score < best_score:
                best_score = score
                best_entry = (header, seq)
                
        if best_entry:
            # Create a clean header for ColabFold based on the filename (which is the backbone name)
            base_name = os.path.basename(filepath).replace(".fa", "")
            clean_header = f"{base_name} | MPNN_score={best_score:.4f}"
            best_sequences.append((clean_header, best_entry[1]))

    print(f"Extracted {len(best_sequences)} best sequences.")
    
    # Write the combined FASTA file
    with open(args.output_file, 'w') as f:
        for header, seq in best_sequences:
            f.write(f">{header}\n{seq}\n")
            
    print(f"Saved combined sequences to {args.output_file}")

if __name__ == "__main__":
    main()
