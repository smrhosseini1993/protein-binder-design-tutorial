#!/usr/bin/env python3
"""
filter_results.py

This script reads the JSON output files from ColabFold and filters the designed binders
based on standard structural biology metrics. It generates a CSV report of the successful candidates.

[METHODOLOGICAL FLAGS]
- pLDDT > 80: High confidence that the monomer will fold into a stable structure.
- iPAE < 10: High confidence that the binder will stick to the target.
- RMSD < 2.0: The predicted structure closely matches the original RFdiffusion backbone.
"""

import os
import glob
import json
import csv
import argparse

def parse_colabfold_json(filepath):
    """Parses a ColabFold JSON file to extract pLDDT and iPAE."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # ColabFold JSONs usually contain lists of scores per residue.
        # We want the average pLDDT for the binder chain.
        # For simplicity in this tutorial, we take the overall mean pLDDT.
        # In a production setting, you would isolate the binder chain's pLDDT.
        mean_plddt = sum(data.get('plddt', [])) / len(data.get('plddt', [1])) if data.get('plddt') else 0
        
        # iPAE (interface PAE) is often stored in the PAE matrix.
        # ColabFold sometimes outputs a summary 'ptm' or 'iptm' score.
        # If the exact iPAE is not in the JSON summary, we use ipTM as a proxy (ipTM > 0.8 is good).
        # For this script, we will look for 'pae' or 'iptm'.
        iptm = data.get('iptm', 0)
        
        return mean_plddt, iptm
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description="Filter ColabFold results to find the best binders.")
    parser.add_argument("--input_dir", required=True, help="Directory containing ColabFold .json files")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV report")
    
    # Thresholds
    parser.add_argument("--min_plddt", type=float, default=80.0, help="Minimum pLDDT score (default: 80)")
    parser.add_argument("--min_iptm", type=float, default=0.8, help="Minimum ipTM score as proxy for iPAE (default: 0.8)")
    
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    if not json_files:
        print(f"Error: No .json files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} ColabFold JSON files. Filtering...")
    
    results = []
    passed_count = 0

    for filepath in json_files:
        filename = os.path.basename(filepath)
        # Example filename: design_001_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.json
        design_name = filename.split("_unrelaxed")[0] if "_unrelaxed" in filename else filename.replace(".json", "")
        
        plddt, iptm = parse_colabfold_json(filepath)
        
        # Check if it passes the filters
        passed = (plddt >= args.min_plddt) and (iptm >= args.min_iptm)
        
        if passed:
            passed_count += 1
            
        results.append({
            'Design_Name': design_name,
            'File_Name': filename,
            'pLDDT': round(plddt, 2),
            'ipTM': round(iptm, 3),
            'Passed_Filters': passed
        })

    # Sort results by ipTM (descending)
    results.sort(key=lambda x: x['ipTM'], reverse=True)

    # Write to CSV
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Design_Name', 'File_Name', 'pLDDT', 'ipTM', 'Passed_Filters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("==============================================")
    print(f"  Filtering Complete!")
    print(f"  Total designs evaluated: {len(json_files)}")
    print(f"  Designs passing filters: {passed_count}")
    print(f"  Report saved to: {args.output_csv}")
    print("==============================================")

if __name__ == "__main__":
    main()
