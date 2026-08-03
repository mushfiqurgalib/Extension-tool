import csv
import os
import sys
import pandas as pd
from statistics import mean

# Import metrics from our existing evaluation script
from evaluate_ablation_sheet_metrics import (
    normalize_cell, fre_score, gfi_score, lexical_overlap,
    bleu_score, meteor_score, rouge_n, rouge_l
)

INPUT_CSV = r"f:\Extension\Result\k_study_results.csv"

def compute_row_metrics(reference, candidate):
    ref_norm = normalize_cell(reference)
    cand_norm = normalize_cell(candidate)
    
    if not ref_norm or not cand_norm:
        return None
        
    return {
        "FRE": fre_score(cand_norm),
        "GFI": gfi_score(cand_norm),
        "Lexical_Overlap": lexical_overlap(ref_norm, cand_norm),
        "BLEU_1": bleu_score(ref_norm, cand_norm, 1),
        "BLEU_4": bleu_score(ref_norm, cand_norm, 4),
        "METEOR": meteor_score(ref_norm, cand_norm),
        "ROUGE_1": rouge_n(ref_norm, cand_norm, 1),
        "ROUGE_L": rouge_l(ref_norm, cand_norm)
    }

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found. Please run the study runner first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    k_values = [1, 3, 5]
    
    results = {}
    for k in k_values:
        results[k] = {
            "FRE": [],
            "GFI": [],
            "Lexical_Overlap": [],
            "BLEU_1": [],
            "BLEU_4": [],
            "METEOR": [],
            "ROUGE_1": [],
            "ROUGE_L": []
        }

    for idx, row in df.iterrows():
        ref = row.get("docstring", "")
        for k in k_values:
            cand = row.get(f"generated_comment_k_{k}", "")
            metrics = compute_row_metrics(ref, cand)
            if metrics:
                for key, val in metrics.items():
                    if val is not None:
                        results[k][key].append(val)

    summary_rows = []
    for k in k_values:
        summary_row = {"K": f"K={k}"}
        for key in results[k]:
            values = results[k][key]
            summary_row[key] = round(mean(values), 4) if values else 0.0
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*80)
    print("Ablation Study Summary of Retrieval Context Size (K)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    print("Note: For GFI, lower is better. For all other metrics, higher is better.\n")

if __name__ == "__main__":
    main()
