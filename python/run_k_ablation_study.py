import csv
import os
import sys
import json
from comment_generator_ollama import generate_comment_from_code

INPUT_CSV = r"f:\Extension\Result\python_train_0_sample.csv"
OUTPUT_CSV = r"f:\Extension\Result\k_study_results.csv"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    print("Reading samples from CSV...", file=sys.stderr)
    samples = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            docstring = (row.get("docstring") or "").strip()
            if code and docstring:
                samples.append({
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "func_name": row.get("func_name", ""),
                    "code": code,
                    "docstring": docstring
                })
            if len(samples) >= 10:
                break

    print(f"Selected {len(samples)} samples for study.", file=sys.stderr)

    fieldnames = ["repo", "path", "func_name", "code", "docstring", 
                  "generated_comment_k_1", "generated_comment_k_3", "generated_comment_k_5",
                  "error_k_1", "error_k_3", "error_k_5"]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sample in enumerate(samples, 1):
            print(f"\n[{idx}/10] Processing {sample['repo']} / {sample['func_name']}...", file=sys.stderr)
            row_result = {
                "repo": sample["repo"],
                "path": sample["path"],
                "func_name": sample["func_name"],
                "code": sample["code"],
                "docstring": sample["docstring"]
            }

            for k in [1, 3, 5]:
                print(f"  Generating comment for K={k}...", file=sys.stderr)
                try:
                    # We use full_rag profile with retrieval size k
                    res = generate_comment_from_code(sample["code"], ablation_profile="full_rag", k=k)
                    comment = res.get("comment", "")
                    # Extract the comment body if it returned a dictionary structure
                    if isinstance(comment, str):
                        # clean up if it's JSON output inside string
                        if comment.strip().startswith("{"):
                            try:
                                parsed = json.loads(comment)
                                comment = parsed.get("comment", comment)
                            except:
                                pass
                    row_result[f"generated_comment_k_{k}"] = comment
                    row_result[f"error_k_{k}"] = ""
                except Exception as e:
                    print(f"    Error for K={k}: {e}", file=sys.stderr)
                    row_result[f"generated_comment_k_{k}"] = ""
                    row_result[f"error_k_{k}"] = str(e)

            writer.writerow(row_result)
            f.flush()

    print(f"\nDone! Results written to {OUTPUT_CSV}", file=sys.stderr)

if __name__ == "__main__":
    main()
