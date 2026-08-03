import argparse
import csv
import os
import sys

from comment_generator_ollama_nocontext import generate_comment_from_code


DEFAULT_REPO_ROOT = r"D:\Reengineering\cloned_repos"


def resolve_code(row, repo_root):
    code = (row.get("code") or "").strip()
    if code:
        return code

    repo = (row.get("repo") or "").strip()
    relative_path = (row.get("path") or "").strip()
    if not repo or not relative_path:
        return ""

    repo_name = repo.split("/")[-1]
    file_path = os.path.join(repo_root, repo_name, relative_path)
    if not os.path.exists(file_path):
        return ""

    with open(file_path, "r", encoding="utf-8") as source_file:
        return source_file.read()


def process_csv(input_csv, output_csv, repo_root, start_row=1):
    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])

        if "generated_comment" not in fieldnames:
            fieldnames.append("generated_comment")

        if "generation_error" not in fieldnames:
            fieldnames.append("generation_error")

        mode = "a" if start_row > 1 and os.path.exists(output_csv) else "w"
        with open(output_csv, mode, encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()

            for index, row in enumerate(reader, start=1):
                if index < start_row:
                    continue

                code_text = resolve_code(row, repo_root)
                generated_comment = ""
                generation_error = ""

                if code_text:
                    try:
                        generated_comment = generate_comment_from_code(code_text)
                    except Exception as error:
                        generation_error = str(error)
                else:
                    generation_error = "Code not found in row or cloned repo fallback."

                row["generated_comment"] = generated_comment
                row["generation_error"] = generation_error
                writer.writerow(row)

                if index % 10 == 0:
                    print(f"Processed {index} rows...", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate docstring comments for CSV rows using Ollama without retrieval context."
    )
    parser.add_argument(
        "--input",
        default=r"F:\Extension\python_train_0_sample.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default=r"F:\Extension\python_train_0_sample_with_qwen_nocontext_comments.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--repo-root",
        default=DEFAULT_REPO_ROOT,
        help="Root directory containing cloned repositories for fallback code lookup.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=215,
        help="Row index to start processing from (1-indexed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    process_csv(arguments.input, arguments.output, arguments.repo_root, arguments.start_row)
