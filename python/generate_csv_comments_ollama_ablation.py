import argparse
import csv
import json
import os
import sys

from comment_generator_ollama import ABLATION_PROFILES, generate_comment_from_code


DEFAULT_REPO_ROOT = r"D:\Reengineering\cloned_repos"
DEFAULT_PROFILES = [
    "phi_only",
    "phi_ast",
    "phi_ast_intent",
    "phi_ast_intent_repository",
    "phi_ast_intent_repository_external_docs",
    "full_rag",
]


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


def append_ablation_fieldnames(fieldnames, profiles):
    for profile in profiles:
        comment_field = f"generated_comment_{profile}"
        response_field = f"generation_response_{profile}"
        error_field = f"generation_error_{profile}"
        for field in (comment_field, response_field, error_field):
            if field not in fieldnames:
                fieldnames.append(field)


def process_csv(input_csv, output_csv, repo_root, profiles, start_row=1):
    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        append_ablation_fieldnames(fieldnames, profiles)

        mode = "a" if start_row > 233 and os.path.exists(output_csv) else "w"
        with open(output_csv, mode, encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()

            for index, row in enumerate(reader, start=1):
                if index < start_row:
                    continue

                code_text = resolve_code(row, repo_root)
                if not code_text:
                    for profile in profiles:
                        row[f"generated_comment_{profile}"] = ""
                        row[f"generation_response_{profile}"] = ""
                        row[f"generation_error_{profile}"] = "Code not found in row or cloned repo fallback."
                    writer.writerow(row)
                    continue

                for profile in profiles:
                    try:
                        result = generate_comment_from_code(code_text, ablation_profile=profile)
                        row[f"generated_comment_{profile}"] = result.get("comment", "")
                        row[f"generation_response_{profile}"] = json.dumps(result, ensure_ascii=False)
                        row[f"generation_error_{profile}"] = ""
                    except Exception as error:
                        row[f"generated_comment_{profile}"] = ""
                        row[f"generation_response_{profile}"] = ""
                        row[f"generation_error_{profile}"] = str(error)

                writer.writerow(row)

                if index % 10 == 0:
                    print(f"Processed {index} rows...", file=sys.stderr)


def parse_profiles(raw_profiles):
    profiles = [profile.strip() for profile in raw_profiles.split(",") if profile.strip()]
    unknown_profiles = [profile for profile in profiles if profile not in ABLATION_PROFILES]
    if unknown_profiles:
        available_profiles = ", ".join(sorted(ABLATION_PROFILES))
        raise ValueError(
            f"Unknown ablation profile(s): {', '.join(unknown_profiles)}. "
            f"Available profiles: {available_profiles}"
        )
    return profiles


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ollama/Phi ablation study over CSV code samples.")
    parser.add_argument(
        "--input",
        default=r"F:\Extension\Result\python_train_0_sample.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default=r"F:\Extension\Result\python_train_0_sample_ablation_comments.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--repo-root",
        default=DEFAULT_REPO_ROOT,
        help="Root directory containing cloned repositories for fallback code lookup.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated ablation profile names.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="Row index to start processing from (1-indexed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected_profiles = parse_profiles(arguments.profiles)
    process_csv(arguments.input, arguments.output, arguments.repo_root, selected_profiles, arguments.start_row)
