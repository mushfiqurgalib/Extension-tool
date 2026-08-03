import argparse
import ast
import csv
import json
from typing import Any, Dict


def parse_generated_comment(value: str) -> Dict[str, Any]:
    raw = (value or "").strip()
    if not raw:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return {"comment": raw}


def flatten_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    edit = payload.get("edit")
    if not isinstance(edit, dict):
        edit = {}

    return {
        "generated_mode": stringify(payload.get("mode")),
        "generated_message": stringify(payload.get("message")),
        "generated_comment_text": stringify(payload.get("comment")),
        "generated_edit_type": stringify(edit.get("type")),
        "generated_edit_start_line": stringify(edit.get("startLine")),
        "generated_edit_start_character": stringify(edit.get("startCharacter")),
        "generated_edit_end_line": stringify(edit.get("endLine")),
        "generated_edit_end_character": stringify(edit.get("endCharacter")),
        "generated_edit_text": stringify(edit.get("text")),
    }


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def process_csv(input_csv: str, output_csv: str) -> None:
    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])

        extra_columns = [
            "generated_mode",
            "generated_message",
            "generated_comment_text",
            "generated_edit_type",
            "generated_edit_start_line",
            "generated_edit_start_character",
            "generated_edit_end_line",
            "generated_edit_end_character",
            "generated_edit_text",
        ]

        for column in extra_columns:
            if column not in fieldnames:
                fieldnames.append(column)

        with open(output_csv, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                payload = parse_generated_comment(row.get("generated_comment", ""))
                row.update(flatten_payload(payload))
                
                if not row.get("generated_comment_text", "").strip():
                    row["generated_comment_text"] = row.get("docstring", "")
                    
                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a generated_comment payload into separate CSV columns."
    )
    parser.add_argument(
        "--input",
        default=r"F:\Extension\python_train_0_sample_with_qwen_comments.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default=r"F:\Extension\python_train_0_sample_with_qwen_comments_split.csv",
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    process_csv(arguments.input, arguments.output)
