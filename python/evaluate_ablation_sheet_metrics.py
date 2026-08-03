import argparse
import ast
import math
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_INPUT = Path("Result") / "Ablation study - Sheet1.csv"
DEFAULT_OUTPUT = Path("Result") / "Ablation study - Sheet1_with_scores.csv"
DEFAULT_GENERATED_PREFIX = "generated_comment_"
METRIC_NAMES = [
    "FRE_score",
    "GFI_score",
    "Lexical_Overlap_score",
    "CodeBERT_consistency_score",
    "BLEU_1_score",
    "BLEU_2_score",
    "BLEU_3_score",
    "BLEU_4_score",
    "METEOR_score",
    "ROUGE_1_score",
    "ROUGE_2_score",
    "ROUGE_L_score",
]


def normalize_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("{") and "comment" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and parsed.get("comment"):
                text = str(parsed["comment"])
        except (SyntaxError, ValueError):
            pass

    text = text.strip()
    if (
        (text.startswith('"""') and text.endswith('"""'))
        or (text.startswith("'''") and text.endswith("'''"))
    ):
        text = text[3:-3]

    text = re.sub(r"^\s*[rubfRUBF]*([\"'])\1\1", "", text).strip()
    text = re.sub(r"([\"'])\1\1\s*$", "", text).strip()
    return re.sub(r"\s+", " ", text)


def word_tokens(text):
    return re.findall(r"[A-Za-z0-9_]+", normalize_cell(text).lower())


def sentence_count(text):
    sentences = re.split(r"[.!?]+", normalize_cell(text))
    return max(1, sum(1 for sentence in sentences if sentence.strip()))


def count_syllables(word):
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0

    vowels = "aeiouy"
    syllables = 0
    previous_is_vowel = False
    for character in word:
        is_vowel = character in vowels
        if is_vowel and not previous_is_vowel:
            syllables += 1
        previous_is_vowel = is_vowel

    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def fre_score(text):
    tokens = word_tokens(text)
    if not tokens:
        return None
    words = len(tokens)
    sentences = sentence_count(text)
    syllables = sum(count_syllables(token) for token in tokens)
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)


def gfi_score(text):
    tokens = word_tokens(text)
    if not tokens:
        return None
    words = len(tokens)
    sentences = sentence_count(text)
    complex_words = sum(1 for token in tokens if count_syllables(token) >= 3)
    return 0.4 * ((words / sentences) + 100 * (complex_words / words))


def lexical_overlap(reference, candidate):
    reference_tokens = set(word_tokens(reference))
    candidate_tokens = set(word_tokens(candidate))
    if not reference_tokens or not candidate_tokens:
        return None
    return len(reference_tokens & candidate_tokens) / len(reference_tokens | candidate_tokens)


def ngrams(tokens, size):
    if size <= 0 or len(tokens) < size:
        return []
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def bleu_score(reference, candidate, max_order):
    reference_tokens = word_tokens(reference)
    candidate_tokens = word_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return None

    precisions = []
    for order in range(1, max_order + 1):
        reference_counts = Counter(ngrams(reference_tokens, order))
        candidate_counts = Counter(ngrams(candidate_tokens, order))
        overlap = sum(
            min(count, reference_counts[gram]) for gram, count in candidate_counts.items()
        )
        total = sum(candidate_counts.values())
        precisions.append((overlap + 1) / (total + 1))

    brevity_penalty = (
        1.0
        if len(candidate_tokens) > len(reference_tokens)
        else math.exp(1 - len(reference_tokens) / max(1, len(candidate_tokens)))
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


def meteor_score(reference, candidate):
    reference_tokens = word_tokens(reference)
    candidate_tokens = word_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return None

    reference_positions = {}
    for index, token in enumerate(reference_tokens):
        reference_positions.setdefault(token, []).append(index)

    matched_reference_indexes = []
    used_reference_indexes = set()
    matches = 0
    for token in candidate_tokens:
        for index in reference_positions.get(token, []):
            if index not in used_reference_indexes:
                used_reference_indexes.add(index)
                matched_reference_indexes.append(index)
                matches += 1
                break

    if matches == 0:
        return 0.0

    precision = matches / len(candidate_tokens)
    recall = matches / len(reference_tokens)
    harmonic = (10 * precision * recall) / (recall + 9 * precision)

    chunks = 1
    for previous, current in zip(matched_reference_indexes, matched_reference_indexes[1:]):
        if current != previous + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic * (1 - penalty)


def rouge_n(reference, candidate, size):
    reference_ngrams = Counter(ngrams(word_tokens(reference), size))
    candidate_ngrams = Counter(ngrams(word_tokens(candidate), size))
    if not reference_ngrams or not candidate_ngrams:
        return None
    overlap = sum(min(count, candidate_ngrams[gram]) for gram, count in reference_ngrams.items())
    return overlap / sum(reference_ngrams.values())


def lcs_length(left, right):
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference, candidate):
    reference_tokens = word_tokens(reference)
    candidate_tokens = word_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return None
    return lcs_length(reference_tokens, candidate_tokens) / len(reference_tokens)


def text_metric_values(reference, candidate):
    return {
        "FRE_score": fre_score(candidate),
        "GFI_score": gfi_score(candidate),
        "Lexical_Overlap_score": lexical_overlap(reference, candidate),
        "BLEU_1_score": bleu_score(reference, candidate, 1),
        "BLEU_2_score": bleu_score(reference, candidate, 2),
        "BLEU_3_score": bleu_score(reference, candidate, 3),
        "BLEU_4_score": bleu_score(reference, candidate, 4),
        "METEOR_score": meteor_score(reference, candidate),
        "ROUGE_1_score": rouge_n(reference, candidate, 1),
        "ROUGE_2_score": rouge_n(reference, candidate, 2),
        "ROUGE_L_score": rouge_l(reference, candidate),
    }


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
        input_mask.sum(1), min=1e-9
    )


def codebert_consistency_batch(tokenizer, model, codes, comments, device, max_length):
    texts = []
    for code, comment in zip(codes, comments):
        texts.append(normalize_cell(code))
        texts.append(normalize_cell(comment))

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    code_embeddings = embeddings[0::2]
    comment_embeddings = embeddings[1::2]
    return torch.sum(code_embeddings * comment_embeddings, dim=1).detach().cpu().tolist()


def discover_generated_columns(df, requested_columns):
    if requested_columns:
        missing = [column for column in requested_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing generated comment columns: {', '.join(missing)}")
        return requested_columns

    return [
        column
        for column in df.columns
        if column.startswith(DEFAULT_GENERATED_PREFIX)
        and not column.startswith("generation_")
        and not column.endswith("_response")
        and not column.endswith("_error")
    ]


def metric_column_name(generated_column, metric_name):
    variant_name = generated_column.removeprefix(DEFAULT_GENERATED_PREFIX)
    return f"{variant_name}_{metric_name}"


def add_text_metrics(df, generated_columns, reference_column):
    for generated_column in generated_columns:
        metric_rows = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Text metrics: {generated_column}",
        ):
            reference = normalize_cell(row.get(reference_column, ""))
            candidate = normalize_cell(row.get(generated_column, ""))
            metric_rows.append(text_metric_values(reference, candidate))

        for metric_name in METRIC_NAMES:
            if metric_name == "CodeBERT_consistency_score":
                continue
            df[metric_column_name(generated_column, metric_name)] = [
                metric_row[metric_name] for metric_row in metric_rows
            ]


def add_codebert_scores(
    df,
    generated_columns,
    code_column,
    model_name,
    batch_size,
    max_length,
    device,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    for generated_column in generated_columns:
        score_column = metric_column_name(generated_column, "CodeBERT_consistency_score")
        scores = [None] * len(df)
        rows_to_score = [
            index
            for index, row in df.iterrows()
            if normalize_cell(row.get(code_column, "")) and normalize_cell(row.get(generated_column, ""))
        ]

        for start in tqdm(
            range(0, len(rows_to_score), batch_size),
            desc=f"CodeBERT: {generated_column}",
        ):
            batch_indexes = rows_to_score[start : start + batch_size]
            codes = df.loc[batch_indexes, code_column].tolist()
            comments = df.loc[batch_indexes, generated_column].tolist()
            batch_scores = codebert_consistency_batch(
                tokenizer,
                model,
                codes,
                comments,
                device,
                max_length,
            )
            for index, score in zip(batch_indexes, batch_scores):
                scores[index] = score

        df[score_column] = scores


def add_average_row(df, generated_columns):
    average_row = {column: "" for column in df.columns}
    average_row["repo"] = "AVERAGE"
    for generated_column in generated_columns:
        for metric_name in METRIC_NAMES:
            column = metric_column_name(generated_column, metric_name)
            if column in df.columns:
                average_row[column] = df[column].mean(skipna=True)
    return pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score every generated comment in the ablation Sheet1 result CSV."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--reference-column", default="docstring", help="Reference comment column.")
    parser.add_argument("--code-column", default="code", help="Code snippet column.")
    parser.add_argument(
        "--generated-columns",
        nargs="*",
        help="Generated comment columns to score. Defaults to all generated_comment_* columns.",
    )
    parser.add_argument(
        "--codebert-model",
        default="microsoft/codebert-base",
        help="CodeBERT model name or local model path.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="CodeBERT batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="CodeBERT tokenizer max length.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for CodeBERT scoring.",
    )
    parser.add_argument(
        "--skip-codebert",
        action="store_true",
        help="Calculate all non-CodeBERT metrics only.",
    )
    parser.add_argument(
        "--no-average-row",
        action="store_true",
        help="Do not append a final row containing metric averages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = [args.reference_column]
    if not args.skip_codebert:
        required_columns.append(args.code_column)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    generated_columns = discover_generated_columns(df, args.generated_columns)
    if not generated_columns:
        raise ValueError("No generated comment columns found.")

    add_text_metrics(df, generated_columns, args.reference_column)

    if not args.skip_codebert:
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        add_codebert_scores(
            df,
            generated_columns,
            args.code_column,
            args.codebert_model,
            args.batch_size,
            args.max_length,
            device,
        )

    if not args.no_average_row:
        df = add_average_row(df, generated_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Scored {len(generated_columns)} generated comment columns.")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()
