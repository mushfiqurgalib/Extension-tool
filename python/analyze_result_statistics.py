import argparse
import csv
import math
import os
from statistics import mean, stdev


SYSTEM_PREFIXES = ("phi", "gpt", "phi_nocontext")
LOWER_IS_BETTER_KEYWORDS = ("GFI",)


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def to_float(value):
    return float(str(value).strip())


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def wilcoxon_signed_rank(x_values, y_values, alternative="two-sided"):
    differences = [y - x for x, y in zip(x_values, y_values)]
    non_zero = [(abs(diff), diff) for diff in differences if diff != 0]
    n = len(non_zero)
    if n == 0:
        return {
            "w_stat": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "n_nonzero": 0,
        }

    sorted_diffs = sorted(non_zero, key=lambda item: item[0])
    ranks = []
    index = 0
    while index < n:
        end = index + 1
        while end < n and sorted_diffs[end][0] == sorted_diffs[index][0]:
            end += 1

        average_rank = (index + 1 + end) / 2.0
        for _, diff in sorted_diffs[index:end]:
            ranks.append((average_rank, diff))
        index = end

    w_plus = sum(rank for rank, diff in ranks if diff > 0)
    w_minus = sum(rank for rank, diff in ranks if diff < 0)

    # Use scipy.stats.wilcoxon for exact/standard p-value calculation if available
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        res = scipy_wilcoxon(y_values, x_values, alternative=alternative)
        p_value = res.pvalue
        w_stat = res.statistic
    except ImportError:
        w_stat = min(w_plus, w_minus)
        expected = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance == 0:
            z_score = 0.0
            p_value = 1.0
        else:
            z_score = (w_plus - expected) / math.sqrt(variance)
            if alternative == "greater":
                p_value = 1.0 - normal_cdf(z_score)
            elif alternative == "less":
                p_value = normal_cdf(z_score)
            else:
                p_value = 2.0 * min(normal_cdf(z_score), 1.0 - normal_cdf(z_score))
                p_value = min(1.0, p_value)

    # Calculate tie-corrected z-score for reporting consistency
    from collections import Counter
    abs_diffs = [abs(d) for d in differences if d != 0]
    tie_counts = Counter(abs_diffs)
    tie_sum = sum(t**3 - t for t in tie_counts.values() if t > 1)
    variance = (n * (n + 1) * (2 * n + 1)) / 24.0 - tie_sum / 48.0
    expected = n * (n + 1) / 4.0
    
    if variance <= 0:
        z_score = 0.0
    else:
        z_score = (w_plus - expected) / math.sqrt(variance)

    total_rank = n * (n + 1) / 2.0
    rank_biserial = (w_plus - w_minus) / total_rank if total_rank else 0.0

    return {
        "w_stat": w_stat,
        "z": z_score,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "n_nonzero": n,
    }



def cohen_dz_from_improvements(improvement_values):
    differences = improvement_values
    if len(differences) < 2:
        return ""
    sd_diff = stdev(differences)
    if sd_diff == 0:
        return ""
    return mean(differences) / sd_diff


def cliffs_delta(x_values, y_values):
    greater = 0
    less = 0
    total = 0
    for x_value in x_values:
        for y_value in y_values:
            if y_value > x_value:
                greater += 1
            elif y_value < x_value:
                less += 1
            total += 1
    if total == 0:
        return ""
    return (greater - less) / total


def cliff_magnitude(delta):
    if delta == "":
        return ""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_holm_adjusted_p_values(rows):
    sorted_rows = sorted(
        [row for row in rows if row.get("wilcoxon_p") != ""],
        key=lambda row: row["wilcoxon_p"],
    )
    total = len(sorted_rows)
    running_max = 0.0

    for index, row in enumerate(sorted_rows):
        adjusted = min(1.0, (total - index) * row["wilcoxon_p"])
        running_max = max(running_max, adjusted)
        row["holm_adjusted_p"] = running_max
        row["holm_significance"] = p_to_stars(running_max)

    for row in rows:
        row.setdefault("holm_adjusted_p", "")
        row.setdefault("holm_significance", "")


def format_number(value):
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.6g}"


def base_metric_name(column_name, prefix):
    metric = column_name[len(prefix) + 1 :]
    return metric.replace(" score", "_score")


def metric_direction(metric_name):
    return "lower" if any(keyword in metric_name for keyword in LOWER_IS_BETTER_KEYWORDS) else "higher"


def is_average_row(row):
    identity_columns = ("repo", "path", "func_name", "code", "docstring")
    return all(not (row.get(column) or "").strip() for column in identity_columns)


def load_rows(input_csv):
    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return [row for row in rows if not is_average_row(row)]


def discover_metric_groups(fieldnames):
    groups = {}
    for column in fieldnames:
        for prefix in SYSTEM_PREFIXES:
            expected_start = f"{prefix}_"
            if column.startswith(expected_start) and is_score_column(column):
                metric = base_metric_name(column, prefix)
                groups.setdefault(metric, {})[prefix] = column
    return {metric: columns for metric, columns in groups.items() if len(columns) >= 2}


def is_score_column(column):
    return column.endswith("_score") or column.endswith(" score")


def paired_values(rows, left_column, right_column):
    left_values = []
    right_values = []
    for row in rows:
        left = row.get(left_column)
        right = row.get(right_column)
        if is_number(left) and is_number(right):
            left_values.append(to_float(left))
            right_values.append(to_float(right))
    return left_values, right_values


def summarize_system(rows, metric, system, column):
    values = [to_float(row[column]) for row in rows if is_number(row.get(column))]
    return {
        "metric": metric,
        "system": system,
        "n": len(values),
        "mean": mean(values) if values else "",
        "std": stdev(values) if len(values) > 1 else "",
        "median": sorted(values)[len(values) // 2] if values else "",
        "min": min(values) if values else "",
        "max": max(values) if values else "",
    }


def summarize_comparison(rows, metric, baseline, candidate, baseline_column, candidate_column):
    baseline_values, candidate_values = paired_values(rows, baseline_column, candidate_column)
    direction = metric_direction(metric)

    if direction == "lower":
        improvement_values = [base - cand for base, cand in zip(baseline_values, candidate_values)]
        wilcoxon = wilcoxon_signed_rank(candidate_values, baseline_values, alternative="greater")
    else:
        improvement_values = [cand - base for base, cand in zip(baseline_values, candidate_values)]
        wilcoxon = wilcoxon_signed_rank(baseline_values, candidate_values, alternative="greater")

    n = len(improvement_values)
    mean_improvement = mean(improvement_values) if improvement_values else ""
    std_improvement = stdev(improvement_values) if len(improvement_values) > 1 else ""
    improved_count = sum(1 for value in improvement_values if value > 0)
    worsened_count = sum(1 for value in improvement_values if value < 0)
    tied_count = sum(1 for value in improvement_values if value == 0)

    dz = cohen_dz_from_improvements(improvement_values)
    cliff = cliffs_delta(baseline_values, candidate_values)
    if direction == "lower" and cliff != "":
        cliff = -cliff

    return {
        "metric": metric,
        "direction": direction,
        "baseline": baseline,
        "candidate": candidate,
        "n_pairs": n,
        "baseline_mean": mean(baseline_values) if baseline_values else "",
        "baseline_std": stdev(baseline_values) if len(baseline_values) > 1 else "",
        "candidate_mean": mean(candidate_values) if candidate_values else "",
        "candidate_std": stdev(candidate_values) if len(candidate_values) > 1 else "",
        "mean_improvement": mean_improvement,
        "std_improvement": std_improvement,
        "improved_count": improved_count,
        "worsened_count": worsened_count,
        "tied_count": tied_count,
        "wilcoxon_w": wilcoxon["w_stat"],
        "wilcoxon_z": wilcoxon["z"],
        "wilcoxon_p": wilcoxon["p_value"],
        "significance": p_to_stars(wilcoxon["p_value"]),
        "rank_biserial_effect": wilcoxon["rank_biserial"],
        "cohen_dz": dz,
        "cliffs_delta": cliff,
        "cliffs_delta_magnitude": cliff_magnitude(cliff),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(row.get(key, "")) for key in fieldnames})


def analyze(input_csv, output_dir):
    rows = load_rows(input_csv)
    if not rows:
        raise RuntimeError("No data rows found after removing the average row.")

    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        fieldnames = csv.DictReader(input_file).fieldnames or []

    metric_groups = discover_metric_groups(fieldnames)
    descriptive_rows = []
    comparison_rows = []

    for metric, columns in sorted(metric_groups.items()):
        for system, column in sorted(columns.items()):
            descriptive_rows.append(summarize_system(rows, metric, system, column))

        comparison_pairs = [
            ("gpt", "phi"),
            ("phi_nocontext", "phi"),
            ("gpt", "phi_nocontext"),
        ]
        for baseline, candidate in comparison_pairs:
            if baseline in columns and candidate in columns:
                comparison_rows.append(
                    summarize_comparison(
                        rows,
                        metric,
                        baseline,
                        candidate,
                        columns[baseline],
                        columns[candidate],
                    )
                )

    add_holm_adjusted_p_values(comparison_rows)

    os.makedirs(output_dir, exist_ok=True)
    descriptive_path = os.path.join(output_dir, "statistical_descriptive_summary.csv")
    comparison_path = os.path.join(output_dir, "statistical_pairwise_tests.csv")

    write_csv(
        descriptive_path,
        descriptive_rows,
        ["metric", "system", "n", "mean", "std", "median", "min", "max"],
    )
    write_csv(
        comparison_path,
        comparison_rows,
        [
            "metric",
            "direction",
            "baseline",
            "candidate",
            "n_pairs",
            "baseline_mean",
            "baseline_std",
            "candidate_mean",
            "candidate_std",
            "mean_improvement",
            "std_improvement",
            "improved_count",
            "worsened_count",
            "tied_count",
            "wilcoxon_w",
            "wilcoxon_z",
            "wilcoxon_p",
            "significance",
            "holm_adjusted_p",
            "holm_significance",
            "rank_biserial_effect",
            "cohen_dz",
            "cliffs_delta",
            "cliffs_delta_magnitude",
        ],
    )

    return descriptive_path, comparison_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute standard deviations, Wilcoxon tests, and effect sizes for paired metric results."
    )
    parser.add_argument(
        "--input",
        default=r"E:\Extension\Extension-tool\Result\Result_summary - python_sample_with_scores.csv",
        help="Input CSV containing per-sample metric scores and a final average row.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"E:\Extension\Extension-tool\Result",
        help="Directory where statistical result CSV files will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    descriptive_output, comparison_output = analyze(args.input, args.output_dir)
    print(f"Wrote descriptive statistics: {descriptive_output}")
    print(f"Wrote pairwise statistical tests: {comparison_output}")
