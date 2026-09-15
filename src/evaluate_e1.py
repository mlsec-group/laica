import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import click
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from util import log

CONFIDENCE = 1.644850

METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "auc", "average_precision"]

RESULT_COLUMNS = [
    "task",
    "variant",
    "fraction",
    "seed",
    "train_samples",
    "test_samples",
    "learning_rate",
    "num_augmentations",
    "metric",
    "performance",
] + METRIC_COLUMNS

RUN_COLUMNS = ["learning_rate", "num_augmentations", "train_samples"]

SUMMARY_COLUMNS = [
    "task",
    "variant",
    "fraction",
    "train_samples",
    "metric",
    "n",
    "mean",
    "std",
    "ci",
]


def read_predictions(path, task):
    y_true = []
    y_pred = []

    with path.open() as f:
        for line in f:
            true, pred = line.split()
            if task == "function_similarity":
                y_true.append(float(true))
                y_pred.append(float(pred))
            else:
                y_true.append(int(float(true)))
                y_pred.append(int(float(pred)))

    return y_true, y_pred


def cell_metrics(task, y_true, y_pred):
    if task == "function_similarity":
        return "auc", {
            "auc": roc_auc_score(y_true, y_pred),
            "average_precision": average_precision_score(y_true, y_pred),
        }

    return "accuracy", {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect(e1_dir):
    rows = []

    for predictions_path in sorted(e1_dir.glob("*/*/*/*/predictions.csv")):
        cell = predictions_path.parent
        task, variant, fraction, seed = cell.relative_to(e1_dir).parts

        if not (fraction.isdigit() and seed.isdigit()):
            log(f"[!] Skipping {cell}: not a <task>/<variant>/<fraction>/<seed> directory")
            continue

        y_true, y_pred = read_predictions(predictions_path, task)
        if len(y_true) == 0:
            log(f"[!] Skipping {cell}: predictions.csv is empty")
            continue

        metric, values = cell_metrics(task, y_true, y_pred)

        row = {
            "task": task,
            "variant": variant,
            "fraction": int(fraction),
            "seed": int(seed),
            "test_samples": len(y_true),
            "metric": metric,
            "performance": values[metric],
        }

        for name in ["objective.json", "metrics.json"]:
            record = read_json(cell / name)
            row.update({column: record[column] for column in RUN_COLUMNS if column in record})

        row.update(values)
        rows.append(row)

    missing = sum(1 for row in rows if "train_samples" not in row)
    if missing:
        log(f"[!] {missing} of {len(rows)} cells record no train_samples, re-run them with the current e1.py to fill the x-axis")

    return rows


def baseline_sizes(rows):
    sizes = defaultdict(list)

    for row in rows:
        if row["variant"] == "baseline" and "train_samples" in row:
            sizes[(row["task"], row["fraction"])].append(row["train_samples"])

    return sizes


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["variant"], row["fraction"])].append(row)

    sizes = baseline_sizes(rows)
    summary = []

    for (task, variant, fraction), cells in groups.items():
        values = [cell["performance"] for cell in cells]
        n = len(values)
        mean = sum(values) / n
        squares = sum((value - mean) ** 2 for value in values)

        train_samples = sizes.get((task, fraction)) or [cell["train_samples"] for cell in cells if "train_samples" in cell]

        summary.append({
            "task": task,
            "variant": variant,
            "fraction": fraction,
            "train_samples": sum(train_samples) / len(train_samples) if train_samples else None,
            "metric": cells[0]["metric"],
            "n": n,
            "mean": mean,
            "std": math.sqrt(squares / (n - 1)) if n > 1 else 0.0,
            "ci": CONFIDENCE * math.sqrt(squares / n) / math.sqrt(n),
        })

    return sorted(summary, key=lambda row: (row["task"], row["variant"], -row["fraction"]))


def report_incomplete(summary):
    expected = defaultdict(int)
    for row in summary:
        expected[row["task"]] = max(expected[row["task"]], row["n"])

    for row in summary:
        if row["n"] < expected[row["task"]]:
            log(f"[!] {row['task']} {row['variant']} 1/{row['fraction']}: {row['n']} seeds, expected {expected[row['task']]}")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[+] Wrote {len(rows)} rows to {path}")


@click.command()
@click.option("--data-path", "-d", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), help="Data directory holding the e1 results.")
@click.option("--output-dir", "-o", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Where to write the csv files.")
def run(data_path, output_dir):
    e1_dir = data_path / "e1"

    if not e1_dir.exists():
        raise FileNotFoundError(f"No e1 results at {e1_dir}")

    if output_dir is None:
        output_dir = data_path / "results"

    rows = collect(e1_dir)

    if len(rows) == 0:
        raise ValueError(f"No completed cells under {e1_dir}")

    rows = sorted(rows, key=lambda row: (row["task"], row["variant"], -row["fraction"], row["seed"]))
    summary = summarize(rows)

    report_incomplete(summary)

    write_csv(output_dir / "results.csv", RESULT_COLUMNS, rows)
    write_csv(output_dir / "summary.csv", SUMMARY_COLUMNS, summary)


if __name__ == "__main__":
    run()
