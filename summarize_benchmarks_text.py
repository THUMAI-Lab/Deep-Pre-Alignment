#!/usr/bin/env python3

import csv
import re
from pathlib import Path
import argparse
from typing import Dict, Optional


def extract_checkpoint_number(name: str) -> Optional[int]:
    m = re.search(r'checkpoint-(\d+)', name)
    return int(m.group(1)) if m else None


def read_summary_csv(csv_path: Path) -> Dict[str, float]:
    """
    从 summary.csv 中提取三个指标
    """
    scores = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = row["datasets"]
            metric = row["metric"]
            score = float(row["score"])

            if dataset == "mmlu_redux" and metric == "naive_average":
                scores["MMLU"] = score

            elif dataset == "GPQA_diamond" and metric == "accuracy":
                scores["GPQA"] = score

            elif dataset == "math_prm800k_500" and metric == "accuracy":
                scores["Math"] = score

    return scores


def scan_checkpoints(base_dir: Path) -> Dict[int, Dict[str, float]]:
    results = {}

    results_dir = base_dir / "results"

    checkpoint_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-") and d.name.endswith("-text")
    )

    for ckpt_dir in checkpoint_dirs:

        ckpt_id = extract_checkpoint_number(ckpt_dir.name)
        if ckpt_id is None:
            continue

        outputs_dir = ckpt_dir / "outputs"
        summary_files = list(outputs_dir.glob("*_summary.csv"))

        if not summary_files:
            continue

        scores = read_summary_csv(summary_files[0])

        if scores:
            results[ckpt_id] = scores

    return results


def print_table(results: Dict[int, Dict[str, float]]):

    if not results:
        print("No results found.")
        return

    checkpoints = sorted(results.keys())

    print("\n" + "=" * 60)
    print("TEXT BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"{'Checkpoint':<15}{'MMLU':<12}{'GPQA':<12}{'Math':<12}")
    print("-" * 60)

    for ckpt in checkpoints:

        mmlu = results[ckpt].get("MMLU")
        gpqa = results[ckpt].get("GPQA")
        math = results[ckpt].get("Math")

        def fmt(x):
            return f"{x:.2f}" if x is not None else "N/A"

        print(
            f"{ckpt:<15}"
            f"{fmt(mmlu):<12}"
            f"{fmt(gpqa):<12}"
            f"{fmt(math):<12}"
        )

    print("=" * 60)


def save_csv(results: Dict[int, Dict[str, float]], output_file: Path):

    checkpoints = sorted(results.keys())

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["checkpoint", "mmlu", "gpqa", "math"])

        for ckpt in checkpoints:

            row = [
                ckpt,
                results[ckpt].get("MMLU"),
                results[ckpt].get("GPQA"),
                results[ckpt].get("Math"),
            ]

            writer.writerow(row)

    print(f"\nSaved to {output_file}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("-o", "--output", default="text_benchmark_summary.csv")

    args = parser.parse_args()

    base_dir = Path(args.input_folder)

    print(f"Scanning {base_dir} ...")

    results = scan_checkpoints(base_dir)

    print_table(results)

    save_csv(results, Path(args.output))


if __name__ == "__main__":
    main()