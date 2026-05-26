#!/usr/bin/env python3
"""
Benchmark Score Summarizer

This script automatically extracts and summarizes evaluation scores from benchmark results.
It scans through checkpoint folders and collects scores from various benchmark evaluation files.

Usage:
    python summarize_benchmarks.py <input_folder>

Example:
    python summarize_benchmarks.py /path/to/output/v0-20251114-174720
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def extract_checkpoint_number(checkpoint_path: str) -> Optional[int]:
    """Extract checkpoint number from path like 'checkpoint-1000'."""
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    return int(match.group(1)) if match else None


def read_csv_score(csv_path: str, score_column: str = "Overall", category_column: Optional[str] = None, category_value: Optional[str] = None) -> Optional[float]:
    """Read score from CSV file, looking for the specified column.
    
    Args:
        csv_path: Path to CSV file
        score_column: Column name to extract score from
        category_column: Optional column name to filter by (e.g., "Category")
        category_value: Optional value to match in category_column (e.g., "Overall")
    """
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # If category filtering is specified, check if this row matches
                if category_column and category_value:
                    if row.get(category_column) != category_value:
                        continue
                
                # Extract score from the specified column
                if score_column in row:
                    score = row[score_column]
                    if score and score != "":
                        return float(score)
        return None
    except Exception as e:
        print(f"Warning: Could not read {csv_path}: {e}")
        return None


def read_json_score(json_path: str, score_key: str = "Final Score Norm") -> Optional[float]:
    """Read score from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if score_key in data:
                return float(data[score_key])
        return None
    except Exception as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return None


def count_matching_failures(result_file_path: Path) -> int:
    """Count occurrences of 'no GPT-based answer matching' failures in result file.
    
    Tries to read from Excel file first (more reliable), falls back to pkl file if needed.
    """
    # Try Excel file first
    xlsx_path = result_file_path.parent / (result_file_path.stem + '.xlsx')
    if xlsx_path.exists():
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path)
            if 'log' in df.columns:
                target_message = 'no GPT-based answer matching under `exact_matching` policy.'
                count = df['log'].astype(str).str.contains(target_message, na=False).sum()
                return int(count)
        except Exception as e:
            print(f"Warning: Could not read Excel file {xlsx_path}: {e}")
    
    # Fall back to pkl file
    try:
        import pickle
        with open(result_file_path, 'rb') as f:
            data = pickle.load(f)
        
        target_message = 'no GPT-based answer matching under `exact_matching` policy.'
        count = 0
        
        for key, value in data.items():
            if isinstance(value, dict) and 'log' in value:
                if target_message in value['log']:
                    count += 1
        
        return count
    except Exception as e:
        print(f"Warning: Could not read pkl file {result_file_path}: {e}")
        return 0


def scan_checkpoint_results(checkpoint_path: Path) -> tuple[Dict[str, float], Dict[str, int]]:
    """Scan a checkpoint folder and extract all benchmark scores and matching failure counts.
    
    Returns:
        Tuple of (scores_dict, failures_dict)
    """
    scores = {}
    failures = {}
    
    # Look for results in output subdirectories
    output_dir = checkpoint_path / "output"
    if not output_dir.exists():
        return scores, failures
    
    # Find the model output directory (e.g., DuplexThinkerS2ForwardvLLMPrefixCustom)
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        # Define benchmark patterns and their extraction methods
        # Format: (name, file_pattern, file_type, score_column, category_column, category_value, pkl_pattern)
        benchmark_patterns = [
            ("MMBench", "*_MMBench_DEV_EN_V11_acc.csv", "csv", "Overall", None, None, "*_MMBench_DEV_EN_V11_gpt-4o_result.pkl"),
            ("MMStar", "*_MMStar_acc.csv", "csv", "Overall", None, None, "*_MMStar_gpt-4o_result.pkl"),
            ("MMMU", "*_MMMU_DEV_VAL_acc.csv", "csv", "Overall", "split", "validation", "*_MMMU_DEV_VAL_gpt-4o_result.pkl"),
            ("MathVista", "*_MathVista_MINI_gpt-4o_score.csv", "csv", "acc", None, None, "*_MathVista_MINI_gpt-4o.pkl"),
            ("OCRBench", "*_OCRBench_score.json", "json", "Final Score Norm", None, None, None),
            ("MMVet", "*_MMVet_gpt-4-turbo_score.csv", "csv", "acc", "Category", "Overall", "*_MMVet_gpt-4-turbo.pkl"),
            ("AI2D", "*AI2D*_acc.csv", "csv", "Overall", None, None, "*AI2D_TEST_gpt-4o_result.pkl"),
            ("SEEDBench2_Plus", "*SEEDBench2*_acc.csv", "csv", "Overall", None, None, "*SEEDBench2_Plus_gpt-4o_result.pkl"),
            ("MathVision", "*MathVision*_gpt-4o_score.csv", "csv", "acc", None, None, "*MathVision_gpt-4o.pkl"),
        ]
        
        for bench_name, pattern, file_type, key, cat_col, cat_val, pkl_pattern in benchmark_patterns:
            matching_files = list(model_dir.glob(pattern))
            if matching_files:
                file_path = matching_files[0]
                if file_type == "csv":
                    score = read_csv_score(str(file_path), key, cat_col, cat_val)
                elif file_type == "json":
                    score = read_json_score(str(file_path), key)
                else:
                    score = None
                
                if score is not None:
                    scores[bench_name] = score
            
            # Count matching failures from pkl files if pattern is provided
            if pkl_pattern:
                pkl_files = list(model_dir.glob(pkl_pattern))
                if pkl_files:
                    failure_count = count_matching_failures(pkl_files[0])
                    if failure_count > 0 or pkl_files:  # Record even if 0, if file exists
                        failures[bench_name] = failure_count
    
    return scores, failures


def summarize_benchmarks(input_folder: str) -> tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, int]]]:
    """
    Scan input folder for all checkpoints and collect benchmark scores and matching failures.
    
    Args:
        input_folder: Path to the folder containing checkpoint results
        
    Returns:
        Tuple of (scores_dict, failures_dict) mapping checkpoint numbers to their data
    """
    input_path = Path(input_folder)
    results_dir = input_path / "results"
    
    if not results_dir.exists():
        results_dir = input_path
    
    all_results = {}
    all_failures = {}
    
    # Scan all checkpoint directories
    checkpoint_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_num = extract_checkpoint_number(checkpoint_dir.name)
        if checkpoint_num is None:
            continue
        
        print(f"Processing {checkpoint_dir.name}...")
        scores, failures = scan_checkpoint_results(checkpoint_dir)
        
        if scores:
            all_results[checkpoint_num] = scores
        if failures:
            all_failures[checkpoint_num] = failures
    
    return all_results, all_failures


def normalize_score(score: float) -> float:
    """Normalize score to 0-100 range."""
    if 0 <= score <= 1:
        return score * 100
    return score


def calculate_checkpoint_average(scores: Dict[str, float], all_benchmarks: set) -> Optional[float]:
    """Calculate average score across all benchmarks for a checkpoint.
    
    Args:
        scores: Dictionary of benchmark scores for a checkpoint
        all_benchmarks: Set of all benchmark names that should be present
        
    Returns:
        Average score, or None if not all benchmarks are complete
    """
    if not scores:
        return None
    
    # Only calculate average if all benchmarks are present
    if set(scores.keys()) != all_benchmarks:
        return None
    
    normalized_scores = [normalize_score(score) for score in scores.values()]
    return sum(normalized_scores) / len(normalized_scores)


def print_summary_table(results: Dict[int, Dict[str, float]]):
    """Print a formatted table of all benchmark scores."""
    if not results:
        print("No results found.")
        return
    
    # Get all unique benchmark names
    all_benchmarks = set()
    for scores in results.values():
        all_benchmarks.update(scores.keys())
    
    # Define the desired order (matching the image)
    benchmark_order = ["MMBench", "MMStar", "MMMU", "MathVista", "AI2D", "OCRBench", "MMVet", "SEEDBench2_Plus", "MathVision" ]
    # Only include benchmarks that are actually present in results
    all_benchmarks_sorted = [b for b in benchmark_order if b in all_benchmarks]
    # Add any benchmarks not in the predefined order at the end
    all_benchmarks_sorted += sorted([b for b in all_benchmarks if b not in benchmark_order])
    
    # Get all checkpoint numbers
    checkpoints = sorted(results.keys())
    
    # Calculate averages for each checkpoint (only if all benchmarks complete)
    averages = {}
    for ckpt, scores in results.items():
        avg = calculate_checkpoint_average(scores, all_benchmarks)
        if avg is not None:
            averages[ckpt] = avg
    
    # Print header
    print("\n" + "="*120)
    print("BENCHMARK SUMMARY")
    print("="*120)
    print(f"{'Checkpoint':<15}", end="")
    for bench in all_benchmarks_sorted:
        print(f"{bench:<15}", end="")
    print(f"{'Average':<15}")
    print("-"*120)
    
    # Print scores for each checkpoint
    for ckpt in checkpoints:
        print(f"{ckpt:<15}", end="")
        for bench in all_benchmarks_sorted:
            score = results[ckpt].get(bench)
            if score is not None:
                # Convert to percentage if score is between 0 and 1
                if 0 <= score <= 1:
                    score_str = f"{score*100:.2f}%"
                else:
                    score_str = f"{score:.2f}"
                print(f"{score_str:<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        
        # Print average (N/A if not all benchmarks complete)
        if ckpt in averages:
            print(f"{averages[ckpt]:.2f}")
        else:
            print(f"{'N/A':<15}")
    
    print("="*120)


def print_matching_failures_table(failures: Dict[int, Dict[str, int]]):
    """Print a formatted table of matching failure counts."""
    if not failures:
        print("\nNo matching failure data found.")
        return
    
    # Get all unique benchmark names
    all_benchmarks = set()
    for failure_counts in failures.values():
        all_benchmarks.update(failure_counts.keys())
    
    # Define the desired order (matching the benchmark order)
    benchmark_order = ["MMBench", "MMStar", "MMMU", "MathVista", "AI2D", "OCRBench", "MMVet", "SEEDBench2_Plus", "MathVision" ]
    all_benchmarks_sorted = [b for b in benchmark_order if b in all_benchmarks]
    all_benchmarks_sorted += sorted([b for b in all_benchmarks if b not in benchmark_order])
    
    # Get all checkpoint numbers
    checkpoints = sorted(failures.keys())
    
    # Print header
    print("\n" + "="*120)
    print("GPT MATCHING FAILURES (no GPT-based answer matching under `exact_matching` policy)")
    print("="*120)
    print(f"{'Checkpoint':<15}", end="")
    for bench in all_benchmarks_sorted:
        print(f"{bench:<15}", end="")
    print()
    print("-"*120)
    
    # Print failure counts for each checkpoint
    for ckpt in checkpoints:
        print(f"{ckpt:<15}", end="")
        for bench in all_benchmarks_sorted:
            count = failures[ckpt].get(bench, 0) if ckpt in failures else 0
            print(f"{count:<15}", end="")
        print()
    
    print("="*120)
    
    # Calculate and print total failures per benchmark
    print("\nTOTAL FAILURES PER BENCHMARK:")
    print("-"*100)
    for bench in all_benchmarks_sorted:
        total = sum(failures[ckpt].get(bench, 0) for ckpt in checkpoints if ckpt in failures)
        print(f"{bench:<15}: {total}")
    print("="*100)


def delete_problematic_files(checkpoint_path: Path, bench_name: str, failure_count: int, threshold: int = 10) -> List[str]:
    """Delete gpt-4o and score files for benchmarks with too many matching failures.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        bench_name: Benchmark name
        failure_count: Number of matching failures
        threshold: Threshold for deletion (default: 10)
        
    Returns:
        List of deleted file paths
    """
    if failure_count <= threshold:
        return []
    
    deleted_files = []
    output_dir = checkpoint_path / "output"
    if not output_dir.exists():
        return deleted_files
    
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    # Define file patterns to delete for each benchmark
    patterns_map = {
        "MMBench": [
            "*_MMBench_DEV_EN_V11_gpt-4o_result.pkl",
            "*_MMBench_DEV_EN_V11_gpt-4o_result.xlsx",
            "*_MMBench_DEV_EN_V11_acc.csv"
        ],
        "MMStar": [
            "*_MMStar_gpt-4o_result.pkl",
            "*_MMStar_gpt-4o_result.xlsx",
            "*_MMStar_acc.csv"
        ],
        "MMMU": [
            "*_MMMU_DEV_VAL_gpt-4o_result.pkl",
            "*_MMMU_DEV_VAL_gpt-4o_result.xlsx",
            "*_MMMU_DEV_VAL_acc.csv"
        ],
        "MathVista": [
            "*_MathVista_MINI_gpt-4o.pkl",
            "*_MathVista_MINI_gpt-4o.xlsx",
            "*_MathVista_MINI_gpt-4o_score.csv"
        ],
        "MMVet": [
            "*_MMVet_gpt-4-turbo.pkl",
            "*_MMVet_gpt-4-turbo.xlsx",
            "*_MMVet_gpt-4-turbo_score.csv",
            "*_MMVet_gpt-4-turbo_score_fine.csv"
        ],
    }
    
    if bench_name not in patterns_map:
        return deleted_files
    
    for model_dir in model_dirs:
        for pattern in patterns_map[bench_name]:
            matching_files = list(model_dir.glob(pattern))
            for file_path in matching_files:
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    print(f"  Deleted: {file_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not delete {file_path}: {e}")
    
    return deleted_files


def cleanup_high_failure_files(input_folder: str, failures: Dict[int, Dict[str, int]], threshold: int = 10) -> Dict[str, List[str]]:
    """Delete files for all benchmarks with matching failures exceeding threshold.
    
    Args:
        input_folder: Path to the folder containing checkpoint results
        failures: Dictionary mapping checkpoint numbers to their failure counts
        threshold: Threshold for deletion (default: 10)
        
    Returns:
        Dictionary mapping checkpoint-benchmark to list of deleted files
    """
    input_path = Path(input_folder)
    results_dir = input_path / "results"
    
    if not results_dir.exists():
        return {}
    
    all_deleted = {}
    checkpoint_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_num = extract_checkpoint_number(checkpoint_dir.name)
        if checkpoint_num is None or checkpoint_num not in failures:
            continue
        
        for bench_name, failure_count in failures[checkpoint_num].items():
            if failure_count > threshold:
                print(f"\n⚠ {checkpoint_dir.name} - {bench_name}: {failure_count} failures (>{threshold})")
                deleted = delete_problematic_files(checkpoint_dir, bench_name, failure_count, threshold)
                if deleted:
                    key = f"{checkpoint_dir.name}-{bench_name}"
                    all_deleted[key] = deleted
    
    return all_deleted


def save_summary_csv(results: Dict[int, Dict[str, float]], output_file: str):
    """Save summary to a CSV file."""
    if not results:
        print("No results to save.")
        return
    
    # Get all unique benchmark names
    all_benchmarks = set()
    for scores in results.values():
        all_benchmarks.update(scores.keys())
    
    # Define the desired order (matching the image)
    benchmark_order = ["MMBench", "MMStar", "MMMU", "MathVista", "AI2D", "OCRBench", "MMVet", "SEEDBench2_Plus", "MathVision" ]
    # Only include benchmarks that are actually present in results
    all_benchmarks_sorted = [b for b in benchmark_order if b in all_benchmarks]
    # Add any benchmarks not in the predefined order at the end
    all_benchmarks_sorted += sorted([b for b in all_benchmarks if b not in benchmark_order])
    
    # Get all checkpoint numbers
    checkpoints = sorted(results.keys())
    
    # Calculate averages for each checkpoint (only if all benchmarks complete)
    averages = {}
    for ckpt, scores in results.items():
        avg = calculate_checkpoint_average(scores, all_benchmarks)
        if avg is not None:
            averages[ckpt] = avg
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["checkpoint"] + all_benchmarks_sorted + ["average"]
        writer.writerow(header)
        
        # Write data rows
        for ckpt in checkpoints:
            row = [ckpt]
            for bench in all_benchmarks_sorted:
                score = results[ckpt].get(bench)
                if score is not None:
                    row.append(f"{score:.4f}")
                else:
                    row.append("")
            
            # Add average
            if ckpt in averages:
                row.append(f"{averages[ckpt]:.4f}")
            else:
                row.append("")
            
            writer.writerow(row)
    
    print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark evaluation scores from checkpoint results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_benchmarks.py /path/to/v0-20251114-174720
  python summarize_benchmarks.py /path/to/v0-20251114-174720 --output summary.csv
  python summarize_benchmarks.py /path/to/v0-20251114-174720 --delete-threshold 10
        """
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing checkpoint results (e.g., v0-20251114-174720)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: benchmark_summary.csv)",
        default="benchmark_summary.csv"
    )
    parser.add_argument(
        "--delete-threshold",
        type=int,
        help="Delete gpt and score files when matching failures exceed this threshold (default: disabled)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return 1
    
    # Collect results
    print(f"Scanning folder: {args.input_folder}")
    print("-"*100)
    results, failures = summarize_benchmarks(args.input_folder)
    
    if not results:
        print("No benchmark results found.")
        return 1
    
    # Print summary table
    print_summary_table(results)
    
    # Print matching failures table
    if failures:
        print_matching_failures_table(failures)
    
    # Delete problematic files if threshold is specified
    if args.delete_threshold is not None and failures:
        print(f"\n{'='*120}")
        print(f"CLEANING UP FILES WITH MORE THAN {args.delete_threshold} MATCHING FAILURES")
        print("="*120)
        deleted_files = cleanup_high_failure_files(args.input_folder, failures, args.delete_threshold)
        if deleted_files:
            print(f"\n✓ Deleted files for {len(deleted_files)} checkpoint-benchmark combinations.")
        else:
            print(f"\n✓ No files needed deletion (all below threshold).")
    
    # Save to CSV
    save_summary_csv(results, args.output)
    
    print(f"\n✓ Summary complete! Found {len(results)} checkpoints.")
    return 0


if __name__ == "__main__":
    exit(main())

