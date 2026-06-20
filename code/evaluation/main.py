"""
Evaluation Script for HackerRank Orchestrate — Claim Review Pipeline

Purpose:
  Run the system against dataset/sample_claims.csv (which includes expected outputs),
  compare predictions against ground truth, and produce evaluation_report.md.

Usage:
  python -m evaluation.main
  python -m evaluation.main --sample dataset/sample_claims.csv --output evaluation_report.md
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path so we can import claim_review
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from claim_review.pipeline import process_claim
from claim_review.csv_io import read_user_history, read_requirements

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
# Approximate cost estimates (USD per 1K tokens, GPT-4o-mini tier pricing)
COST_PER_INPUT_1K_TOKENS = 0.00015
COST_PER_OUTPUT_1K_TOKENS = 0.0006
# Approximate tokens per LLM call (input + output combined)
APPROX_INPUT_TOKENS_PER_CALL = 800
APPROX_OUTPUT_TOKENS_PER_CALL = 300

# Fields we evaluate accuracy on
EVAL_FIELDS = ["claim_status", "issue_type", "object_part", "severity", "valid_image"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize(value: str) -> str:
    """Normalize a value for comparison: lowercase, strip whitespace."""
    return str(value).strip().lower()


def _count_images(image_paths_str: str) -> int:
    """Count the number of images in a semicolon-separated path string."""
    paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    return len(paths)


def _estimate_model_calls(num_claims: int, total_images: int) -> int:
    """
    Estimate the total number of LLM API calls made during pipeline execution.
    
    Per claim the pipeline makes approximately:
      1. Claim Extraction       — 1 call (text-only)
      2. Image Qualification    — 1 call (vision, all images bundled)
      3. Object Inspection      — N calls (1 per image)
    Total ≈ 2 + total_images
    """
    return (2 * num_claims) + total_images


def _estimate_cost(model_calls: int) -> float:
    """Estimate approximate USD cost based on model call count."""
    input_cost = (model_calls * APPROX_INPUT_TOKENS_PER_CALL / 1000) * COST_PER_INPUT_1K_TOKENS
    output_cost = (model_calls * APPROX_OUTPUT_TOKENS_PER_CALL / 1000) * COST_PER_OUTPUT_1K_TOKENS
    return input_cost + output_cost


# ---------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------

def evaluate(
    sample_path: str,
    history_path: str,
    requirements_path: str
) -> Tuple[List[Dict], List[Dict], float]:
    """
    Run the pipeline on every row in sample_claims.csv, collect predictions,
    and return (predictions, ground_truths, elapsed_seconds).
    """
    # Load sample claims (these contain expected outputs)
    rows = []
    with open(sample_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Pre-load supporting datasets
    user_history_map = read_user_history(history_path)
    requirements = read_requirements(requirements_path)

    # Inject requirements into resolver cache
    import claim_review.requirements_resolver as rr
    rr._CACHED_REQUIREMENTS = requirements

    predictions = []
    ground_truths = []

    start_time = time.time()

    for idx, row in enumerate(rows):
        user_id = row.get("user_id", "")
        user_claim = row.get("user_claim", "")
        claim_object = row.get("claim_object", "")
        image_paths_str = row.get("image_paths", "")
        image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]

        print(f"[{idx+1}/{len(rows)}] Evaluating claim for user {user_id}...")

        # Store ground truth
        ground_truths.append({
            "user_id": user_id,
            "claim_status": _normalize(row.get("claim_status", "")),
            "issue_type": _normalize(row.get("issue_type", "")),
            "object_part": _normalize(row.get("object_part", "")),
            "severity": _normalize(row.get("severity", "")),
            "valid_image": _normalize(row.get("valid_image", "")),
        })

        try:
            result = process_claim(
                user_id=user_id,
                user_claim=user_claim,
                claim_object=claim_object,
                image_paths=image_paths,
                evidence_requirement=None
            )
            # Serialize prediction to comparable dict
            predictions.append({
                "user_id": user_id,
                "claim_status": _normalize(result.claim_status.value if hasattr(result.claim_status, "value") else str(result.claim_status)),
                "issue_type": _normalize(result.issue_type.value if hasattr(result.issue_type, "value") else str(result.issue_type)),
                "object_part": _normalize(result.object_part),
                "severity": _normalize(result.severity.value if hasattr(result.severity, "value") else str(result.severity)),
                "valid_image": _normalize(str(result.valid_image)),
            })
        except Exception as e:
            print(f"  [Error] Pipeline failed for user {user_id}: {e}")
            traceback.print_exc()
            # Record a failed prediction so row counts stay aligned
            predictions.append({
                "user_id": user_id,
                "claim_status": "error",
                "issue_type": "error",
                "object_part": "error",
                "severity": "error",
                "valid_image": "error",
            })

    elapsed = time.time() - start_time
    return predictions, ground_truths, elapsed


def compute_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict]
) -> Dict:
    """Compute per-field accuracy and collect failure details."""
    total = len(ground_truths)
    if total == 0:
        return {"total": 0, "fields": {}, "failures": []}

    field_correct = {f: 0 for f in EVAL_FIELDS}
    failures = []

    for i in range(total):
        pred = predictions[i]
        truth = ground_truths[i]
        row_failures = {}

        for field in EVAL_FIELDS:
            if pred[field] == truth[field]:
                field_correct[field] += 1
            else:
                row_failures[field] = {
                    "expected": truth[field],
                    "predicted": pred[field]
                }

        if row_failures:
            failures.append({
                "user_id": truth["user_id"],
                "mismatches": row_failures
            })

    field_accuracy = {}
    for field in EVAL_FIELDS:
        field_accuracy[field] = {
            "correct": field_correct[field],
            "total": total,
            "accuracy": round(field_correct[field] / total * 100, 2)
        }

    return {
        "total": total,
        "fields": field_accuracy,
        "failures": failures
    }


# ---------------------------------------------------------
# Report Generation
# ---------------------------------------------------------

def generate_report(
    metrics: Dict,
    total_images: int,
    model_calls: int,
    estimated_cost: float,
    elapsed_seconds: float,
    output_path: str
) -> None:
    """Generate evaluation_report.md with all required metrics."""
    total = metrics["total"]
    fields = metrics["fields"]
    failures = metrics["failures"]
    timestamp = datetime.now().isoformat()

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Rows Evaluated | {total} |")
    lines.append(f"| Total Images Processed | {total_images} |")
    lines.append(f"| Approximate Model Calls | {model_calls} |")
    lines.append(f"| Approximate Cost (USD) | ${estimated_cost:.4f} |")
    lines.append(f"| Approximate Latency | {elapsed_seconds:.1f}s ({elapsed_seconds/total:.1f}s per claim) |")
    lines.append("")

    # Accuracy by field
    lines.append("## Accuracy by Field")
    lines.append("")
    lines.append("| Field | Correct | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    for field in EVAL_FIELDS:
        f = fields[field]
        lines.append(f"| {field} | {f['correct']} | {f['total']} | {f['accuracy']}% |")
    lines.append("")

    # Overall accuracy (average across fields)
    avg_accuracy = sum(f["accuracy"] for f in fields.values()) / len(fields) if fields else 0
    lines.append(f"**Overall Average Accuracy: {avg_accuracy:.2f}%**")
    lines.append("")

    # Failure breakdown
    lines.append("## Failure Breakdown")
    lines.append("")
    if not failures:
        lines.append("No failures detected. All predictions matched ground truth.")
    else:
        lines.append(f"**{len(failures)} out of {total} claims had at least one mismatch.**")
        lines.append("")
        for fail in failures:
            lines.append(f"### User: `{fail['user_id']}`")
            lines.append("")
            lines.append("| Field | Expected | Predicted |")
            lines.append("|---|---|---|")
            for field, detail in fail["mismatches"].items():
                lines.append(f"| {field} | {detail['expected']} | {detail['predicted']} |")
            lines.append("")

    # Cost breakdown
    lines.append("## Cost Estimation Details")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input tokens per call (approx) | {APPROX_INPUT_TOKENS_PER_CALL} |")
    lines.append(f"| Output tokens per call (approx) | {APPROX_OUTPUT_TOKENS_PER_CALL} |")
    lines.append(f"| Cost per 1K input tokens | ${COST_PER_INPUT_1K_TOKENS} |")
    lines.append(f"| Cost per 1K output tokens | ${COST_PER_OUTPUT_1K_TOKENS} |")
    lines.append(f"| Model calls (2 per claim + 1 per image) | {model_calls} |")
    lines.append("")

    # Write report
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Evaluation report written to {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Claim Review Pipeline against sample data")
    parser.add_argument(
        "--sample",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "dataset", "sample_claims.csv"),
        help="Path to sample_claims.csv (with expected outputs)"
    )
    parser.add_argument(
        "--history",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "dataset", "user_history.csv"),
        help="Path to user_history.csv"
    )
    parser.add_argument(
        "--requirements",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "dataset", "evidence_requirements.csv"),
        help="Path to evidence_requirements.csv"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_report.md"),
        help="Path to write evaluation_report.md"
    )
    args = parser.parse_args()

    # Resolve paths
    sample_path = os.path.abspath(args.sample)
    history_path = os.path.abspath(args.history)
    requirements_path = os.path.abspath(args.requirements)
    output_path = os.path.abspath(args.output)

    print(f"Sample claims: {sample_path}")
    print(f"User history:  {history_path}")
    print(f"Requirements:  {requirements_path}")
    print(f"Output report: {output_path}")
    print()

    # Count total images across all sample claims
    total_images = 0
    with open(sample_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_images += _count_images(row.get("image_paths", ""))

    # Run evaluation
    predictions, ground_truths, elapsed = evaluate(sample_path, history_path, requirements_path)

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Estimate costs
    model_calls = _estimate_model_calls(len(ground_truths), total_images)
    estimated_cost = _estimate_cost(model_calls)

    # Generate report
    generate_report(metrics, total_images, model_calls, estimated_cost, elapsed, output_path)

    print(f"\nDone. {metrics['total']} claims evaluated in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
