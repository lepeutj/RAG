"""Evaluate retrieval against the checked-in reference dataset.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --top-k 3 --json-output evaluation-results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.evaluation import evaluate_retrieval, load_evaluation_cases  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a reference dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/retrieval_cases.json"),
        help="Path to the JSON reference dataset.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks retrieved for each question.")
    parser.add_argument("--json-output", type=Path, help="Optional path for a machine-readable report.")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.top_k
    cases = load_evaluation_cases(args.dataset)
    pipeline = RAGPipeline(settings)
    report = evaluate_retrieval(cases, lambda question, k: pipeline.retrieve(question, top_k=k), top_k)

    print(f"Cases evaluated: {report['cases_evaluated']}")
    print(f"Source recall@{top_k}: {report['source_recall_at_k']:.3f}")
    print(f"Mean reciprocal rank: {report['mean_reciprocal_rank']:.3f}")
    for result in report["results"]:
        print(
            f"- {result['case_id']}: recall={result['source_recall']:.1f}, "
            f"reciprocal_rank={result['reciprocal_rank']:.3f}, sources={result['retrieved_sources']}"
        )

    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
