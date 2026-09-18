"""Compare two matched retrieval runs with paired bootstrap differences.

Usage: python scripts/compare_runs.py baseline.json candidate.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def compare(baseline: dict, candidate: dict, seed: int = 42) -> dict:
    for field in ("dataset_sha256", "corpus_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Cannot compare different {field} values")
    old = {case["id"]: case for case in baseline["cases"]}
    new = {case["id"]: case for case in candidate["cases"]}
    if set(old) != set(new):
        raise ValueError("Runs contain different question IDs")
    answerable = sorted(case_id for case_id, case in old.items() if case["evidence"])
    rng = random.Random(seed)
    result = {"answerable_cases": len(answerable), "changes": {}}
    for metric in ("evidence_recall_at_k", "evidence_hit_at_k", "reciprocal_rank"):
        differences = [float(new[case_id]["retrieval"][metric]) -
                       float(old[case_id]["retrieval"][metric]) for case_id in answerable]
        if not differences:
            continue
        draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
        result["changes"][metric] = {
            "delta_mean": statistics.mean(differences),
            "paired_bootstrap_ci95": [draws[50], draws[1950]],
            "improved": [case_id for case_id, delta in zip(answerable, differences) if delta > 0],
            "worsened": [case_id for case_id, delta in zip(answerable, differences) if delta < 0],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    print(json.dumps(compare(baseline, candidate), indent=2))


if __name__ == "__main__":
    main()
