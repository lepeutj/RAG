"""Create and score structured human reviews of generated RAG answers.

Usage:
  python scripts/review_answers.py init RUN.json REVIEW.json --reviewer alice
  python scripts/review_answers.py score RUN.json REVIEW.json [REVIEW2.json ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.eval_metrics import interval

FIELDS = ("correct", "grounded", "complete", "citation_correct", "abstained")


def run_hash(report: dict) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()


def make_template(report: dict, reviewer: str) -> dict:
    if any("answer" not in case for case in report["cases"]):
        raise ValueError("Run evaluation with --generate before making a review template.")
    return {
        "reviewer": reviewer,
        "dataset_sha256": report["dataset_sha256"],
        "run_sha256": run_hash(report),
        "judgments": [{"id": case["id"], **{name: None for name in FIELDS}, "notes": ""}
                      for case in report["cases"]],
    }


def score_review(report: dict, review: dict) -> dict:
    if any("answer" not in case for case in report["cases"]):
        raise ValueError("This report has no generated answers to review.")
    if review["dataset_sha256"] != report["dataset_sha256"]:
        raise ValueError("Review dataset hash does not match the run.")
    if review.get("run_sha256") != run_hash(report):
        raise ValueError("Review belongs to a different answer run.")
    cases = {case["id"]: case for case in report["cases"]}
    judgments = {item["id"]: item for item in review["judgments"]}
    if len(judgments) != len(review["judgments"]) or set(judgments) != set(cases):
        raise ValueError("Review must contain exactly one judgment per run case.")
    for case_id, item in judgments.items():
        case = cases[case_id]
        required = ("correct", "grounded", "complete", "citation_correct") if case["evidence"] else ("correct", "grounded", "abstained")
        for name in required:
            if item[name] not in (0, 1):
                raise ValueError(f"{case_id}: {name} must be 0 or 1")
    result = {"reviewer": review["reviewer"], "n": len(cases), "metrics": {}}
    for field in FIELDS:
        values = [float(judgments[case["id"]][field]) for case in cases.values()
                  if judgments[case["id"]][field] in (0, 1) and
                  (field != "abstained" or not case["evidence"]) and
                  (field != "citation_correct" or case["evidence"])]
        result["metrics"][field] = interval(values)
    for category in sorted({case["category"] for case in cases.values()}):
        selected = [case for case in cases.values() if case["category"] == category]
        result.setdefault("by_category", {})[category] = {
            "n": len(selected),
            "correct": interval([float(judgments[case["id"]]["correct"]) for case in selected]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("run", type=Path)
    init.add_argument("review", type=Path)
    init.add_argument("--reviewer", required=True)
    score = sub.add_parser("score")
    score.add_argument("run", type=Path)
    score.add_argument("reviews", type=Path, nargs="+")
    args = parser.parse_args()
    report = json.loads(args.run.read_text(encoding="utf-8"))
    if args.command == "init":
        if args.review.exists():
            parser.error(f"Refusing to overwrite {args.review}")
        args.review.write_text(json.dumps(make_template(report, args.reviewer), indent=2), encoding="utf-8")
        print(f"Review template: {args.review}")
        return
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in args.reviews]
    results = [score_review(report, review) for review in reviews]
    output = {"reviews": results}
    if len(reviews) > 1:
        first = {item["id"]: item for item in reviews[0]["judgments"]}
        disagreements = []
        for review in reviews[1:]:
            other = {item["id"]: item for item in review["judgments"]}
            for case_id in first:
                fields = [field for field in FIELDS if first[case_id][field] != other[case_id][field]
                          and first[case_id][field] is not None and other[case_id][field] is not None]
                if fields:
                    disagreements.append({"case_id": case_id, "reviewers": [reviews[0]["reviewer"], review["reviewer"]], "fields": fields})
        output["disagreements"] = disagreements
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
