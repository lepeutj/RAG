"""Offline retrieval evaluation helpers.

The evaluator deliberately measures retrieval only. It does not call an LLM,
so it can be run locally without a generation-provider API key.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.vectorstore.chroma_store import RetrievedChunk


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_sources: list[str]
    expected_answer: str


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    reciprocal_rank: float
    source_recall: float


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a versioned JSON evaluation dataset."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain a non-empty 'cases' list.")

    required_fields = {"id", "question", "expected_sources", "expected_answer"}
    loaded_cases = []
    case_ids = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation case must be a JSON object.")
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(f"Evaluation case is missing required fields: {sorted(missing)}")
        if not isinstance(item["expected_sources"], list) or not item["expected_sources"]:
            raise ValueError(f"Evaluation case '{item['id']}' needs at least one expected source.")
        case_id = str(item["id"])
        if case_id in case_ids:
            raise ValueError(f"Duplicate evaluation case ID: {case_id}")
        case_ids.add(case_id)
        loaded_cases.append(
            EvaluationCase(
                case_id=case_id,
                question=str(item["question"]),
                expected_sources=[str(source) for source in item["expected_sources"]],
                expected_answer=str(item["expected_answer"]),
            )
        )
    return loaded_cases


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[str, int], Sequence[RetrievedChunk]],
    top_k: int,
) -> dict:
    """Calculate source recall@k and mean reciprocal rank for a case set."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results = []
    for case in cases:
        unique_sources = []
        for chunk in retrieve(case.question, top_k):
            source = chunk.metadata.get("filename", chunk.source)
            if source not in unique_sources:
                unique_sources.append(source)

        expected = set(case.expected_sources)
        retrieved = set(unique_sources)
        source_recall = len(expected & retrieved) / len(expected)
        first_match_rank = next((rank for rank, source in enumerate(unique_sources, start=1) if source in expected), None)
        results.append(
            CaseResult(
                case_id=case.case_id,
                expected_sources=case.expected_sources,
                retrieved_sources=unique_sources,
                reciprocal_rank=0.0 if first_match_rank is None else 1 / first_match_rank,
                source_recall=source_recall,
            )
        )

    return {
        "cases_evaluated": len(results),
        "top_k": top_k,
        "source_recall_at_k": sum(result.source_recall for result in results) / len(results),
        "mean_reciprocal_rank": sum(result.reciprocal_rank for result in results) / len(results),
        "results": [asdict(result) for result in results],
    }
