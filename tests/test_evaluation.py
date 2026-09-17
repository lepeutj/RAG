import json

import pytest

from src.evaluation import EvaluationCase, evaluate_retrieval, load_evaluation_cases
from src.vectorstore.chroma_store import RetrievedChunk


def _chunk(source: str) -> RetrievedChunk:
    return RetrievedChunk(text="text", source=source, score=0.9, metadata={"filename": source})


def test_evaluate_retrieval_calculates_source_recall_and_rank():
    cases = [
        EvaluationCase("first", "question one", ["refund-policy.md"], "answer"),
        EvaluationCase("second", "question two", ["delivery-policy.md"], "answer"),
    ]

    results_by_question = {
        "question one": [_chunk("delivery-policy.md"), _chunk("refund-policy.md")],
        "question two": [_chunk("delivery-policy.md")],
    }
    report = evaluate_retrieval(cases, lambda question, _: results_by_question[question], top_k=3)

    assert report["source_recall_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75
    assert report["results"][0]["reciprocal_rank"] == 0.5


def test_load_evaluation_cases_rejects_missing_required_fields(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps({"cases": [{"id": "missing-fields"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_evaluation_cases(dataset)
