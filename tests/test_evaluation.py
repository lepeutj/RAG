import json
from pathlib import Path

import pytest

from scripts.eval_metrics import interval, retrieval_case, summarize
from scripts.compare_runs import compare
from scripts.evaluate import validate_cases
from scripts.review_answers import make_template, score_review


def test_retrieval_requires_gold_source_and_all_evidence_for_full_recall():
    gold = [{"source": "a.md", "text": "first fact"},
            {"source": "b.md", "text": "second fact"}]
    chunks = [{"source": "wrong.md", "text": "first fact"},
              {"source": "a.md", "text": "The first fact is here."},
              {"source": "b.md", "text": "unrelated"}]
    result = retrieval_case(gold, chunks, 3)
    assert result["evidence_ranks"] == [2, None]
    assert result["evidence_recall_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_unanswerable_has_no_retrieval_success_label():
    result = retrieval_case([], [{"source": "a.md", "text": "irrelevant"}], 5)
    assert result["evidence_hit_at_k"] is None
    assert result["retrieved_any"] is True


def test_bootstrap_is_reproducible():
    assert interval([0.0, 1.0, 1.0]) == interval([0.0, 1.0, 1.0])


def test_gold_evidence_is_checked_against_corpus(tmp_path):
    (tmp_path / "a.md").write_text("The first fact is here.", encoding="utf-8")
    cases = [{"id": "one", "split": "test", "category": "direct", "question": "Question?",
              "evidence": [{"source": "a.md", "text": "first fact"}]}]
    validate_cases(cases, tmp_path)
    cases[0]["evidence"][0]["text"] = "invented"
    with pytest.raises(ValueError):
        validate_cases(cases, tmp_path)


def test_review_requires_completed_labels():
    report = {"dataset_sha256": "abc", "cases": [{"id": "one", "category": "direct",
              "evidence": [{"source": "a.md", "text": "fact"}], "answer": "Fact."}]}
    review = make_template(report, "reviewer")
    with pytest.raises(ValueError):
        score_review(report, review)
    review["judgments"][0].update(correct=1, grounded=1, complete=1, citation_correct=0)
    scored = score_review(report, review)
    assert scored["metrics"]["correct"]["mean"] == 1
    assert scored["metrics"]["citation_correct"]["mean"] == 0


def test_shipped_cases_have_valid_gold_spans():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "data/evaluation/cases.json").read_text(encoding="utf-8"))
    validate_cases(cases, root / "data/documents")
    assert {case["split"] for case in cases} == {"dev", "test"}


def test_paired_comparison_reports_case_level_regressions():
    def run(hit):
        return {"dataset_sha256": "same", "corpus_sha256": "same", "cases": [
            {"id": "one", "evidence": [{"source": "a", "text": "fact"}],
             "retrieval": {"evidence_recall_at_k": hit, "evidence_hit_at_k": bool(hit),
                           "reciprocal_rank": hit}}
        ]}
    result = compare(run(1), run(0))
    assert result["changes"]["evidence_recall_at_k"]["delta_mean"] == -1
    assert result["changes"]["evidence_recall_at_k"]["worsened"] == ["one"]
