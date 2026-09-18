"""Dependency-free retrieval metrics and bootstrap intervals."""
from __future__ import annotations

import random
import statistics


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def retrieval_case(evidence: list[dict], chunks: list[dict], k: int) -> dict:
    ranked = chunks[:k]
    ranks = [next((i for i, chunk in enumerate(ranked, 1)
                   if chunk["source"] == item["source"] and normalize(item["text"]) in normalize(chunk["text"])), None)
             for item in evidence]
    relevant = [any(chunk["source"] == item["source"] and
                    normalize(item["text"]) in normalize(chunk["text"]) for item in evidence)
                for chunk in ranked]
    first = next((i for i, value in enumerate(relevant, 1) if value), None)
    return {
        "evidence_ranks": ranks,
        "evidence_recall_at_k": sum(rank is not None for rank in ranks) / len(ranks) if ranks else None,
        "evidence_hit_at_k": bool(first) if evidence else None,
        "reciprocal_rank": 1 / first if first else (0.0 if evidence else None),
        "retrieved_any": bool(ranked),
    }


def interval(values: list[float], seed: int = 42, samples: int = 2000) -> dict | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {"mean": statistics.mean(values), "ci95": [means[int(.025 * samples)], means[int(.975 * samples)]]}


def summarize(rows: list[dict], k: int) -> dict:
    answerable = [r for r in rows if r["evidence"]]
    unknown = [r for r in rows if not r["evidence"]]
    metrics = {"cases": len(rows), "answerable": len(answerable), "unanswerable": len(unknown), "k": k}
    for name in ("evidence_recall_at_k", "evidence_hit_at_k", "reciprocal_rank"):
        metrics[name] = interval([float(r["retrieval"][name]) for r in answerable])
    metrics["unanswerable_retrieved_any"] = interval(
        [float(r["retrieval"]["retrieved_any"]) for r in unknown])
    latencies = sorted(ms for r in rows for ms in r["retrieval_runs_ms"])
    metrics["retrieval_latency_ms"] = {
        "p50": statistics.median(latencies) if latencies else None,
        "p95": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)] if latencies else None,
    }
    metrics["by_category"] = {}
    for category in sorted({r["category"] for r in rows}):
        selected = [r for r in answerable if r["category"] == category]
        if selected:
            metrics["by_category"][category] = {
                "n": len(selected),
                "evidence_recall_at_k": statistics.mean(
                    r["retrieval"]["evidence_recall_at_k"] for r in selected),
            }
    return metrics
