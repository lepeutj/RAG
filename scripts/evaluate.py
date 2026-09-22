"""Run a fixed RAG evaluation set and save reviewable retrieval/answer traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.eval_metrics import normalize, retrieval_case, summarize  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.generation.llm import SYSTEM_PROMPT  # noqa: E402
from src.vectorstore.chroma_store import RetrievedChunk  # noqa: E402


def validate_cases(cases: list[dict], corpus: Path) -> None:
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        ids.add(case["id"])
        if not case["question"].strip() or case["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid question or split: {case['id']}")
        for item in case["evidence"]:
            path = corpus / item["source"]
            if not path.is_file() or normalize(item["text"]) not in normalize(path.read_text(encoding="utf-8")):
                raise ValueError(f"Gold evidence missing from corpus: {case['id']} / {item['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/cases.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/documents"))
    parser.add_argument("--output", type=Path, default=Path("storage/evaluation/latest.json"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generate", action="store_true", help="Generate answers for human review.")
    parser.add_argument("--context", choices=("retrieved", "oracle"), default="retrieved",
                        help="Use retrieved chunks or gold source documents for generation.")
    args = parser.parse_args()
    if args.top_k < 1 or args.repeats < 1:
        parser.error("--top-k and --repeats must be positive")
    if args.context == "oracle" and not args.generate:
        parser.error("--context oracle requires --generate")
    raw_cases = args.cases.read_bytes()
    cases = json.loads(raw_cases)
    validate_cases(cases, args.corpus)
    cases = [case for case in cases if args.split == "all" or case["split"] == args.split]
    if not cases:
        parser.error("No cases for selected split")

    settings = get_settings()
    pipeline = RAGPipeline(settings)
    if pipeline.stats()["chunks_indexed"] == 0:
        parser.error("Index is empty. Ingest the evaluation corpus first.")
    indexed_files = {document["filename"] for document in pipeline.list_documents()}
    corpus_files = {path.name for path in args.corpus.iterdir() if path.suffix.lower() in {".txt", ".md", ".pdf"}}
    if indexed_files != corpus_files:
        parser.error(f"Index documents {sorted(indexed_files)} do not match corpus {sorted(corpus_files)}. Rebuild the index.")
    pipeline.retrieve(cases[0]["question"], top_k=args.top_k)  # warmup
    rows = []
    for case in cases:
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            chunks = pipeline.retrieve(case["question"], top_k=args.top_k)
            timings.append(round((time.perf_counter() - start) * 1000, 2))
        retrieved = [{"text": chunk.text,
                      "source": chunk.metadata.get("filename", Path(chunk.source).name),
                      "score": round(chunk.score, 5)} for chunk in chunks]
        result = retrieval_case(case["evidence"], retrieved, args.top_k)
        row = {**case, "retrieved": retrieved, "retrieval": result,
               "retrieval_runs_ms": timings}
        if args.generate:
            generation_chunks = chunks
            if args.context == "oracle":
                generation_chunks = [RetrievedChunk(
                    text=(args.corpus / source).read_text(encoding="utf-8"),
                    source=source, score=1.0, metadata={"filename": source})
                    for source in dict.fromkeys(item["source"] for item in case["evidence"])]
            row["generation_context"] = [
                {"source": chunk.metadata.get("filename", Path(chunk.source).name), "text": chunk.text}
                for chunk in generation_chunks]
            start = time.perf_counter()
            row["answer"] = pipeline._get_llm().generate(case["question"], generation_chunks) if generation_chunks else "I do not know."
            row["generation_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        rows.append(row)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "corpus_sha256": hashlib.sha256(b"".join(
            path.name.encode("utf-8") + b"\0" + path.read_bytes()
            for path in sorted(args.corpus.glob("*")) if path.is_file()
        )).hexdigest(),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "config": {"split": args.split, "top_k": args.top_k, "repeats": args.repeats,
                   "generation_context": args.context if args.generate else None,
                   "embedding_model": settings.embedding_model_name,
                   "embedding_provider": settings.embedding_provider,
                   "llm_provider": settings.llm_provider if args.generate else None,
                   "llm_model": (settings.anthropic_model if settings.llm_provider == "anthropic" else
                                 settings.openai_model if settings.llm_provider == "openai" else
                                 settings.llama_cpp_model) if args.generate else None,
                   "similarity_threshold": settings.similarity_threshold,
                   "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                   "chunks_indexed": pipeline.stats()["chunks_indexed"]},
        "metrics": summarize(rows, args.top_k), "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
