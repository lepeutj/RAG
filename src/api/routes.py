from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from src.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StatsResponse,
)
from src.pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


def get_pipeline() -> RAGPipeline:
    # Overridden in main.py through app.dependency_overrides to inject
    # one pipeline instance, initialized once at application startup.
    raise NotImplementedError


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/stats", response_model=StatsResponse)
def stats(pipeline: RAGPipeline = Depends(get_pipeline)) -> StatsResponse:
    return StatsResponse(**pipeline.stats())


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    try:
        result = pipeline.query(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query processing failed")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.") from exc

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        chunks=[
            SourceChunk(text=c.text, source=c.source, score=c.score)
            for c in result.retrieved_chunks
        ],
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chunks_created = pipeline.ingest_file(tmp_path, source=original_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document ingestion failed for %s", original_name)
        raise HTTPException(status_code=500, detail="An internal error occurred while ingesting the document.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(documents_processed=1, chunks_created=chunks_created)
