from __future__ import annotations

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


def get_pipeline() -> RAGPipeline:
    # Overridée dans main.py via app.dependency_overrides pour injecter
    # une instance unique du pipeline (chargée une seule fois au démarrage).
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
    suffix = Path(file.filename).suffix
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"Format non supporté: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chunks_created = pipeline.ingest_file(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(documents_processed=1, chunks_created=chunks_created)
