from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status

from src.config import get_settings
from src.ingestion.loader import document_id_for_source
from src.observability import failure_fields
from src.vectorstore.chroma_store import LegacyIndexError

from src.api.schemas import (
    DocumentListResponse,
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


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> DocumentListResponse:
    try:
        return DocumentListResponse(documents=pipeline.list_documents())
    except LegacyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> Response:
    try:
        deleted = pipeline.delete_document(document_id)
    except LegacyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/documents/{document_id}/reindex", response_model=IngestResponse)
def reindex_document(document_id: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    try:
        chunks_created = pipeline.reindex_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except FileNotFoundError as exc:
        logger.error("Document source is missing", extra=failure_fields(exc, event="reindex_failed"))
        raise HTTPException(status_code=409, detail="The stored source file is unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LegacyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Document reindexing failed", extra=failure_fields(exc, event="reindex_failed"))
        raise HTTPException(status_code=500, detail="An internal error occurred while reindexing the document.") from exc

    return IngestResponse(documents_processed=1, chunks_created=chunks_created)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    try:
        result = pipeline.query(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        # The pipeline records the failing stage and sanitized exception type.
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
    original_name = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    document_id = document_id_for_source(original_name)
    document_store_path = get_settings().document_store_path
    document_store_path.mkdir(parents=True, exist_ok=True)
    stored_path = document_store_path / f"{document_id}{suffix}"

    max_bytes = get_settings().max_upload_bytes
    with tempfile.NamedTemporaryFile(dir=document_store_path, prefix=".upload-", suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while block := file.file.read(1024 * 1024):
            size += len(block)
            if size > max_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Document exceeds the upload limit.")
            tmp.write(block)

    try:
        chunks_created = pipeline.ingest_file(
            tmp_path,
            source=original_name,
            managed_storage=True,
            storage_path=stored_path,
        )
        tmp_path.replace(stored_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LegacyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Document ingestion failed", extra=failure_fields(exc, event="ingest_failed"))
        raise HTTPException(status_code=500, detail="An internal error occurred while ingesting the document.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(documents_processed=1, chunks_created=chunks_created)
