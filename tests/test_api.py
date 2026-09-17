import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from src import main
from src.api import routes
from src.api.schemas import QueryRequest
from src.ingestion.loader import document_id_for_source
from src.main import APIKeyMiddleware, app
from src.pipeline import RAGAnswer
from src.vectorstore.chroma_store import LegacyIndexError


def _mock_pipeline() -> MagicMock:
    return MagicMock()


async def _middleware_response(path: str, api_key: str | None, supplied_key: str | None = None) -> int:
    messages = []

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    headers = [] if supplied_key is None else [(b"x-api-key", supplied_key.encode())]
    middleware = APIKeyMiddleware(downstream, SimpleNamespace(api_key=api_key))
    await middleware({"type": "http", "path": path, "headers": headers}, receive, send)
    return messages[0]["status"]


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def test_mocked_api_lifespan_does_not_initialize_production_pipeline(monkeypatch):
    pipeline_constructor = MagicMock()
    monkeypatch.setattr(main, "RAGPipeline", pipeline_constructor)
    app.dependency_overrides[routes.get_pipeline] = lambda: MagicMock()

    async def run_lifespan():
        async with main.lifespan(app):
            pass

    asyncio.run(run_lifespan())
    pipeline_constructor.assert_not_called()


def test_health_endpoint():
    assert routes.health().model_dump() == {"status": "ok"}


def test_health_endpoint_is_public_when_api_key_is_configured():
    assert asyncio.run(_middleware_response("/api/v1/health", "test-secret")) == 200


def test_query_requires_api_key_when_configured():
    assert asyncio.run(_middleware_response("/api/v1/query", "test-secret")) == 401
    assert asyncio.run(_middleware_response("/api/v1/query", "test-secret", "test-secret")) == 200


def test_query_endpoint_returns_answer():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.query.return_value = RAGAnswer(
        answer="Here is the answer.", sources=["doc1.pdf"], retrieved_chunks=[]
    )

    response = routes.query(QueryRequest(question="What is the policy?"), mock_pipeline)

    assert response.answer == "Here is the answer."
    assert response.sources == ["doc1.pdf"]


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_endpoint_hides_pipeline_error():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.query.side_effect = RuntimeError("boom")

    with pytest.raises(HTTPException) as exc_info:
        routes.query(QueryRequest(question="test"), mock_pipeline)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An internal error occurred while processing the query."


def test_ingest_persists_uploaded_file_with_stable_identity(tmp_path, monkeypatch):
    mock_pipeline = _mock_pipeline()
    mock_pipeline.ingest_file.return_value = 2
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(document_store_path=tmp_path))
    upload = UploadFile(filename="refund-policy.md", file=BytesIO(b"Document content"))

    response = asyncio.run(routes.ingest(upload, mock_pipeline))

    assert response.model_dump() == {"documents_processed": 1, "chunks_created": 2}
    _, kwargs = mock_pipeline.ingest_file.call_args
    assert kwargs["source"] == "refund-policy.md"
    assert kwargs["managed_storage"] is True
    stored_file = tmp_path / f"{document_id_for_source('refund-policy.md')}.md"
    assert kwargs["storage_path"] == stored_file
    assert stored_file.read_bytes() == b"Document content"


def test_ingest_rejects_empty_documents_without_persisting_them(tmp_path, monkeypatch):
    mock_pipeline = _mock_pipeline()
    mock_pipeline.ingest_file.side_effect = ValueError("Document contains no extractable text.")
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(document_store_path=tmp_path))
    upload = UploadFile(filename="empty.md", file=BytesIO(b""))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.ingest(upload, mock_pipeline))

    assert exc_info.value.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_list_documents_returns_pipeline_summaries():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.list_documents.return_value = [
        {"document_id": "document-1", "source": "refund-policy.md", "filename": "refund-policy.md", "chunks_indexed": 3}
    ]

    response = routes.list_documents(mock_pipeline)

    assert response.documents[0].document_id == "document-1"


def test_list_documents_reports_legacy_index_migration_requirement():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.list_documents.side_effect = LegacyIndexError("Rebuild the index.")

    with pytest.raises(HTTPException) as exc_info:
        routes.list_documents(mock_pipeline)

    assert exc_info.value.status_code == 409


def test_delete_document_returns_404_when_unknown():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.delete_document.return_value = False

    with pytest.raises(HTTPException) as exc_info:
        routes.delete_document("missing", mock_pipeline)

    assert exc_info.value.status_code == 404


def test_reindex_document_returns_chunk_count():
    mock_pipeline = _mock_pipeline()
    mock_pipeline.reindex_document.return_value = 4

    response = routes.reindex_document("document-1", mock_pipeline)

    assert response.model_dump() == {"documents_processed": 1, "chunks_created": 4}
