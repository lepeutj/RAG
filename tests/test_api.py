from unittest.mock import MagicMock
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import routes
from src.ingestion.loader import document_id_for_source
from src.vectorstore.chroma_store import LegacyIndexError
from src import main
from src.main import app
from src.pipeline import RAGAnswer


def _client_with_mock_pipeline() -> tuple[TestClient, MagicMock]:
    mock_pipeline = MagicMock()
    app.dependency_overrides[routes.get_pipeline] = lambda: mock_pipeline
    client = TestClient(app)
    return client, mock_pipeline


def test_health_endpoint():
    client, _ = _client_with_mock_pipeline()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_is_public_when_api_key_is_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "test-secret")
    client, _ = _client_with_mock_pipeline()

    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_query_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "test-secret")
    client, _ = _client_with_mock_pipeline()

    response = client.post("/api/v1/query", json={"question": "test"})

    assert response.status_code == 401


def test_query_endpoint_returns_answer():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.query.return_value = RAGAnswer(
        answer="Here is the answer.",
        sources=["doc1.pdf"],
        retrieved_chunks=[],
    )

    response = client.post("/api/v1/query", json={"question": "What is the policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Here is the answer."
    assert body["sources"] == ["doc1.pdf"]


def test_query_endpoint_rejects_empty_question():
    client, _ = _client_with_mock_pipeline()
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_query_endpoint_handles_pipeline_error():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.query.side_effect = RuntimeError("boom")

    response = client.post("/api/v1/query", json={"question": "test"})

    assert response.status_code == 500
    assert response.json()["detail"] == "An internal error occurred while processing the query."


def test_ingest_persists_uploaded_file_with_stable_identity(tmp_path, monkeypatch):
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.ingest_file.return_value = 2
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(document_store_path=tmp_path))

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("refund-policy.md", b"Document content", "text/markdown")},
    )

    assert response.status_code == 200
    _, kwargs = mock_pipeline.ingest_file.call_args
    assert kwargs["source"] == "refund-policy.md"
    assert kwargs["managed_storage"] is True
    stored_file = tmp_path / f"{document_id_for_source('refund-policy.md')}.md"
    assert kwargs["storage_path"] == stored_file
    assert stored_file.read_bytes() == b"Document content"


def test_ingest_rejects_empty_documents_without_persisting_them(tmp_path, monkeypatch):
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.ingest_file.side_effect = ValueError("Document contains no extractable text.")
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(document_store_path=tmp_path))

    response = client.post("/api/v1/ingest", files={"file": ("empty.md", b"", "text/markdown")})

    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_list_documents_returns_pipeline_summaries():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.list_documents.return_value = [
        {
            "document_id": "document-1",
            "source": "refund-policy.md",
            "filename": "refund-policy.md",
            "chunks_indexed": 3,
        }
    ]

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["document_id"] == "document-1"


def test_list_documents_reports_legacy_index_migration_requirement():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.list_documents.side_effect = LegacyIndexError("Rebuild the index.")

    response = client.get("/api/v1/documents")

    assert response.status_code == 409


def test_delete_document_returns_404_when_unknown():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.delete_document.return_value = False

    response = client.delete("/api/v1/documents/missing")

    assert response.status_code == 404


def test_reindex_document_returns_chunk_count():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.reindex_document.return_value = 4

    response = client.post("/api/v1/documents/document-1/reindex")

    assert response.status_code == 200
    assert response.json() == {"documents_processed": 1, "chunks_created": 4}
