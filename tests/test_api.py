from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api import routes
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


def test_ingest_preserves_uploaded_filename():
    client, mock_pipeline = _client_with_mock_pipeline()
    mock_pipeline.ingest_file.return_value = 2

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("refund-policy.md", b"Document content", "text/markdown")},
    )

    assert response.status_code == 200
    _, kwargs = mock_pipeline.ingest_file.call_args
    assert kwargs["source"] == "refund-policy.md"
