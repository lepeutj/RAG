from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.ingestion.loader import Document, document_id_for_source
from src.pipeline import RAGPipeline
from src.vectorstore.chroma_store import RetrievedChunk


def test_reingestion_replaces_chunks_for_the_same_document():
    source = "refund-policy.md"
    document = Document(
        content="Updated policy",
        source=source,
        metadata={"document_id": document_id_for_source(source)},
    )
    chunk = Chunk(text="Updated policy", source=source, chunk_index=0, metadata=document.metadata)

    pipeline = object.__new__(RAGPipeline)
    pipeline._chunker = MagicMock()
    pipeline._chunker.split_document.return_value = [chunk]
    pipeline._embedder = MagicMock()
    pipeline._embedder.embed.return_value = np.array([[0.1, 0.2]], dtype="float32")
    pipeline._vector_store = MagicMock()

    chunks_created = pipeline._ingest_documents([document])

    assert chunks_created == 1
    pipeline._vector_store.replace_document_chunks.assert_called_once_with(
        document_id_for_source(source),
        [chunk],
        pipeline._embedder.embed.return_value,
    )


def test_public_retrieve_does_not_initialize_llm():
    pipeline = object.__new__(RAGPipeline)
    pipeline._retriever = MagicMock()
    expected = [RetrievedChunk(text="Evidence", source="doc.md", score=0.9, metadata={})]
    pipeline._retriever.retrieve.return_value = expected
    pipeline._llm = None

    assert pipeline.retrieve("question", top_k=3) == expected
    pipeline._retriever.retrieve.assert_called_once_with("question", top_k=3)
    assert pipeline._llm is None


def test_empty_document_stops_indexing():
    pipeline = object.__new__(RAGPipeline)
    pipeline._chunker = MagicMock()
    pipeline._chunker.split_document.return_value = []
    document = Document(content="", source="empty.md", metadata={"document_id": "empty"})

    with pytest.raises(ValueError, match="no extractable text"):
        pipeline._ingest_documents([document])


@pytest.mark.parametrize(
    ("provider", "key_field", "model_field"),
    [
        ("openrouter", "openrouter_api_key", "openrouter_model"),
        ("deepseek", "deepseek_api_key", "deepseek_model"),
    ],
)
def test_pipeline_selects_external_provider_credentials(monkeypatch, provider, key_field, model_field):
    settings = Settings(_env_file=None, llm_provider=provider,
                        **{key_field: "provider-secret", model_field: "chosen-model"})
    pipeline = object.__new__(RAGPipeline)
    pipeline._settings = settings
    pipeline._llm = None
    builder = MagicMock()
    monkeypatch.setattr("src.pipeline.build_llm_provider", builder)

    pipeline._get_llm()

    builder.assert_called_once_with(
        provider=provider,
        api_key="provider-secret",
        model="chosen-model",
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        base_url=None,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
