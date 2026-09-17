from unittest.mock import MagicMock

import numpy as np

from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import RetrievedChunk


def _make_retriever(similarity_threshold: float = 0.0) -> tuple[Retriever, MagicMock, MagicMock]:
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float32")

    vector_store = MagicMock()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=3,
        similarity_threshold=similarity_threshold,
    )
    return retriever, embedder, vector_store


def test_retrieve_calls_embedder_then_vector_store():
    retriever, embedder, vector_store = _make_retriever()
    vector_store.query.return_value = [
        RetrievedChunk(text="a", source="s1", score=0.9, metadata={}),
    ]

    results = retriever.retrieve("my question")

    embedder.embed.assert_called_once_with(["my question"])
    vector_store.query.assert_called_once()
    assert len(results) == 1
    assert results[0].score == 0.9


def test_retrieve_filters_by_similarity_threshold():
    retriever, _, vector_store = _make_retriever(similarity_threshold=0.5)
    vector_store.query.return_value = [
        RetrievedChunk(text="a", source="s1", score=0.9, metadata={}),
        RetrievedChunk(text="b", source="s2", score=0.2, metadata={}),
    ]

    results = retriever.retrieve("my question")

    assert len(results) == 1
    assert results[0].source == "s1"


def test_retrieve_respects_custom_top_k():
    retriever, _, vector_store = _make_retriever()
    vector_store.query.return_value = []

    retriever.retrieve("my question", top_k=10)

    _, kwargs = vector_store.query.call_args
    assert kwargs["top_k"] == 10
