"""Retrieve relevant chunks for a user question.

The retrieval layer stays independent from storage and embedding concerns so
each component can be tested in isolation.
"""
from __future__ import annotations

from src.embeddings.embedder import EmbeddingProvider
from src.vectorstore.chroma_store import ChromaVectorStore, RetrievedChunk


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed([query])[0]
        results = self._vector_store.query(query_embedding, top_k=top_k or self._top_k)
        return [r for r in results if r.score >= self._similarity_threshold]
