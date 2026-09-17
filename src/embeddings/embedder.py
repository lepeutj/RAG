"""Embedding providers with interchangeable local and remote backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Shared interface: text input and vector output."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix with shape ``(n_texts, dimension)``."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """CPU-friendly local embeddings powered by sentence-transformers."""

    def __init__(self, model_name: str):
        # Delay importing heavy dependencies until this provider is used.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity <=> dot product
            show_progress_bar=False,
        ).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings through the OpenAI API for comparison or higher quality."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._dimension = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    @property
    def dimension(self) -> int:
        return self._dimension


def build_embedding_provider(provider: str, model_name: str, api_key: str | None = None) -> EmbeddingProvider:
    """Build the configured embedding provider."""
    if provider == "local":
        return LocalEmbeddingProvider(model_name)
    if provider == "openai":
        if not api_key:
            raise ValueError("openai_api_key is required for the 'openai' provider")
        return OpenAIEmbeddingProvider(api_key=api_key)
    raise ValueError(f"Unknown embedding provider: {provider}")
