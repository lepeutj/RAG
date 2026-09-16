"""
Providers d'embeddings.

On définit une interface abstraite `EmbeddingProvider` pour pouvoir changer
de backend (modèle local vs API externe) sans toucher au reste du pipeline
(Dependency Inversion Principle). En prod sur un petit VPS, le modèle local
(sentence-transformers) évite tout coût d'API et toute latence réseau pour
l'étape d'embedding, qui est appelée très fréquemment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Interface commune: du texte en entrée, des vecteurs en sortie."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Retourne une matrice (n_texts, dimension) de vecteurs float32."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embeddings calculés localement via sentence-transformers (CPU-friendly)."""

    def __init__(self, model_name: str):
        # Import différé: sentence-transformers/torch sont lourds à charger,
        # on ne paie ce coût que si ce provider est réellement utilisé.
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
    """Embeddings via l'API OpenAI, pour comparaison ou plus haute qualité."""

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
    """Factory: instancie le bon provider à partir de la config."""
    if provider == "local":
        return LocalEmbeddingProvider(model_name)
    if provider == "openai":
        if not api_key:
            raise ValueError("openai_api_key requis pour le provider 'openai'")
        return OpenAIEmbeddingProvider(api_key=api_key)
    raise ValueError(f"Provider d'embedding inconnu: {provider}")
