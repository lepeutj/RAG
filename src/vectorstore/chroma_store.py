"""
Store vectoriel basé sur ChromaDB, en mode persistant sur disque.

Chroma est choisi ici car il tourne "embedded" (pas de service séparé à
gérer sur le VPS), persiste sur disque, et gère nativement les métadonnées
et le filtrage — suffisant pour un projet de démonstration sans la
complexité opérationnelle d'un Qdrant/Milvus en cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from src.ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str):
        self._client = chromadb.PersistentClient(path=str(persist_path))
        # cosine similarity: cohérent avec les embeddings normalisés
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[{**chunk.metadata, "source": chunk.source} for chunk in chunks],
        )

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=query_embedding.reshape(1, -1).tolist(),
            n_results=top_k,
        )

        retrieved: list[RetrievedChunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            # Chroma renvoie une distance cosine; on la convertit en score de similarité
            similarity = 1 - distance
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    source=metadata.get("source", "unknown"),
                    score=similarity,
                    metadata=metadata,
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
