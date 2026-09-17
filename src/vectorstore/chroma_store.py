"""Persistent ChromaDB-backed vector store for the demonstration service."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from src.ingestion.chunker import Chunk


class LegacyIndexError(RuntimeError):
    """Raised when a Chroma collection predates document-lifecycle metadata."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str):
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection_name = collection_name
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        # Cosine distance is appropriate for normalized embeddings.
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_lifecycle_metadata(self) -> None:
        results = self._collection.get(include=["metadatas"])
        if any("document_id" not in metadata for metadata in results.get("metadatas") or []):
            raise LegacyIndexError(
                "The existing index predates document lifecycle metadata. Rebuild it with --reset before using lifecycle operations."
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

    def replace_document_chunks(self, document_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Replace every indexed chunk for one document with its latest version."""
        self._ensure_lifecycle_metadata()
        existing = self._collection.get(where={"document_id": document_id}, include=[])
        existing_ids = set(existing.get("ids") or [])
        self.add_chunks(chunks, embeddings)
        new_ids = {chunk.id for chunk in chunks}
        stale_ids = existing_ids - new_ids
        if stale_ids:
            self._collection.delete(ids=sorted(stale_ids))

    def list_documents(self) -> list[dict]:
        """Return one summary per indexed document."""
        self._ensure_lifecycle_metadata()
        results = self._collection.get(include=["metadatas"])
        documents: dict[str, dict] = {}
        for metadata in results.get("metadatas") or []:
            document_id = metadata["document_id"]
            summary = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source": metadata["source"],
                    "filename": metadata["filename"],
                    "chunks_indexed": 0,
                },
            )
            summary["chunks_indexed"] += 1
        return sorted(documents.values(), key=lambda document: document["filename"])

    def get_document_metadata(self, document_id: str) -> dict | None:
        self._ensure_lifecycle_metadata()
        results = self._collection.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        return metadatas[0] if metadatas else None

    def delete_document(self, document_id: str) -> dict | None:
        metadata = self.get_document_metadata(document_id)
        if metadata is None:
            return None
        self._collection.delete(where={"document_id": document_id})
        return metadata

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
            # Convert Chroma's cosine distance to a similarity score.
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
        self._collection = self._get_or_create_collection()
