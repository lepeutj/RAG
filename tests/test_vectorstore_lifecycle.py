from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.ingestion.chunker import Chunk
from src.vectorstore.chroma_store import ChromaVectorStore


@patch("src.vectorstore.chroma_store.chromadb.HttpClient")
def test_http_client_is_used_when_chroma_host_is_set(http_client):
    client = http_client.return_value
    client.get_or_create_collection.return_value = MagicMock()

    ChromaVectorStore(Path("unused"), "documents", host="chroma", port=8000)

    http_client.assert_called_once_with(host="chroma", port=8000)


def test_replace_document_chunks_upserts_before_removing_stale_chunks():
    store = object.__new__(ChromaVectorStore)
    store._collection = MagicMock()
    store._collection.get.side_effect = [
        {"metadatas": []},
        {"ids": ["document-1::chunk-0", "document-1::chunk-1"]},
    ]
    chunk = Chunk(
        text="Latest content",
        source="refund-policy.md",
        chunk_index=0,
        metadata={"document_id": "document-1"},
    )

    store.replace_document_chunks("document-1", [chunk], np.array([[0.1, 0.2]], dtype="float32"))

    store._collection.upsert.assert_called_once()
    store._collection.delete.assert_called_once_with(ids=["document-1::chunk-1"])
