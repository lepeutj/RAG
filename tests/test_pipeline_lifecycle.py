from unittest.mock import MagicMock

import numpy as np

from src.ingestion.chunker import Chunk
from src.ingestion.loader import Document, document_id_for_source
from src.pipeline import RAGPipeline


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
