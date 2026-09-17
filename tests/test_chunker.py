from src.ingestion.chunker import RecursiveChunker
from src.ingestion.loader import Document


def test_split_text_respects_chunk_size():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a test sentence. " * 20  # long text

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    # Overlap can make the final chunk slightly larger than chunk_size.
    assert all(len(c) <= 70 for c in chunks)


def test_split_text_short_text_returns_single_chunk():
    chunker = RecursiveChunker(chunk_size=800, chunk_overlap=120)
    text = "A short text."

    chunks = chunker.split_text(text)

    assert chunks == ["A short text."]


def test_split_document_preserves_metadata_and_source():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    doc = Document(content="a" * 60, source="file.txt", metadata={"filename": "file.txt"})

    chunks = chunker.split_document(doc)

    assert all(c.source == "file.txt" for c in chunks)
    assert all(c.metadata == {"filename": "file.txt"} for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
