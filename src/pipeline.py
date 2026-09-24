"""RAG pipeline orchestration for ingestion, retrieval, and generation.

This is the facade used by the API and CLI scripts. It depends on component
interfaces rather than their implementation details.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.embeddings.embedder import build_embedding_provider
from src.generation.llm import build_llm_provider
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.loader import DocumentLoader
from src.observability import failure_fields
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self._settings = settings

        self._loader = DocumentLoader()
        self._chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = build_embedding_provider(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model_name,
            api_key=settings.openai_api_key,
        )
        self._vector_store = ChromaVectorStore(
            persist_path=settings.vector_store_path,
            collection_name=settings.collection_name,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        # Instantiate the LLM lazily so ingestion does not require an API key.
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            s = self._settings
            if s.llm_provider == "anthropic":
                api_key, model, base_url = s.anthropic_api_key, s.anthropic_model, None
            elif s.llm_provider == "openai":
                api_key, model, base_url = s.openai_api_key, s.openai_model, None
            elif s.llm_provider == "openrouter":
                api_key, model, base_url = s.openrouter_api_key, s.openrouter_model, None
            elif s.llm_provider == "deepseek":
                api_key, model, base_url = s.deepseek_api_key, s.deepseek_model, None
            elif s.llm_provider == "mistral":
                api_key, model, base_url = s.mistral_api_key, s.mistral_model, None
            else:
                api_key, model, base_url = None, s.llama_cpp_model, s.llama_cpp_base_url
            self._llm = build_llm_provider(
                provider=s.llm_provider,
                api_key=api_key,
                model=model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
                base_url=base_url,
                timeout=s.llm_timeout_seconds,
                max_retries=s.llm_max_retries,
            )
        return self._llm

    def ingest_directory(self, directory: Path) -> int:
        """Ingest every supported document in a directory and return the chunk count."""
        documents = self._loader.load_directory(directory)
        return self._ingest_documents(documents)

    def ingest_file(
        self,
        path: Path,
        source: str | None = None,
        managed_storage: bool = False,
        storage_path: Path | None = None,
    ) -> int:
        """Ingest a file while preserving its logical identifier when provided."""
        document = self._loader.load_file(
            path,
            source=source,
            managed_storage=managed_storage,
            storage_path=storage_path,
        )
        return self._ingest_documents([document])

    def _ingest_documents(self, documents) -> int:
        chunks_created = 0
        for document in documents:
            chunks = self._chunker.split_document(document)
            document_id = document.metadata["document_id"]
            if not chunks:
                raise ValueError("Document contains no extractable text.")

            embeddings = self._embedder.embed([chunk.text for chunk in chunks])
            self._vector_store.replace_document_chunks(document_id, chunks, embeddings)
            chunks_created += len(chunks)

        logger.info("Ingestion complete: %d documents -> %d chunks", len(documents), chunks_created)
        return chunks_created

    def list_documents(self) -> list[dict]:
        return self._vector_store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        metadata = self._vector_store.delete_document(document_id)
        if metadata is None:
            return False

        if metadata.get("managed_storage"):
            Path(metadata["storage_path"]).unlink(missing_ok=True)
        return True

    def reindex_document(self, document_id: str) -> int:
        metadata = self._vector_store.get_document_metadata(document_id)
        if metadata is None:
            raise KeyError(document_id)

        storage_path = Path(metadata["storage_path"])
        if not storage_path.is_file():
            raise FileNotFoundError(storage_path)
        return self.ingest_file(
            storage_path,
            source=metadata["source"],
            managed_storage=bool(metadata.get("managed_storage")),
        )

    def query(self, question: str, top_k: int | None = None) -> RAGAnswer:
        stage = "retrieval"
        started = time.perf_counter()
        try:
            chunks = self.retrieve(question, top_k=top_k)
            retrieval_ms = round((time.perf_counter() - started) * 1000)

            if not chunks:
                logger.info("Query completed without retrieved chunks", extra={
                    "event": "query_completed", "duration_ms": retrieval_ms,
                    "retrieval_ms": retrieval_ms, "generation_ms": 0, "retrieved_count": 0,
                })
                return RAGAnswer(
                    answer="I could not find any relevant document to answer this question.",
                    sources=[],
                    retrieved_chunks=[],
                )

            stage = "generation"
            generation_started = time.perf_counter()
            llm = self._get_llm()
            answer_text = llm.generate(question, chunks)
            if not isinstance(answer_text, str) or not answer_text.strip():
                raise ValueError("LLM returned an empty answer.")
            sources = sorted({c.metadata.get("filename", c.source) for c in chunks})
            answer = RAGAnswer(answer=answer_text, sources=sources, retrieved_chunks=chunks)
            logger.info("Query completed", extra={
                "event": "query_completed", "duration_ms": round((time.perf_counter() - started) * 1000),
                "retrieval_ms": retrieval_ms,
                "generation_ms": round((time.perf_counter() - generation_started) * 1000),
                "retrieved_count": len(chunks),
            })
            return answer
        except Exception as exc:
            logger.error("RAG query failed", extra=failure_fields(exc, event="query_failed", stage=stage))
            raise

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve chunks without initializing or calling an LLM."""
        return self._retriever.retrieve(question, top_k=top_k)

    def stats(self) -> dict:
        return {
            "chunks_indexed": self._vector_store.count(),
            "embedding_provider": self._settings.embedding_provider,
            "llm_provider": self._settings.llm_provider,
        }
