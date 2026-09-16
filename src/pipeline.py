"""
Pipeline RAG: point d'orchestration entre ingestion, retrieval et génération.

C'est la façade que l'API (et les scripts CLI) appellent. Elle ne connaît
pas les détails d'implémentation de chaque composant, juste leurs interfaces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.embeddings.embedder import build_embedding_provider
from src.generation.llm import build_llm_provider
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.loader import DocumentLoader
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
        )
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        # Le LLM est instancié en lazy pour ne pas exiger de clé API
        # juste pour faire de l'ingestion.
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            s = self._settings
            api_key = s.anthropic_api_key if s.llm_provider == "anthropic" else s.openai_api_key
            model = s.anthropic_model if s.llm_provider == "anthropic" else s.openai_model
            self._llm = build_llm_provider(
                provider=s.llm_provider,
                api_key=api_key,
                model=model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
            )
        return self._llm

    def ingest_directory(self, directory: Path) -> int:
        """Ingère tous les documents d'un dossier. Retourne le nb de chunks créés."""
        documents = self._loader.load_directory(directory)
        return self._ingest_documents(documents)

    def ingest_file(self, path: Path) -> int:
        document = self._loader.load_file(path)
        return self._ingest_documents([document])

    def _ingest_documents(self, documents) -> int:
        all_chunks = []
        for document in documents:
            all_chunks.extend(self._chunker.split_document(document))

        if not all_chunks:
            logger.warning("Aucun chunk produit lors de l'ingestion.")
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = self._embedder.embed(texts)
        self._vector_store.add_chunks(all_chunks, embeddings)

        logger.info("Ingestion terminée: %d documents -> %d chunks", len(documents), len(all_chunks))
        return len(all_chunks)

    def query(self, question: str, top_k: int | None = None) -> RAGAnswer:
        chunks = self._retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return RAGAnswer(
                answer="Je n'ai trouvé aucun document pertinent pour répondre à cette question.",
                sources=[],
                retrieved_chunks=[],
            )

        llm = self._get_llm()
        answer_text = llm.generate(question, chunks)
        sources = sorted({c.metadata.get("filename", c.source) for c in chunks})

        return RAGAnswer(answer=answer_text, sources=sources, retrieved_chunks=chunks)

    def stats(self) -> dict:
        return {
            "chunks_indexed": self._vector_store.count(),
            "embedding_provider": self._settings.embedding_provider,
            "llm_provider": self._settings.llm_provider,
        }
