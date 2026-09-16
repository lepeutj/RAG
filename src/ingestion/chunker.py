"""
Découpage de texte en chunks.

Implémentation volontairement "from scratch" (plutôt que d'importer
directement le TextSplitter de LangChain) pour montrer la compréhension du
problème : on veut des chunks de taille bornée, qui se coupent de préférence
sur des frontières sémantiques (paragraphe > phrase > mot), avec un overlap
pour ne pas perdre de contexte à la frontière entre deux chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingestion.loader import Document

# Séparateurs testés du plus "fort" (paragraphe) au plus faible (caractère)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}::chunk-{self.chunk_index}"


class RecursiveChunker:
    """Découpe un texte en essayant les séparateurs du plus au moins naturel."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        raw_chunks = self._recursive_split(text, _SEPARATORS)
        return self._merge_with_overlap(raw_chunks)

    def split_document(self, document: Document) -> list[Chunk]:
        pieces = self.split_text(document.content)
        return [
            Chunk(
                text=piece,
                source=document.source,
                chunk_index=i,
                metadata=document.metadata,
            )
            for i, piece in enumerate(pieces)
        ]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator, remaining_separators = separators[0], separators[1:]
        if separator == "":
            # dernier recours: découpe brute par taille
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(separator)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > self.chunk_size:
                    # la partie elle-même est trop grosse, on redescend d'un niveau
                    chunks.extend(self._recursive_split(part, remaining_separators))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            chunks.append(buffer)

        return chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Ajoute un overlap entre chunks consécutifs pour préserver le contexte."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        merged = [chunks[0]]
        for prev, current in zip(chunks, chunks[1:]):
            overlap_text = prev[-self.chunk_overlap :]
            merged.append(f"{overlap_text}{current}")
        return merged
