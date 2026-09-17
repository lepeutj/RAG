"""Document loading from disk.

Supported formats are normalized into a shared ``Document`` structure before
they are sent to the chunking and embedding pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """Normalized representation of a source document before chunking."""

    content: str
    source: str  # chemin ou identifiant d'origine
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Load a file or all supported files in a directory into ``Document`` objects."""

    def load_file(self, path: Path, source: str | None = None) -> Document:
        path = Path(path)
        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension: {path.suffix}. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )

        if path.suffix == ".pdf":
            text = self._read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        source = source or str(path)
        return Document(
            content=text,
            source=source,
            metadata={"filename": Path(source).name, "extension": path.suffix},
        )

    def load_directory(self, directory: Path) -> list[Document]:
        directory = Path(directory)
        documents: list[Document] = []

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
                try:
                    documents.append(self.load_file(path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load %s: %s", path, exc)

        logger.info("Loaded %d documents from %s", len(documents), directory)
        return documents

    @staticmethod
    def _read_pdf(path: Path) -> str:
        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text)
