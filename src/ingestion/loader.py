"""
Chargement de documents depuis le disque.

Objectif pédagogique : montrer qu'on sait normaliser des formats hétérogènes
(txt, md, pdf) vers une structure de données commune (`Document`) avant de
les envoyer dans le pipeline de chunking/embedding.
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
    """Représentation normalisée d'un document source, avant chunking."""

    content: str
    source: str  # chemin ou identifiant d'origine
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Charge un fichier ou un dossier entier vers une liste de `Document`."""

    def load_file(self, path: Path) -> Document:
        path = Path(path)
        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extension non supportée: {path.suffix}. "
                f"Supportées: {SUPPORTED_EXTENSIONS}"
            )

        if path.suffix == ".pdf":
            text = self._read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        return Document(
            content=text,
            source=str(path),
            metadata={"filename": path.name, "extension": path.suffix},
        )

    def load_directory(self, directory: Path) -> list[Document]:
        directory = Path(directory)
        documents: list[Document] = []

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
                try:
                    documents.append(self.load_file(path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Échec du chargement de %s: %s", path, exc)

        logger.info("Chargé %d documents depuis %s", len(documents), directory)
        return documents

    @staticmethod
    def _read_pdf(path: Path) -> str:
        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text)
