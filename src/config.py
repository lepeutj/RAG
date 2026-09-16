"""
Configuration centralisée de l'application.

On utilise pydantic-settings pour charger la config depuis les variables
d'environnement (ou un fichier .env), avec validation de types intégrée.
Ça évite les `os.environ.get(...)` disséminés partout dans le code et ça
donne une seule source de vérité, typée, pour toute l'app.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    app_name: str = "rag-system"
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Embeddings ---
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384

    # --- Vector store ---
    vector_store_path: Path = Path("./storage/chroma")
    collection_name: str = "documents"

    # --- Retrieval ---
    top_k: int = 5
    similarity_threshold: float = 0.0  # score minimal pour garder un chunk

    # --- Génération (LLM) ---
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.2

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str | None = None  # protège l'API en prod, si défini


@lru_cache
def get_settings() -> Settings:
    """Singleton de config, mis en cache pour éviter de reparser l'env à chaque appel."""
    return Settings()
