"""Centralized, typed application configuration loaded from the environment."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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
    document_store_path: Path = Path("./storage/documents")
    collection_name: str = "documents"

    # --- Retrieval ---
    top_k: int = 5
    similarity_threshold: float = 0.0  # minimum similarity score to keep a chunk

    # --- Generation (LLM) ---
    llm_provider: Literal["anthropic", "openai", "llama_cpp"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llama_cpp_base_url: str = "http://127.0.0.1:8080/v1"
    llama_cpp_model: str = "local-model"
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.2, ge=0.0, le=2.0)

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str | None = None  # protects the API when configured
    public_demo_query: bool = False
    admin_api_enabled: bool = False
    max_upload_bytes: int = Field(5_000_000, ge=1, le=10_000_000)

    # --- Outbound calls ---
    llm_timeout_seconds: float = Field(45.0, ge=1.0, le=120.0)
    llm_max_retries: int = Field(1, ge=0, le=2)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings to avoid reparsing the environment for each call."""
    return Settings()
