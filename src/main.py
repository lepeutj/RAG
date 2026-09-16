"""
Point d'entrée de l'application FastAPI.

Lancement local:      uvicorn src.main:app --reload
Lancement en prod:     voir deploy/rag-system.service ou docker-compose.yml
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api import routes
from src.config import get_settings
from src.pipeline import RAGPipeline

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialisation du pipeline RAG (chargement du modèle d'embedding, etc.)...")
    pipeline = RAGPipeline(settings)
    app.state.pipeline = pipeline
    app.dependency_overrides[routes.get_pipeline] = lambda: pipeline
    logger.info("Pipeline prêt. %s", pipeline.stats())
    yield
    logger.info("Arrêt de l'application.")


app = FastAPI(
    title=settings.app_name,
    description="Système RAG (Retrieval-Augmented Generation) — démonstration technique.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """Protection simple par clé API si RAG_API_KEY est définie (utile en prod sur VPS)."""
    if settings.api_key and request.url.path not in ("/health", "/docs", "/openapi.json"):
        provided = request.headers.get("x-api-key")
        if provided != settings.api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Clé API invalide ou manquante."},
            )
    return await call_next(request)


app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running", "docs": "/docs"}
