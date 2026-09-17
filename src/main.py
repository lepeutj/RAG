"""FastAPI application entry point.

Run locally with ``uvicorn src.main:app --reload``.
For production, see ``deploy/rag-system.service`` or ``docker-compose.yml``.
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
    logger.info("Initializing the RAG pipeline (loading the embedding model, etc.)...")
    pipeline = RAGPipeline(settings)
    app.state.pipeline = pipeline
    app.dependency_overrides[routes.get_pipeline] = lambda: pipeline
    logger.info("Pipeline ready. %s", pipeline.stats())
    yield
    logger.info("Application stopped.")


app = FastAPI(
    title=settings.app_name,
    description="Retrieval-Augmented Generation (RAG) system — technical demonstration.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """Use a simple API key guard when ``RAG_API_KEY`` is configured."""
    public_paths = {"/api/v1/health", "/docs", "/openapi.json"}
    if settings.api_key and request.url.path not in public_paths:
        provided = request.headers.get("x-api-key")
        if provided != settings.api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid API key."},
            )
    return await call_next(request)


app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running", "docs": "/docs"}
