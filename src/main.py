"""FastAPI application entry point.

Run locally with ``uvicorn src.main:app --reload``.
For production, see ``deploy/rag-system.service`` or ``docker-compose.yml``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api import routes
from src.config import get_settings
from src.pipeline import RAGPipeline

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class APIKeyMiddleware:
    """Minimal ASGI middleware that protects HTTP routes with an optional API key."""

    def __init__(self, app, app_settings):
        self.app = app
        self._settings = app_settings
        self._public_paths = {"/api/v1/health", "/docs", "/openapi.json", "/"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_demo_query = getattr(self._settings, "public_demo_query", False) and scope["path"] == "/api/v1/query" and scope["method"] == "POST"
        is_static = scope["path"].startswith("/static/")
        if self._settings.api_key and scope["path"] not in self._public_paths and not is_demo_query and not is_static:
            headers = dict(scope.get("headers", []))
            provided = headers.get(b"x-api-key", b"").decode("latin-1")
            if provided != self._settings.api_key:
                response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing or invalid API key."},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.environment == "prod" and not settings.api_key:
        raise RuntimeError("API_KEY is required in production.")
    # Tests can inject a lightweight pipeline through FastAPI's dependency
    # overrides. Do not initialize the production embedding model in that case.
    pipeline = None
    if routes.get_pipeline not in app.dependency_overrides:
        logger.info("Initializing the RAG pipeline (loading the embedding model, etc.)...")
        pipeline = RAGPipeline(settings)
        app.state.pipeline = pipeline
        app.dependency_overrides[routes.get_pipeline] = lambda: pipeline
        logger.info("Pipeline ready. %s", pipeline.stats())

    try:
        yield
    finally:
        if pipeline is not None:
            logger.info("Application stopped.")


app = FastAPI(
    title=settings.app_name,
    description="Retrieval-Augmented Generation (RAG) system — technical demonstration.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(APIKeyMiddleware, app_settings=settings)


app.include_router(routes.router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
