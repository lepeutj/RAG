"""FastAPI application entry point.

Run locally with ``uvicorn src.main:app --reload``.
For production, see ``deploy/rag-system.service`` or ``docker-compose.yml``.
"""
from __future__ import annotations

import logging
import secrets
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


class SecurityMiddleware:
    """Fail-closed route protection and browser security headers."""

    def __init__(self, app, app_settings):
        self.app = app
        self._settings = app_settings
        # Docs routes are absent in production, so allowing them through here
        # preserves FastAPI's 404 instead of turning it into an auth response.
        self._public_paths = {"/api/v1/health", "/", "/docs", "/openapi.json"}
        self._admin_paths = {"/api/v1/stats", "/api/v1/documents", "/api/v1/ingest"}

    @staticmethod
    def _response(code: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
        return JSONResponse(status_code=code, content={"detail": detail}, headers=headers)

    def _is_admin_path(self, path: str) -> bool:
        return path in self._admin_paths or path.startswith("/api/v1/documents/")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def secure_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                    (b"content-security-policy",
                     b"default-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"),
                ])
                message["headers"] = headers
            await send(message)

        path = scope["path"]
        method = scope.get("method", "GET")
        is_query = path == "/api/v1/query" and method == "POST"
        is_public = path in self._public_paths or path.startswith("/static/")

        if self._is_admin_path(path):
            if not self._settings.admin_api_enabled:
                await self._response(status.HTTP_404_NOT_FOUND, "Not found.")(scope, receive, secure_send)
                return
            if not self._settings.api_key:
                await self._response(status.HTTP_503_SERVICE_UNAVAILABLE, "Administrative API is unavailable.")(
                    scope, receive, secure_send)
                return

        if is_query and self._settings.public_demo_query:
            is_public = True

        if not is_public:
            headers = dict(scope.get("headers", []))
            provided = headers.get(b"x-api-key", b"").decode("latin-1")
            expected = self._settings.api_key or ""
            if not provided or not secrets.compare_digest(provided, expected):
                await self._response(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key.")(
                    scope, receive, secure_send)
                return

        await self.app(scope, receive, secure_send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.admin_api_enabled and not settings.api_key:
        raise RuntimeError("API_KEY is required when ADMIN_API_ENABLED=true.")
    # Tests can inject a lightweight pipeline through FastAPI's dependency
    # overrides. Do not initialize the production embedding model in that case.
    pipeline = None
    if routes.get_pipeline not in app.dependency_overrides:
        logger.info("Initializing the RAG pipeline (loading the embedding model, etc.)...")
        pipeline = RAGPipeline(settings)
        if settings.demo_corpus_path is not None:
            pipeline.ingest_directory(settings.demo_corpus_path)
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
    docs_url=None if settings.environment == "prod" else "/docs",
    redoc_url=None,
    openapi_url=None if settings.environment == "prod" else "/openapi.json",
)
app.add_middleware(SecurityMiddleware, app_settings=settings)


app.include_router(routes.router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
