"""Small, content-free JSON logs for Cloud Run."""
from __future__ import annotations

import json
import logging
import traceback
from contextvars import ContextVar

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        current_id = request_id.get()
        if current_id:
            entry["request_id"] = current_id
        for key in (
            "event", "stage", "exception_type", "location", "duration_ms",
            "retrieval_ms", "generation_ms", "retrieved_count",
            "method", "path", "upstream_status",
        ):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        return json.dumps(entry, ensure_ascii=False)


def failure_fields(exc: Exception, *, event: str, stage: str | None = None) -> dict:
    """Describe a failure without serializing its potentially sensitive message."""
    fields = {"event": event, "exception_type": type(exc).__name__}
    if stage:
        fields["stage"] = stage
    upstream_status = getattr(exc, "status_code", None)
    if isinstance(upstream_status, int) and 100 <= upstream_status <= 599:
        fields["upstream_status"] = upstream_status
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        frame = frames[-1]
        fields["location"] = f"{frame.filename}:{frame.lineno}:{frame.name}"
    return fields
