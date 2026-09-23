import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.observability import JsonFormatter, failure_fields, request_id
from src.pipeline import RAGPipeline
from src.vectorstore.chroma_store import RetrievedChunk


def test_failure_log_does_not_include_exception_message_or_question():
    try:
        raise RuntimeError("secret key and private question")
    except RuntimeError as exc:
        fields = failure_fields(exc, event="query_failed", stage="generation")

    record = logging.LogRecord("src.test", logging.ERROR, __file__, 1, "RAG query failed", (), None)
    for key, value in fields.items():
        setattr(record, key, value)
    token = request_id.set("request-123")
    try:
        entry = json.loads(JsonFormatter().format(record))
    finally:
        request_id.reset(token)

    assert entry["request_id"] == "request-123"
    assert entry["stage"] == "generation"
    assert entry["exception_type"] == "RuntimeError"
    assert "secret key" not in json.dumps(entry)
    assert "private question" not in json.dumps(entry)


def test_query_failure_identifies_generation_stage():
    pipeline = object.__new__(RAGPipeline)
    pipeline._retriever = MagicMock()
    pipeline._retriever.retrieve.return_value = [
        RetrievedChunk(text="public passage", source="policy.md", score=0.9, metadata={})
    ]
    pipeline._llm = MagicMock()
    pipeline._llm.generate.side_effect = RuntimeError("private question")

    with patch("src.pipeline.logger.error") as log_error:
        with pytest.raises(RuntimeError):
            pipeline.query("private question")

    fields = log_error.call_args.kwargs["extra"]
    assert fields["event"] == "query_failed"
    assert fields["stage"] == "generation"
    assert fields["exception_type"] == "RuntimeError"
    assert "private question" not in str(fields)
