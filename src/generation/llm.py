"""LLM generation providers.

The configured provider can be changed without modifying the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.vectorstore.chroma_store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are an assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the requested information, clearly say that "
    "you do not know instead of inventing an answer. Cite the source filename for "
    "the passages you use."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("filename", chunk.source)
        context_blocks.append(f"[Excerpt {i} - source: {source}]\n{chunk.text}")

    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer precisely and concisely using the context above."
    )


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float,
                 timeout: float, max_retries: int):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        prompt = build_prompt(query, chunks)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float,
                 timeout: float, max_retries: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        prompt = build_prompt(query, chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class LlamaCppProvider(LLMProvider):
    """Generate answers through a local llama.cpp OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, max_tokens: int, temperature: float,
                 timeout: float = 45.0, max_retries: int = 1):
        from openai import OpenAI

        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"

        # The OpenAI client requires a value, while llama.cpp does not require
        # authentication when it is bound to a local interface.
        self._client = OpenAI(
            base_url=normalized_base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        prompt = build_prompt(query, chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def build_llm_provider(
    provider: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: str | None = None,
    timeout: float = 45.0,
    max_retries: int = 1,
) -> LLMProvider:
    if provider == "anthropic":
        if not api_key:
            raise ValueError("Missing API key for LLM provider 'anthropic'")
        return AnthropicProvider(api_key, model, max_tokens, temperature, timeout, max_retries)
    if provider == "openai":
        if not api_key:
            raise ValueError("Missing API key for LLM provider 'openai'")
        return OpenAIProvider(api_key, model, max_tokens, temperature, timeout, max_retries)
    if provider == "llama_cpp":
        if not base_url:
            raise ValueError("Missing base URL for LLM provider 'llama_cpp'")
        return LlamaCppProvider(base_url, model, max_tokens, temperature, timeout, max_retries)
    raise ValueError(f"Unknown LLM provider: {provider}")
