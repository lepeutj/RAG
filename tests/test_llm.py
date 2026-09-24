import sys
from types import SimpleNamespace

import pytest

from src.generation.llm import LlamaCppProvider, OpenAIProvider, build_llm_provider


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Local answer."))])


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_llama_cpp_normalizes_server_url_and_generates(monkeypatch):
    created_clients = []

    def create_client(**kwargs):
        client = _FakeOpenAI(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=create_client))
    provider = LlamaCppProvider("http://127.0.0.1:8080", "local-model", 64, 0.1)

    answer = provider.generate("What is the policy?", [])

    assert answer == "Local answer."
    assert created_clients[0].kwargs == {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "not-needed",
        "timeout": 45.0,
        "max_retries": 1,
    }
    assert created_clients[0].completions.calls[0]["model"] == "local-model"


def test_llama_cpp_provider_does_not_require_an_api_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: _FakeOpenAI(**kwargs)))

    provider = build_llm_provider(
        provider="llama_cpp",
        api_key=None,
        model="local-model",
        max_tokens=64,
        temperature=0.1,
        base_url="http://127.0.0.1:8080/v1",
    )

    assert isinstance(provider, LlamaCppProvider)


@pytest.mark.parametrize(
    ("provider_name", "base_url", "model"),
    [
        ("openai", None, "gpt-4o-mini"),
        ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ("deepseek", "https://api.deepseek.com", "deepseek-flash"),
        ("mistral", "https://api.mistral.ai/v1", "mistral-small-latest"),
    ],
)
def test_external_openai_compatible_provider_uses_its_own_key_and_endpoint(
    monkeypatch, provider_name, base_url, model
):
    created_clients = []

    def create_client(**kwargs):
        client = _FakeOpenAI(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=create_client))
    provider = build_llm_provider(
        provider=provider_name,
        api_key="provider-secret",
        model=model,
        max_tokens=64,
        temperature=0.2,
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.generate("What is the policy?", []) == "Local answer."
    assert created_clients[0].kwargs == {
        "api_key": "provider-secret",
        "base_url": base_url,
        "timeout": 45.0,
        "max_retries": 1,
    }
    assert created_clients[0].completions.calls[0]["model"] == model
    if provider_name == "deepseek":
        assert created_clients[0].completions.calls[0]["extra_body"] == {
            "thinking": {"type": "disabled"}
        }
    else:
        assert "extra_body" not in created_clients[0].completions.calls[0]


@pytest.mark.parametrize("provider_name", ["openrouter", "deepseek", "mistral"])
def test_external_provider_requires_its_api_key(provider_name):
    with pytest.raises(ValueError, match=f"Missing API key for LLM provider '{provider_name}'"):
        build_llm_provider(provider_name, None, "model", 64, 0.2)
