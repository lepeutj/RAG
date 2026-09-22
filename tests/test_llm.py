import sys
from types import SimpleNamespace

from src.generation.llm import LlamaCppProvider, build_llm_provider


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
