"""Client-level behaviour: caching, budgets, accounting."""

from __future__ import annotations

import pytest

from sft_forge.cache import ResponseCache
from sft_forge.client import Budget, BudgetExceeded, LLMClient
from sft_forge.providers.base import Completion, Provider, ProviderError

MESSAGES = [{"role": "user", "content": "hello"}]


class CountingProvider(Provider):
    name = "counting"

    def __init__(self, text: str = "answer", fail: bool = False):
        self.calls = 0
        self.text = text
        self.fail = fail

    def chat(self, messages, *, model, **kwargs) -> Completion:
        self.calls += 1
        if self.fail:
            raise ProviderError("nope")
        return Completion(
            text=self.text,
            model=model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            provider=self.name,
        )


def test_cache_roundtrip_and_stats(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    key = cache.make_key({"a": 1})
    assert cache.get(key) is None
    cache.put(key, {"text": "x"}, model="m")
    assert cache.get(key) == {"text": "x"}
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["hit_rate"] == 0.5
    cache.close()


def test_cache_is_content_addressed():
    assert ResponseCache.make_key({"a": 1}) == ResponseCache.make_key({"a": 1})
    assert ResponseCache.make_key({"a": 1}) != ResponseCache.make_key({"a": 2})


def test_disabled_cache_never_stores(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite", enabled=False)
    key = cache.make_key({"a": 1})
    cache.put(key, {"text": "x"})
    assert cache.get(key) is None


def test_identical_requests_hit_the_cache(tmp_path):
    provider = CountingProvider()
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = LLMClient(provider, cache=cache)

    first = client.chat(MESSAGES, model="m", hint="response")
    second = client.chat(MESSAGES, model="m", hint="response")

    assert provider.calls == 1
    assert second.cached and not first.cached
    assert client.stats()["cache"]["hits"] == 1
    cache.close()


def test_different_hints_are_different_cache_entries(tmp_path):
    provider = CountingProvider()
    client = LLMClient(provider, cache=ResponseCache(tmp_path / "cache.sqlite"))
    client.chat(MESSAGES, model="m", hint="judge")
    client.chat(MESSAGES, model="m", hint="response")
    assert provider.calls == 2


def test_budget_caps_requests():
    client = LLMClient(CountingProvider(), budget=Budget(max_requests=1))
    client.chat(MESSAGES, model="m")
    with pytest.raises(BudgetExceeded, match="request budget"):
        client.chat(MESSAGES, model="m")


def test_budget_caps_tokens_and_cost():
    # the cap is checked before each request, so the call that crosses it still completes
    token_client = LLMClient(CountingProvider(), budget=Budget(max_tokens=100))
    token_client.chat(MESSAGES, model="m")  # charges 150 tokens
    with pytest.raises(BudgetExceeded, match="token budget"):
        token_client.chat(MESSAGES, model="m")

    cost_client = LLMClient(CountingProvider(), budget=Budget(max_usd=0.00001))
    cost_client.chat(MESSAGES, model="gpt-4o-mini")  # 100 in / 50 out, priced
    with pytest.raises(BudgetExceeded, match="cost budget"):
        cost_client.chat(MESSAGES, model="gpt-4o-mini")


def test_accounting_tracks_tokens_and_cost_by_task():
    client = LLMClient(CountingProvider())
    client.chat(MESSAGES, model="gpt-4o-mini", hint="judge")
    stats = client.stats()
    assert stats["tokens"] == 150
    assert stats["requests"] == 1
    assert stats["counters"]["calls.judge"] == 1
    assert stats["estimated_usd"] > 0


def test_cost_estimation_uses_price_table():
    client = LLMClient(CountingProvider(), prices={"custom": {"in": 1.0, "out": 2.0}})
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert client.estimate_cost("custom", usage) == pytest.approx(3.0)
    assert client.estimate_cost("unknown-model", usage) == 0.0
    assert "custom" in client.price_table(["custom", ""])


def test_provider_errors_are_recorded_not_swallowed():
    client = LLMClient(CountingProvider(fail=True))
    with pytest.raises(ProviderError):
        client.chat(MESSAGES, model="m")
    stats = client.stats()
    assert stats["errors"] == 1
    assert stats["last_errors"]
