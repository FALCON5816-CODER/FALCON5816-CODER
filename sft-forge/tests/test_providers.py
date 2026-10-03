"""Provider behaviour, exercised against httpx's mock transport (no sockets)."""

from __future__ import annotations

import json

import httpx
import pytest

from sft_forge.providers import build_provider
from sft_forge.providers.base import Completion, ProviderError
from sft_forge.providers.openai_compat import OpenAICompatProvider

MESSAGES = [{"role": "user", "content": "hi"}]


def make_provider(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url="https://example.test/v1",
        api_key="test-key",
        max_retries=2,
        backoff_base_s=0.0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def test_successful_completion_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == MESSAGES
        return ok_response(
            {
                "model": "test-model",
                "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        )

    completion = make_provider(handler).chat(MESSAGES, model="test-model")
    assert completion.text == "hello"
    assert completion.total_tokens == 7
    assert completion.finish_reason == "stop"
    assert completion.provider == "openai_compat"


def test_retries_then_succeeds_on_429():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return ok_response({"choices": [{"message": {"content": "ok"}}]})

    completion = make_provider(handler).chat(MESSAGES, model="m")
    assert completion.text == "ok"
    assert calls["n"] == 2


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    with pytest.raises(ProviderError) as excinfo:
        make_provider(handler).chat(MESSAGES, model="m")
    assert excinfo.value.status == 400
    assert not excinfo.value.retryable
    assert calls["n"] == 1


def test_retries_are_bounded_and_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    with pytest.raises(ProviderError) as excinfo:
        make_provider(handler).chat(MESSAGES, model="m")
    assert excinfo.value.retryable


def test_transport_errors_are_retried_then_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(ProviderError, match="transport error"):
        make_provider(handler).chat(MESSAGES, model="m")


def test_malformed_and_empty_responses_raise():
    def non_json(request):
        return httpx.Response(200, text="<html>nope</html>")

    with pytest.raises(ProviderError, match="non-JSON"):
        make_provider(non_json).chat(MESSAGES, model="m")

    def no_choices(request):
        return ok_response({"choices": []})

    with pytest.raises(ProviderError, match="no choices"):
        make_provider(no_choices).chat(MESSAGES, model="m")


def test_content_parts_and_reasoning_fallback():
    def parts(request):
        return ok_response({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}]})

    assert make_provider(parts).chat(MESSAGES, model="m").text == "ab"

    def reasoning(request):
        return ok_response({"choices": [{"message": {"reasoning_content": "thought"}}]})

    assert make_provider(reasoning).chat(MESSAGES, model="m").text == "thought"


def test_extra_body_and_response_format_are_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return ok_response({"choices": [{"message": {"content": "ok"}}]})

    make_provider(handler).chat(
        MESSAGES,
        model="m",
        temperature=0.1,
        top_p=0.5,
        max_tokens=99,
        response_format={"type": "json_object"},
        seed=7,
        extra={"top_k": 40},
    )
    assert seen["temperature"] == 0.1
    assert seen["top_p"] == 0.5
    assert seen["max_tokens"] == 99
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["seed"] == 7
    assert seen["top_k"] == 40


def test_build_provider_factory():
    provider = build_provider({"kind": "mock", "mock_latency_s": 0})
    assert provider.name == "mock"
    with pytest.raises(ValueError, match="No API key"):
        build_provider({"kind": "openai_compat", "api_key": ""})
    with pytest.raises(ValueError, match="unknown provider kind"):
        build_provider({"kind": "telepathy"})


def test_completion_serialisation():
    completion = Completion(text="x", model="m", usage={"total_tokens": 3}, latency_s=1.23456)
    restored = Completion.from_dict(completion.to_dict(), cached=True)
    assert restored.text == "x"
    assert restored.cached
    assert restored.latency_s == pytest.approx(1.2346, abs=1e-4)
