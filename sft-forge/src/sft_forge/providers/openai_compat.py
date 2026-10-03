"""OpenAI-compatible ``/chat/completions`` backend.

Works with OpenAI, Azure OpenAI-style gateways, Groq, Together, Fireworks, OpenRouter,
vLLM, LM Studio, Ollama (``/v1``), llama.cpp server, and anything else that speaks the
same wire format.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Sequence
from typing import Any

import httpx

from ..utils import get_logger
from .base import Completion, Message, Provider, ProviderError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        extra_headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff_base_s: float = 1.5,
        backoff_cap_s: float = 60.0,
    ):
        if not base_url:
            raise ValueError("provider.base_url is required for the openai_compat provider")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        headers = {"Content-Type": "application/json", "User-Agent": "sft-forge/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(extra_headers or {})
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=15.0),
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

    # -- request building ---------------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        top_p: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        seed: int | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        if extra:
            payload.update(extra)
        return payload

    # -- transport ----------------------------------------------------------------------

    def _sleep_for_retry(self, attempt: int, response: httpx.Response | None) -> float:
        delay = min(self.backoff_cap_s, self.backoff_base_s * (2**attempt))
        delay *= 0.7 + 0.6 * random.random()  # jitter, avoids synchronised retries
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                with contextlib.suppress(ValueError):
                    delay = max(delay, min(self.backoff_cap_s, float(header)))
        return delay

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 1.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        hint: str | None = None,  # noqa: ARG002 - part of the shared interface
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            seed=seed,
            extra=extra,
        )
        last_error: ProviderError | None = None

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ProviderError(f"transport error: {exc!r}", retryable=True)
                if attempt >= self.max_retries:
                    break
                delay = self._sleep_for_retry(attempt, None)
                get_logger().warning(
                    "request failed (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            latency = time.perf_counter() - started
            if response.status_code >= 400:
                body = response.text[:500]
                retryable = response.status_code in RETRYABLE_STATUS
                last_error = ProviderError(
                    f"HTTP {response.status_code} from {self.base_url}: {body}",
                    status=response.status_code,
                    retryable=retryable,
                )
                if not retryable or attempt >= self.max_retries:
                    break
                delay = self._sleep_for_retry(attempt, response)
                get_logger().warning(
                    "HTTP %d from %s; retry %d/%d in %.1fs",
                    response.status_code,
                    self.base_url,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            return self._parse(response, latency)

        assert last_error is not None
        raise last_error

    def _parse(self, response: httpx.Response, latency: float) -> Completion:
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"endpoint returned non-JSON response: {response.text[:200]!r}") from exc

        if isinstance(data, dict) and data.get("error"):
            message = data["error"].get("message") if isinstance(data["error"], dict) else data["error"]
            raise ProviderError(f"endpoint returned an error object: {message}")

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"response contained no choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if text is None:
            # Some servers put the text in a non-standard field or return reasoning only.
            text = choice.get("text") or message.get("reasoning_content") or ""
        if isinstance(text, list):  # content-parts style responses
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

        usage = data.get("usage") or {}
        return Completion(
            text=str(text),
            model=data.get("model") or choice.get("model") or "",
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(
                    usage.get("total_tokens", 0)
                    or (int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0))
                ),
            },
            finish_reason=choice.get("finish_reason"),
            latency_s=latency,
            provider=self.name,
            raw=data,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
