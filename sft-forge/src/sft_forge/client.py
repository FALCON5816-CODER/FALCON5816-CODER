"""Resilience + accounting wrapper around a :class:`~sft_forge.providers.base.Provider`.

Every model call in the pipeline goes through :class:`LLMClient`, which adds:

* response caching (identical requests are free and instant on re-runs),
* client-side rate limiting,
* a hard budget guard (requests / tokens / dollars) so a runaway config cannot burn money,
* token + cost accounting per model and per task.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .cache import ResponseCache
from .providers.base import Completion, Message, Provider, ProviderError
from .utils import Counter, RateLimiter, get_logger

#: USD per 1M tokens, used only for *estimates*. Override them in the config.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "o4-mini": {"in": 1.10, "out": 4.40},
    "claude-3-5-haiku": {"in": 0.80, "out": 4.00},
    "claude-3-7-sonnet": {"in": 3.00, "out": 15.00},
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
    "mixtral-8x7b-32768": {"in": 0.24, "out": 0.24},
    "Qwen/Qwen2.5-72B-Instruct": {"in": 0.59, "out": 0.59},
}


class BudgetExceeded(RuntimeError):
    """Raised when a configured budget cap is reached; the run stops gracefully."""


@dataclass
class Budget:
    max_requests: int | None = None
    max_tokens: int | None = None
    max_usd: float | None = None
    requests: int = 0
    tokens: int = 0
    usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceeded(f"request budget reached ({self.requests}/{self.max_requests})")
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExceeded(f"token budget reached ({self.tokens}/{self.max_tokens})")
            if self.max_usd is not None and self.usd >= self.max_usd:
                raise BudgetExceeded(f"cost budget reached (${self.usd:.4f}/${self.max_usd})")

    def charge(self, *, tokens: int, usd: float) -> None:
        with self._lock:
            self.requests += 1
            self.tokens += max(0, tokens)
            self.usd += max(0.0, usd)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "requests": self.requests,
            "tokens": self.tokens,
            "usd": round(self.usd, 6),
        }
        for key in ("max_requests", "max_tokens", "max_usd"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


class LLMClient:
    """The only thing in the codebase that talks to a model."""

    def __init__(
        self,
        provider: Provider,
        *,
        cache: ResponseCache | None = None,
        rate_limit_per_minute: float | None = None,
        budget: Budget | None = None,
        prices: dict[str, dict[str, float]] | None = None,
        seed: int | None = None,
    ):
        self.provider = provider
        self.cache = cache
        self.limiter = RateLimiter(rate_limit_per_minute)
        self.budget = budget or Budget()
        self.prices = {**DEFAULT_PRICES, **(prices or {})}
        self.seed = seed
        self.counters = Counter()
        self.errors: list[str] = []
        self._lock = threading.Lock()

    # -- pricing ------------------------------------------------------------------------

    def estimate_cost(self, model: str, usage: dict[str, int]) -> float:
        price = self.prices.get(model) or self.prices.get(model.split("/")[-1]) or {}
        return (
            usage.get("prompt_tokens", 0) / 1_000_000 * float(price.get("in", 0.0))
            + usage.get("completion_tokens", 0) / 1_000_000 * float(price.get("out", 0.0))
        )

    def price_table(self, models: Sequence[str]) -> dict[str, dict[str, float]]:
        return {m: self.prices.get(m, {}) for m in models if m}

    # -- main entry point ---------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 1.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        hint: str | None = None,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        """Send one chat request. Raises :class:`BudgetExceeded` or :class:`ProviderError`."""
        request = {
            "provider": self.provider.name,
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "hint": hint,
            "seed": seed if seed is not None else self.seed,
            "extra": extra or {},
        }
        key = ResponseCache.make_key(request)
        task = hint or "chat"

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.counters.incr(f"cache_hits.{task}")
                return Completion.from_dict(hit, cached=True)

        self.counters.incr(f"cache_misses.{task}")
        self.budget.check()
        self.limiter.acquire()

        try:
            completion = self.provider.chat(
                messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=request["seed"],
                hint=hint,
                extra=extra,
            )
        except ProviderError as exc:
            with self._lock:
                self.errors.append(f"{task}: {exc}")
            self.counters.incr(f"errors.{task}")
            raise

        cost = self.estimate_cost(model, completion.usage)
        self.budget.charge(tokens=completion.total_tokens, usd=cost)
        self.counters.incr(f"calls.{task}")
        self.counters.incr(f"tokens.{task}", completion.total_tokens)
        self.counters.add_float(f"usd.{task}", cost)
        self.counters.add_float("latency_s", completion.latency_s)

        if self.cache is not None:
            self.cache.put(key, completion.to_dict(), model=model, hint=hint)
        return completion

    # -- reporting ----------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        counters = self.counters.snapshot()
        return {
            "provider": self.provider.name,
            "requests": self.budget.requests,
            "tokens": self.budget.tokens,
            "estimated_usd": round(self.budget.usd, 6),
            "errors": len(self.errors),
            "last_errors": self.errors[-5:],
            "cache": self.cache.stats() if self.cache is not None else None,
            "counters": counters,
        }

    def close(self) -> None:
        try:
            self.provider.close()
        finally:
            if self.cache is not None:
                self.cache.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def log_progress(message: str) -> None:  # pragma: no cover - thin convenience
    get_logger().info(message)
