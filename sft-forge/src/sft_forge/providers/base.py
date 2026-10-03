"""Provider interface shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, str]


class ProviderError(RuntimeError):
    """Raised when a backend cannot satisfy a request after its retries."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class Completion:
    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    latency_s: float = 0.0
    cached: bool = False
    provider: str = ""
    raw: dict[str, Any] | None = None

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0))

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "latency_s": round(self.latency_s, 4),
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, cached: bool = False) -> Completion:
        return cls(
            text=data.get("text", ""),
            model=data.get("model", ""),
            usage=data.get("usage", {}) or {},
            finish_reason=data.get("finish_reason"),
            latency_s=float(data.get("latency_s", 0.0)),
            cached=cached,
            provider=data.get("provider", ""),
        )


class Provider(ABC):
    """Minimal chat interface. Implement this to add a backend."""

    name: str = "provider"

    @abstractmethod
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
        hint: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        """Return a single completion.

        ``hint`` is a pipeline-level label (``"judge"``, ``"response"`` …). Real backends
        ignore it; the mock backend uses it to pick a behaviour. It is part of the cache
        key so different tasks never collide.
        """

    def close(self) -> None:  # pragma: no cover - most providers hold no resources
        return None

    def __enter__(self) -> Provider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
