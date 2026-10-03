"""Backends. Build one from config with :func:`build_provider`."""

from __future__ import annotations

from typing import Any

from .base import Completion, Message, Provider, ProviderError

__all__ = ["Completion", "Message", "Provider", "ProviderError", "build_provider"]


def build_provider(cfg: dict[str, Any], *, transport: Any = None) -> Provider:
    """Instantiate a provider from the ``provider`` section of the run config."""
    kind = (cfg.get("kind") or "openai_compat").lower()
    if kind in ("mock", "offline"):
        return _build_mock(cfg)
    if kind in ("openai_compat", "openai", "compatible"):
        return _build_openai_compat(cfg, transport=transport)
    raise ValueError(f"unknown provider kind: {kind!r} (expected 'openai_compat' or 'mock')")


def _build_mock(cfg: dict[str, Any]) -> Provider:
    from .mock import MockProvider

    return MockProvider(
        latency_s=float(cfg.get("mock_latency_s", 0.0)),
        strict_rate=float(cfg.get("mock_strict_rate", 1.0)),
    )


def _build_openai_compat(cfg: dict[str, Any], *, transport: Any = None) -> Provider:
    from .openai_compat import OpenAICompatProvider

    api_key = cfg.get("api_key") or ""
    if not api_key:
        raise ValueError(
            "No API key found. Set provider.api_key (e.g. ${OPENAI_API_KEY}) in your config "
            "and export the variable, or use `--set provider.kind=mock` to run offline."
        )
    return OpenAICompatProvider(
        base_url=cfg.get("base_url") or "https://api.openai.com/v1",
        api_key=api_key,
        timeout_s=float(cfg.get("timeout_s", 120)),
        max_retries=int(cfg.get("max_retries", 5)),
        extra_headers=cfg.get("extra_headers") or None,
        transport=transport,
    )
