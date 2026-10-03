"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import json

import pytest

from sft_forge.client import Budget, LLMClient
from sft_forge.config import DEFAULTS, deep_merge, interpolate_env
from sft_forge.providers.mock import MockProvider


@pytest.fixture()
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture()
def client(mock_provider: MockProvider, tmp_path) -> LLMClient:
    """A client wired to the offline mock provider, with caching disabled."""
    return LLMClient(mock_provider, cache=None, budget=Budget())


@pytest.fixture()
def cfg():
    """Default config merged with a mock-provider override (no secrets involved)."""
    base = deep_merge(DEFAULTS, {"provider": {"kind": "mock"}, "run": {"name": "test-run"}})
    return interpolate_env(base)


@pytest.fixture()
def tiny_cfg(cfg):
    """A fast config: 2 seeds, 2 instructions each, judging off."""
    return deep_merge(
        cfg,
        {
            "seeds": {"source": "inline", "inline": ["time management", "sleep quality"]},
            "generation": {"seeds_per_topic": 2, "concurrency": 2},
            "judge": {"enabled": False},
        },
    )


@pytest.fixture()
def write_jsonl_file(tmp_path):
    def _write(rows, name="seeds.jsonl"):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    return _write
