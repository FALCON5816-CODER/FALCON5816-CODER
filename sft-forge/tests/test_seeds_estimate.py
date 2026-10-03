"""Seed assembly, bundled taxonomy integrity, and cost estimation."""

from __future__ import annotations

import json

import pytest

from sft_forge.config import DEFAULTS, deep_merge, load_config
from sft_forge.estimate import estimate_run, render_estimate
from sft_forge.schemas import Seed
from sft_forge.seeds import (
    DEFAULT_TAXONOMY,
    build_seeds,
    dedupe_seeds,
    load_taxonomy,
    seeds_from_file,
    seeds_from_inline,
)

# ---------------------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------------------


def test_bundled_taxonomy_is_wellformed_and_diverse():
    seeds = load_taxonomy()
    assert len(seeds) >= 100, "the bundled taxonomy should give a real starting point"
    assert len({s.domain for s in seeds}) >= 20
    assert len({s.task_type for s in seeds}) >= 10
    assert {s.difficulty for s in seeds} == {"easy", "medium", "hard"}
    assert all(s.topic.strip() for s in seeds)

    # every task type named in the defaults should be reachable from the taxonomy
    configured = set(DEFAULTS["tasks"]["types"])
    assert configured <= {s.task_type for s in seeds}

    # topics are unique, otherwise the first run would waste calls on duplicates
    topics = [s.topic.lower() for s in seeds]
    assert len(topics) == len(set(topics))


def test_taxonomy_file_is_valid_jsonl():
    for line in DEFAULT_TAXONOMY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert "topic" in json.loads(line)


def test_seed_loading_from_file_and_inline(tmp_path):
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        json.dumps({"topic": "a topic", "domain": "d", "constraints": "be brief"}) + "\n",
        encoding="utf-8",
    )
    from_file = seeds_from_file(path)
    assert from_file[0].domain == "d"
    assert from_file[0].constraints == ["be brief"]
    assert from_file[0].source == "file"

    inline = seeds_from_inline(["plain string", {"topic": "mapped", "difficulty": "hard"}])
    assert inline[0].topic == "plain string"
    assert inline[1].difficulty == "hard"
    with pytest.raises(ValueError):
        seeds_from_inline([42])


def test_seed_file_without_topic_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"domain": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="need a 'topic'"):
        seeds_from_file(path)


def test_dedupe_seeds_keeps_first_occurrence():
    seeds = [Seed(topic="Same"), Seed(topic="same"), Seed(topic="different")]
    assert [s.topic for s in dedupe_seeds(seeds)] == ["Same", "different"]


# ---------------------------------------------------------------------------------------
# build_seeds
# ---------------------------------------------------------------------------------------


def mock_cfg(**overrides):
    base = load_config(None, ["provider.kind=mock"], require_api_key=False)
    return deep_merge(base, overrides)


def test_build_seeds_domain_filter_and_limit():
    seeds = build_seeds(mock_cfg(seeds={"domains": ["personal finance"]}))
    assert seeds and all(s.domain == "personal finance" for s in seeds)

    limited = build_seeds(mock_cfg(seeds={"limit": 5}))
    assert len(limited) == 5


def test_build_seeds_is_deterministic_for_a_given_seed():
    cfg = mock_cfg(seeds={"limit": 8})
    assert [s.topic for s in build_seeds(cfg)] == [s.topic for s in build_seeds(cfg)]


def test_build_seeds_reconciles_task_types_to_the_config():
    cfg = mock_cfg(seeds={"limit": 6}, tasks={"types": ["coding"], "difficulties": ["hard"]})
    seeds = build_seeds(cfg)
    assert {s.task_type for s in seeds} == {"coding"}
    assert {s.difficulty for s in seeds} == {"hard"}


def test_build_seeds_inline_and_file_sources(tmp_path):
    cfg = mock_cfg(seeds={"source": "inline", "inline": ["one", "two"]})
    assert [s.topic for s in build_seeds(cfg)] == ["one", "two"]

    path = tmp_path / "s.jsonl"
    path.write_text('{"topic": "from disk"}\n', encoding="utf-8")
    cfg = mock_cfg(seeds={"source": "file", "file": str(path)})
    assert build_seeds(cfg)[0].topic == "from disk"


def test_seed_expansion_uses_the_model(client):
    cfg = mock_cfg(seeds={"expansion": {"enabled": True, "per_domain": 2, "domains": ["gardening"]}, "limit": None})
    seeds = build_seeds(cfg, client)
    assert any(s.source == "llm_expansion" for s in seeds)
    assert all(s.topic for s in seeds)


# ---------------------------------------------------------------------------------------
# cost estimation
# ---------------------------------------------------------------------------------------


def test_estimate_scales_with_configuration():
    small = estimate_run(mock_cfg(seeds={"limit": 10}, generation={"seeds_per_topic": 2}, judge={"enabled": False}))
    large = estimate_run(mock_cfg(seeds={"limit": 100}, generation={"seeds_per_topic": 6}, judge={"enabled": True}))

    assert small["seeds"] == 10 and large["seeds"] == 100
    assert small["total_calls"] == 10 + 20
    assert large["total_calls"] > small["total_calls"] * 10
    assert large["total_tokens"] > small["total_tokens"]
    assert large["estimated_usd"] > small["estimated_usd"]


def test_estimate_counts_best_of_and_rewrites():
    base = mock_cfg(seeds={"limit": 10}, generation={"seeds_per_topic": 2}, judge={"enabled": True})
    best_of = deep_merge(base, {"generation": {"response_candidates": 3}})
    assert estimate_run(best_of)["total_calls"] > estimate_run(base)["total_calls"]
    assert estimate_run(deep_merge(base, {"judge": {"sample_rate": 0.0}}))["calls"].get("judge", 0) == 0


def test_estimate_costs_money_for_a_priced_model():
    cfg = mock_cfg(
        seeds={"limit": 50},
        generation={"seeds_per_topic": 4},
        models={"generator": "gpt-4o-mini", "judge": "gpt-4o-mini"},
    )
    estimate = estimate_run(cfg)
    assert estimate["estimated_usd"] > 0
    assert len(estimate["breakdown"]) >= 3
    assert all(row["usd"] >= 0 for row in estimate["breakdown"])


def test_estimate_warns_about_budget_and_rate_limit():
    cfg = mock_cfg(
        seeds={"limit": 100},
        generation={"seeds_per_topic": 6},
        budget={"max_usd": 0.0001},
        provider={"rate_limit_per_minute": 1},
    )
    warnings = " ".join(estimate_run(cfg)["warnings"])
    assert "budget.max_usd" in warnings
    assert "hours" in warnings


def test_render_estimate_is_readable():
    text = render_estimate(estimate_run(mock_cfg(seeds={"limit": 5})))
    assert "cost estimate" in text
    assert "seeds:" in text and "total model calls" in text
