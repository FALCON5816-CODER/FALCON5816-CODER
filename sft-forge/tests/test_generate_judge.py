"""Instruction writing, answer generation, judging policy and rewriting."""

from __future__ import annotations

import json

import pytest

from sft_forge.config import deep_merge
from sft_forge.generate import (
    WorkItem,
    fallback_instructions,
    instruction_worker_factory,
    response_worker_factory,
)
from sft_forge.judge import Judge, judge_worker_factory, parse_judgement, sampled
from sft_forge.providers.base import Completion, Provider
from sft_forge.schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record, Seed


class ScriptedProvider(Provider):
    """Returns canned replies in order; records the requests it received."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, *, model, **kwargs) -> Completion:
        self.requests.append({"messages": messages, "kwargs": kwargs})
        text = self.replies.pop(0) if self.replies else "{}"
        return Completion(text=text, model=model, usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})


# ---------------------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------------------


def test_instruction_worker_parses_json_and_builds_items(client, cfg):
    seed = Seed(topic="bread baking", domain="cooking & food", task_type="howto")
    worker = instruction_worker_factory(client, cfg)
    rows = worker(seed)

    assert len(rows) == cfg["generation"]["seeds_per_topic"]
    assert all(row["seed"]["id"] == seed.id for row in rows)
    assert all(row["item_id"].startswith("wi_") for row in rows)
    assert all(row["instruction"] for row in rows)


def test_instruction_worker_falls_back_on_garbage_output(cfg):
    provider = ScriptedProvider(["not json at all"])
    from sft_forge.client import LLMClient

    seed = Seed(topic="index scans", task_type="explanation", difficulty="easy")
    rows = instruction_worker_factory(LLMClient(provider, cache=None), cfg)(seed)

    assert rows, "a seed must never be silently dropped"
    assert rows[0]["generation_status"] == "error"
    assert "index scans" in rows[0]["instruction"]
    assert len(rows) == cfg["generation"]["seeds_per_topic"]


def test_instruction_worker_deduplicates_within_a_seed(cfg):
    provider = ScriptedProvider([json.dumps({"instructions": [{"instruction": "same"}, {"instruction": "same"}]})])
    from sft_forge.client import LLMClient

    rows = instruction_worker_factory(LLMClient(provider, cache=None), cfg)(Seed(topic="t"))
    texts = [r["instruction"] for r in rows]
    assert len(texts) == len(set(texts))


def test_fallback_instructions_are_deterministic_and_topic_aware():
    seed = Seed(topic="composting", task_type="howto", audience="a gardener")
    first = fallback_instructions(seed, 3)
    assert first == fallback_instructions(seed, 3)
    assert all("composting" in text for text in first)


# ---------------------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------------------


def test_response_worker_creates_record_with_usage_and_system(client, cfg):
    item = WorkItem(instruction="What is sourdough?", seed=Seed(topic="sourdough", domain="cooking & food"))
    row = response_worker_factory(client, cfg)(item)

    record = Record.from_dict(row)
    assert record.instruction == "What is sourdough?"
    assert record.response
    assert record.system
    assert record.usage["total_tokens"] > 0
    assert record.meta["source"] == "sft-forge"


def test_response_candidates_are_stored_for_best_of(cfg):
    provider = ScriptedProvider(["first answer", "second answer", "third answer (a bit longer)"])
    from sft_forge.client import LLMClient

    overrides = deep_merge(cfg, {"generation": {"response_candidates": 3}})
    item = WorkItem(instruction="q", seed=Seed(topic="t"))
    row = response_worker_factory(LLMClient(provider, cache=None), overrides)(item)

    assert row["response"] == "first answer"
    assert row["meta"]["candidates"] == ["first answer", "second answer", "third answer (a bit longer)"]


def test_work_item_roundtrip():
    item = WorkItem(instruction="i", seed=Seed(topic="t", domain="d"), difficulty="hard")
    restored = WorkItem.from_row(item.to_dict())
    assert restored.instruction == item.instruction
    assert restored.id == item.id
    assert restored.seed.topic == "t"


# ---------------------------------------------------------------------------------------
# judgement parsing
# ---------------------------------------------------------------------------------------


def test_parse_judgement_full_payload():
    payload = {
        "scores": {"correctness": 5, "helpfulness": 4, "clarity": 4, "instruction_following": 5, "safety": 5},
        "overall": 4.6,
        "verdict": "accept",
        "rationale": "Solid.",
        "issues": [],
    }
    parsed = parse_judgement(json.dumps(payload))
    assert parsed["overall"] == 4.6
    assert parsed["scores"]["correctness"] == 5


def test_parse_judgement_computes_missing_overall_and_clamps():
    parsed = parse_judgement('{"scores": {"correctness": 9, "clarity": 3}}')
    assert parsed["scores"]["correctness"] == 5.0
    assert parsed["overall"] == pytest.approx(4.0)


def test_parse_judgement_handles_flat_scores_and_string_issues():
    parsed = parse_judgement('{"correctness": 4, "helpfulness": 4, "overall": 4, "issues": "one problem"}')
    assert parsed["scores"]["correctness"] == 4
    assert parsed["issues"] == ["one problem"]


def test_parse_judgement_rejects_unusable_payloads():
    with pytest.raises(ValueError):
        parse_judgement("no scores here")
    with pytest.raises(ValueError):
        parse_judgement("[1, 2, 3]")


def test_sampling_is_deterministic_and_respects_bounds():
    assert sampled("ex_abc", 1.0) is True
    assert sampled("ex_abc", 0.0) is False
    assert sampled("ex_abc", 0.5) == sampled("ex_abc", 0.5)
    rate = sum(sampled(f"ex_{i}", 0.25) for i in range(400)) / 400
    assert 0.15 < rate < 0.35


# ---------------------------------------------------------------------------------------
# judge policy
# ---------------------------------------------------------------------------------------


def judge_cfg(**overrides):
    from sft_forge.config import DEFAULTS, interpolate_env

    base = interpolate_env(deep_merge(DEFAULTS, {"provider": {"kind": "mock"}}))
    return deep_merge(base, {"judge": overrides})


def make_record(text: str = "A thorough answer with plenty of specific detail.") -> Record:
    return Record(instruction="How should I do this?", response=text, difficulty="medium")


def test_accepting_judge_marks_record_accepted():
    provider = ScriptedProvider(['{"scores": {"correctness": 5}, "overall": 4.8, "verdict": "accept"}'])
    from sft_forge.client import LLMClient

    judge = Judge(LLMClient(provider, cache=None), judge_cfg(min_overall=4.0))
    record = judge.process(make_record())
    assert record.status == STATUS_ACCEPTED
    assert record.scores["overall"] == 4.8


def test_low_score_rejects_when_rewrite_disabled():
    provider = ScriptedProvider(['{"scores": {"correctness": 2}, "overall": 2.4}'])
    from sft_forge.client import LLMClient

    judge = Judge(LLMClient(provider, cache=None), judge_cfg(min_overall=4.0, rewrite={"enabled": False}))
    record = judge.process(make_record())
    assert record.status == STATUS_REJECTED
    assert "judge_below_threshold" in record.reject_reasons


def test_near_miss_is_rewritten_then_rejudged():
    provider = ScriptedProvider(
        [
            '{"scores": {"correctness": 4}, "overall": 3.6, "issues": ["vague"]}',  # first judgement
            "A much better answer that now addresses every point in detail, concretely.",  # rewrite
            '{"scores": {"correctness": 5}, "overall": 4.7, "verdict": "accept"}',  # re-judge
        ]
    )
    from sft_forge.client import LLMClient

    judge = Judge(LLMClient(provider, cache=None), judge_cfg(min_overall=4.0, rewrite={"enabled": True}))
    record = judge.process(make_record())
    assert record.status == STATUS_ACCEPTED
    assert record.meta["rewritten"] is True
    assert record.rewrite_of and record.id.endswith("-r1")
    assert record.meta["judge_attempts"] == 1


def test_unparseable_judgement_respects_on_error_policy():
    from sft_forge.client import LLMClient

    reject_judge = Judge(LLMClient(ScriptedProvider(["garbage"]), cache=None), judge_cfg(on_error="reject"))
    assert reject_judge.process(make_record()).status == STATUS_REJECTED

    keep_judge = Judge(LLMClient(ScriptedProvider(["garbage"]), cache=None), judge_cfg(on_error="keep"))
    kept = keep_judge.process(make_record())
    assert kept.status != STATUS_REJECTED
    assert kept.meta["judge_error"] is True


def test_disabled_judge_passes_records_through():
    judge = Judge(None, judge_cfg(enabled=False))  # type: ignore[arg-type]
    record = judge.process(make_record())
    assert record.status != STATUS_REJECTED and record.scores is None


def test_sample_rate_skips_judging_for_some_records():
    from sft_forge.client import LLMClient

    provider = ScriptedProvider(['{"scores": {"correctness": 5}, "overall": 5}'] * 5)
    judge = Judge(LLMClient(provider, cache=None), judge_cfg(sample_rate=0.0))
    record = judge.process(make_record())
    assert record.judge is None
    assert record.meta["judge_skipped"] == "sampled_out"
    assert provider.requests == []


def test_best_of_picks_the_highest_scoring_candidate():
    provider = ScriptedProvider(
        [
            '{"scores": {"correctness": 3}, "overall": 3.0}',
            '{"scores": {"correctness": 5}, "overall": 4.9}',
        ]
    )
    from sft_forge.client import LLMClient

    judge = Judge(LLMClient(provider, cache=None), judge_cfg())
    record = make_record()
    record.meta["candidates"] = ["weak answer", "strong answer"]
    processed = judge.process(record)
    assert processed.response == "strong answer"
    assert len(processed.meta["candidate_scores"]) == 2


def test_judge_worker_returns_serialisable_row():
    worker = judge_worker_factory(None, judge_cfg(enabled=False))  # type: ignore[arg-type]
    row = worker(make_record().to_dict())
    assert isinstance(row, dict) and "instruction" in row
