"""End-to-end pipeline behaviour, run entirely against the offline mock provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_forge.config import deep_merge, load_config, resolve_run_dir
from sft_forge.pipeline import build_client, preview, run_pipeline
from sft_forge.schemas import Record


def make_cfg(tmp_path, **overrides):
    base = load_config(None, ["provider.kind=mock"], require_api_key=False)
    return deep_merge(
        base,
        {
            "provider": {"kind": "mock"},
            "run": {"name": "pipeline-test", "output_dir": str(tmp_path)},
            "seeds": {"source": "inline", "inline": ["time management", "sleep quality", "home cooking"]},
            "generation": {"seeds_per_topic": 2, "concurrency": 3},
            "judge": {"enabled": True},
            **overrides,
        },
    )


@pytest.fixture()
def cfg(tmp_path):
    return make_cfg(tmp_path)


def test_run_creates_every_artifact(cfg, tmp_path):
    result = run_pipeline(cfg, show_progress=False)
    run_dir = result.run_dir

    for name in [
        "seeds.jsonl",
        "instructions.jsonl",
        "responses.jsonl",
        "judge.jsonl",
        "filtered.jsonl",
        "final.jsonl",
        "dataset_chatml.jsonl",
        "dataset_alpaca.jsonl",
        "rejected.jsonl",
        "manifest.json",
        "stats.json",
        "filter_stats.json",
        "run_config.json",
        "report.md",
        "run_summary.json",
    ]:
        assert (run_dir / name).exists(), f"{name} was not written"
    for stage in ("seeds", "instructions", "responses", "judge"):
        assert (run_dir / f"{stage}.done.json").exists(), f"{stage} did not checkpoint"

    assert result.stats["counts"]["generated"] == 6
    assert result.stats["counts"]["accepted"] > 0
    assert result.manifest["counts"]["accepted"] == result.stats["counts"]["accepted"]
    assert result.manifest["config_hash"]
    assert result.manifest["dataset_fingerprint"]


def test_accepted_rows_are_training_ready(cfg):
    result = run_pipeline(cfg, show_progress=False)
    rows = [json.loads(line) for line in (result.run_dir / "dataset_chatml.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles[0] == "system" and roles[-1] == "assistant"
        assert row["messages"][-1]["content"].strip()
        assert row["metadata"]["id"].startswith("ex_")
        assert row["metadata"]["scores"]["overall"] >= cfg["judge"]["min_overall"]


def test_run_config_is_saved_without_secrets(cfg):
    result = run_pipeline(cfg, show_progress=False)
    saved = json.loads((result.run_dir / "run_config.json").read_text())
    assert saved["provider"]["kind"] == "mock"
    assert saved["provider"].get("api_key", "") in ("", "***redacted***")


def test_second_run_resumes_instead_of_regenerating(cfg):
    first = run_pipeline(cfg, show_progress=False)
    second = run_pipeline(cfg, show_progress=False)

    assert second.run_dir == first.run_dir
    for stage in ("instructions", "responses", "judge"):
        assert second.stages[stage]["resumed_from_checkpoint"] is True, f"{stage} was recomputed"
    # identical data, because every stage was read back from its checkpoint
    assert second.manifest["dataset_fingerprint"] == first.manifest["dataset_fingerprint"]


def test_no_resume_regenerates_from_scratch(tmp_path):
    cfg = make_cfg(tmp_path)
    first = run_pipeline(cfg, show_progress=False)
    second = run_pipeline(cfg, show_progress=False, resume=False)
    assert second.stages["instructions"]["resumed_from_checkpoint"] is False
    assert second.manifest["dataset_fingerprint"] == first.manifest["dataset_fingerprint"]


def test_budget_cap_stops_cleanly_and_stays_resumable(tmp_path):
    cfg = make_cfg(tmp_path, budget={"max_requests": 5})
    stopped = run_pipeline(cfg, show_progress=False)

    assert stopped.stopped_early is True
    incomplete = [
        name
        for name, stage in stopped.stages.items()
        if isinstance(stage, dict) and stage.get("stopped_early")
    ]
    assert incomplete, "some stage should report that it stopped early"
    # the cursor is kept so the work already paid for is not repeated
    run_dir = stopped.run_dir
    assert any((run_dir / f"{name}.cursor.json").exists() for name in incomplete)
    assert not all((run_dir / f"{name}.done.json").exists() for name in incomplete)

    # resuming without the cap completes the run
    relaxed = deep_merge(cfg, {"budget": {"max_requests": None}})
    finished = run_pipeline(relaxed, show_progress=False)
    assert finished.stopped_early is False
    assert not any(
        stage.get("stopped_early") for stage in finished.stages.values() if isinstance(stage, dict)
    )
    assert finished.stats["counts"]["generated"] == 6


def test_preview_returns_records_without_writing_a_run(cfg, tmp_path):
    records = preview(cfg, count=2)
    assert records and all(isinstance(r, Record) for r in records)
    assert all(r.response and r.instruction for r in records)
    # preview works in a scratch directory: no run artifacts are produced
    assert not list(Path(tmp_path).glob("*.jsonl"))


def test_splits_are_written_and_disjoint(tmp_path):
    cfg = make_cfg(tmp_path, export={"splits": {"train": 0.5, "test": 0.5}})
    result = run_pipeline(cfg, show_progress=False)
    train_rows = (result.run_dir / "train_chatml.jsonl").read_text().splitlines()
    test_rows = (result.run_dir / "test_chatml.jsonl").read_text().splitlines()
    train = {json.loads(line)["metadata"]["id"] for line in train_rows}
    test = {json.loads(line)["metadata"]["id"] for line in test_rows}
    assert train and test
    assert not (train & test)
    assert len(train) + len(test) == result.manifest["counts"]["accepted"]


def test_empty_seed_set_fails_loudly(tmp_path):
    cfg = make_cfg(tmp_path, seeds={"source": "inline", "inline": [], "limit": 0})
    with pytest.raises(ValueError, match="no seeds produced"):
        run_pipeline(cfg, show_progress=False)


def test_judge_disabled_path(tmp_path):
    cfg = make_cfg(tmp_path, judge={"enabled": False})
    result = run_pipeline(cfg, show_progress=False)
    assert result.stages["judge"]["enabled"] is False
    assert (result.run_dir / "judge.jsonl").exists() is False
    assert result.stats["counts"]["judged"] == 0


def test_build_client_uses_run_cache_directory(cfg, tmp_path):
    run_dir = resolve_run_dir(cfg)
    client = build_client(cfg, run_dir)
    try:
        assert client.provider.name == "mock"
        assert client.cache is not None and client.cache.path == run_dir / "cache.sqlite"
    finally:
        client.close()


# ---------------------------------------------------------------------------------------
# fingerprint-based invalidation
# ---------------------------------------------------------------------------------------


def test_stage_fingerprints_change_only_with_relevant_inputs(tmp_path):
    from sft_forge.pipeline import stage_fingerprint

    base = make_cfg(tmp_path)
    assert stage_fingerprint(base, "judge") == stage_fingerprint(make_cfg(tmp_path), "judge")

    # a judging change must not invalidate generation, and vice versa
    stricter_judge = deep_merge(base, {"judge": {"min_overall": 4.9}})
    assert stage_fingerprint(stricter_judge, "judge") != stage_fingerprint(base, "judge")
    assert stage_fingerprint(stricter_judge, "responses") == stage_fingerprint(base, "responses")

    hotter = deep_merge(base, {"generation": {"temperature": 0.1}})
    assert stage_fingerprint(hotter, "responses") != stage_fingerprint(base, "responses")
    assert stage_fingerprint(hotter, "seeds") == stage_fingerprint(base, "seeds")


def test_changing_the_seed_limit_recomputes_the_run(tmp_path):
    cfg = make_cfg(tmp_path)
    first = run_pipeline(cfg, show_progress=False)
    assert first.stats["counts"]["generated"] == 6

    limited = deep_merge(cfg, {"seeds": {"limit": 1}})
    second = run_pipeline(limited, show_progress=False)

    assert second.stages["seeds"]["resumed_from_checkpoint"] is False
    assert second.stats["counts"]["generated"] == 2  # 1 seed x 2 instructions
    assert second.run_dir == first.run_dir


def test_changing_a_judge_threshold_reuses_generation(tmp_path):
    cfg = make_cfg(tmp_path)
    run_pipeline(cfg, show_progress=False)

    stricter = deep_merge(cfg, {"judge": {"min_overall": 4.9}})
    second = run_pipeline(stricter, show_progress=False)

    assert second.stages["instructions"]["resumed_from_checkpoint"] is True
    assert second.stages["responses"]["resumed_from_checkpoint"] is True
    assert second.stages["judge"]["resumed_from_checkpoint"] is False  # re-judged, as it must be


def test_partial_checkpoint_carries_its_fingerprint(tmp_path):
    cfg = make_cfg(tmp_path, budget={"max_requests": 4})
    stopped = run_pipeline(cfg, show_progress=False)
    cursor_files = list(stopped.run_dir.glob("*.cursor.json"))
    assert cursor_files, "an interrupted stage must leave a cursor"
    payload = json.loads(cursor_files[0].read_text())
    assert payload["fingerprint"]

    # a different configuration must not try to resume that partial work
    changed = deep_merge(cfg, {"generation": {"seeds_per_topic": 5}, "budget": {"max_requests": None}})
    resumed = run_pipeline(changed, show_progress=False)
    assert resumed.stats["counts"]["generated"] == 15  # 3 seeds x 5 instructions
