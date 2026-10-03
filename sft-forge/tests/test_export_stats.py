"""Export artifacts, statistics and the run report."""

from __future__ import annotations

import json

from sft_forge.export import export_records
from sft_forge.report import render_report
from sft_forge.schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record
from sft_forge.stats import compute_stats, stats_summary_lines

ANSWER = (
    "A clear and specific answer that explains the mechanism, gives one worked example, "
    "names the trade-offs involved, and states when the advice does not apply."
)


def make_records(n: int = 6, reject_every: int = 3) -> list[Record]:
    records = []
    for i in range(n):
        record = Record(
            instruction=f"How should someone approach problem number {i} in practice?",
            response=ANSWER,
            system="You are helpful.",
            domain=["technology & software", "personal finance"][i % 2],
            task_type="howto" if i % 3 else "analysis",
            difficulty=["easy", "medium", "hard"][i % 3],
            meta={"topic": f"topic {i}", "source": "sft-forge"},
            scores={"correctness": 4.5, "overall": 4.2 + (i % 2) * 0.3},
        )
        record.status = STATUS_REJECTED if reject_every and i % reject_every == 0 else STATUS_ACCEPTED
        if record.status == STATUS_REJECTED:
            record.reject_reasons.append("judge_below_threshold")
        records.append(record)
    return records


def test_export_writes_expected_files_and_manifest(tmp_path):
    records = make_records()
    cfg = {
        "run": {"name": "export-test"},
        "models": {"generator": "m1", "judge": "m2"},
        "judge": {"model": "m2"},
        "export": {"formats": ["chatml", "alpaca"], "include_scores": True},
    }
    manifest = export_records(records, cfg, tmp_path)

    accepted = sum(1 for r in records if r.is_accepted)
    assert manifest["counts"]["accepted"] == accepted
    assert manifest["counts"]["total"] == len(records)
    assert manifest["formats"] == ["chatml", "alpaca"]
    assert manifest["config_hash"] and manifest["dataset_fingerprint"]
    assert manifest["models"] == {"generator": "m1", "judge": "m2"}
    assert "review before training" in manifest["notes"].lower()

    for name in ("final.jsonl", "dataset_chatml.jsonl", "dataset_alpaca.jsonl", "rejected.jsonl", "manifest.json"):
        assert (tmp_path / name).exists(), name

    chatml = [json.loads(line) for line in (tmp_path / "dataset_chatml.jsonl").read_text().splitlines()]
    assert len(chatml) == accepted
    assert "messages" in chatml[0] and "metadata" in chatml[0]

    rejected = [json.loads(line) for line in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert len(rejected) == len(records) - accepted
    assert rejected[0]["reject_reasons"]


def test_export_can_strip_system_and_metadata(tmp_path):
    cfg = {
        "run": {"name": "x"},
        "export": {"formats": ["chatml"], "include_system": False, "include_scores": False},
    }
    export_records(make_records(2, reject_every=0), cfg, tmp_path)
    row = json.loads((tmp_path / "dataset_chatml.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert "metadata" not in row


def test_export_splits_are_disjoint_and_sum_to_total(tmp_path):
    records = make_records(10, reject_every=0)
    cfg = {"run": {"name": "x"}, "export": {"formats": ["chatml"], "splits": {"train": 0.6, "test": 0.4}}}
    export_records(records, cfg, tmp_path)

    train = (tmp_path / "train_chatml.jsonl").read_text().splitlines()
    test = (tmp_path / "test_chatml.jsonl").read_text().splitlines()
    assert len(train) + len(test) == 10
    train_ids = {json.loads(line)["metadata"]["id"] for line in train}
    test_ids = {json.loads(line)["metadata"]["id"] for line in test}
    assert train_ids.isdisjoint(test_ids)
    assert json.loads((tmp_path / "splits.json").read_text()) == {"train": 6, "test": 4}


def test_export_can_skip_rejected_file(tmp_path):
    cfg = {"run": {"name": "x"}, "export": {"formats": ["chatml"], "write_rejected": False}}
    export_records(make_records(), cfg, tmp_path)
    assert not (tmp_path / "rejected.jsonl").exists()


def test_compute_stats_shapes():
    stats = compute_stats(make_records(12))
    assert stats["counts"]["generated"] == 12
    assert stats["counts"]["accepted"] + stats["counts"]["rejected"] == 12
    assert stats["response_words"]["accepted"]["p50"] > 0
    assert stats["scores"]["overall"]["mean"] > 4
    assert stats["coverage"]["domains"] == 2
    assert 0 <= stats["coverage"]["domain_balance_gini"] <= 1
    assert stats["rejections"]["reasons"]["judge_below_threshold"] >= 1
    assert stats_summary_lines(stats)


def test_compute_stats_on_empty_input():
    stats = compute_stats([])
    assert stats["counts"]["generated"] == 0
    assert stats["counts"]["accept_rate"] == 0.0
    assert stats_summary_lines(stats)


def test_stats_percentiles_and_balance():
    records = make_records(6, reject_every=0)
    for i, record in enumerate(records):
        record.response = ANSWER + " extra" * (i * 20)
    stats = compute_stats(records)
    words = stats["response_words"]["accepted"]
    assert words["p5"] <= words["p50"] <= words["p95"]
    assert stats["coverage"]["task_type_distribution"]


def test_report_contains_the_important_sections():
    cfg = {"run": {"name": "report-test"}, "models": {"generator": "gpt-x"}, "judge": {"model": "gpt-y"}}
    report = render_report(
        make_records(9),
        cfg,
        manifest={"config_hash": "abc123", "dataset_fingerprint": "def456"},
        client_stats={"requests": 10, "tokens": 1000, "estimated_usd": 0.5, "errors": 1, "cache": {"hit_rate": 0.25, "hits": 3}},
        timings={"responses": 10.0, "judge": 5.0},
        sample_count=2,
    )
    for fragment in [
        "# Dataset report",
        "## Run configuration",
        "## Stage timings",
        "## Spend and reliability",
        "## Coverage",
        "## Answer quality",
        "## Why examples were dropped",
        "## Random sample of accepted examples",
        "abc123",
        "25%",
    ]:
        assert fragment in report, fragment


def test_report_handles_a_dataset_with_nothing_accepted():
    cfg = {"run": {"name": "empty"}}
    report = render_report(make_records(4, reject_every=1), cfg)
    assert "**0 accepted**" in report or "0 accepted" in report
    assert "Why examples were dropped" in report
    assert "Random sample" not in report
