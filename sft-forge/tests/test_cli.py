"""CLI surface: argument handling, exit codes and command output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sft_forge.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path: Path, **overrides) -> Path:
    cfg = {
        "provider": {"kind": "mock"},
        "run": {"name": "cli-test", "output_dir": str(path / "runs")},
        "seeds": {"source": "inline", "inline": ["time management", "sleep quality"]},
        "generation": {"seeds_per_topic": 1, "concurrency": 2},
        "judge": {"enabled": False},
    }
    cfg.update(overrides)
    config_path = path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_init_scaffolds_files(workdir, capsys):
    assert main(["init", str(workdir / "proj"), "--name", "my-data"]) == EXIT_OK
    target = workdir / "proj"
    assert (target / "config.yaml").exists()
    assert (target / ".env.example").exists()
    assert (target / "seeds.example.jsonl").exists()

    # the scaffolded config must be valid for the tool itself
    assert main(["validate", "-c", str(target / "config.yaml")]) == EXIT_USAGE  # no API key exported

    output = capsys.readouterr().out
    assert "next steps" in output
    assert "created" in output


def test_init_refuses_to_clobber_without_force(workdir, capsys):
    main(["init", str(workdir / "proj")])
    assert main(["init", str(workdir / "proj")]) == EXIT_ERROR
    assert "refusing to overwrite" in capsys.readouterr().out
    assert main(["init", str(workdir / "proj"), "--force"]) == EXIT_OK


def test_validate_reports_bad_config(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["validate", "-c", str(config_path)]) == EXIT_OK
    assert "valid" in capsys.readouterr().out

    assert main(["validate", "-c", str(config_path), "--set", "generation.concurrency=0"]) == EXIT_USAGE
    assert "concurrency" in capsys.readouterr().out


def test_missing_config_is_a_usage_error(workdir, capsys):
    assert main(["validate", "-c", "nope.yaml"]) == EXIT_USAGE
    assert "not found" in capsys.readouterr().out


def test_cost_command_prints_an_estimate(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["cost", "-c", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "total model calls" in out and "estimated spend" in out

    assert main(["cost", "-c", str(config_path), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeds"] == 2


def test_run_produces_a_dataset(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["run", "-c", str(config_path), "-q"]) == EXIT_OK
    run_dir = workdir / "runs" / "cli-test"
    assert (run_dir / "dataset_chatml.jsonl").exists()
    assert (run_dir / "report.md").exists()

    capsys.readouterr()
    assert main(["run", "-c", str(config_path), "-q", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["counts"]["accepted"] >= 0
    assert payload["stats"]["counts"]["generated"] == 2


def test_run_limit_override(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["run", "-c", str(config_path), "-q", "--limit", "1", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["counts"]["generated"] == 1


def test_stats_and_report_after_a_run(workdir, capsys):
    config_path = write_config(workdir)
    main(["run", "-c", str(config_path), "-q"])
    run_dir = str(workdir / "runs" / "cli-test")

    assert main(["stats", run_dir]) == EXIT_OK
    out = capsys.readouterr().out
    assert "generated" in out and "domains" in out

    assert main(["stats", run_dir, "--report"]) == EXIT_OK
    assert "# Dataset report" in capsys.readouterr().out

    assert main(["stats", run_dir, "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["counts"]


def test_export_command_reformats_without_regenerating(workdir, capsys):
    config_path = write_config(workdir)
    main(["run", "-c", str(config_path), "-q"])
    run_dir = workdir / "runs" / "cli-test"

    out_path = workdir / "sharegpt.jsonl"
    assert main(["export", str(run_dir), "--format", "sharegpt", "--out", str(out_path)]) == EXIT_OK
    row = json.loads(out_path.read_text().splitlines()[0])
    turns = [c["from"] for c in row["conversations"]]
    assert turns[-2:] == ["human", "gpt"]

    assert (
        main(["export", str(run_dir), "--format", "pairs", "--out", str(workdir / "p.jsonl"), "--include-rejected"])
        == EXIT_OK
    )
    assert (workdir / "p_rejected.jsonl").exists()


def test_stats_on_an_empty_directory_fails_cleanly(workdir, capsys):
    (workdir / "empty").mkdir()
    assert main(["stats", str(workdir / "empty")]) == EXIT_ERROR
    assert "no dataset found" in capsys.readouterr().out


def test_global_flags_are_accepted_before_and_after_the_subcommand(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["-q", "validate", "-c", str(config_path)]) == EXIT_OK
    assert main(["validate", "-c", str(config_path), "-q"]) == EXIT_OK
    capsys.readouterr()


def test_env_file_is_loaded(workdir, capsys, monkeypatch):
    monkeypatch.delenv("SFT_FORGE_TEST_KEY", raising=False)
    (workdir / ".env.secret").write_text("SFT_FORGE_TEST_KEY=from-env-file\n", encoding="utf-8")
    config_path = write_config(workdir, provider={"kind": "openai_compat", "api_key": "${SFT_FORGE_TEST_KEY}"})

    assert main(["validate", "-c", str(config_path), "--env-file", str(workdir / ".env.secret")]) == EXIT_OK
    assert "valid" in capsys.readouterr().out


def test_version_and_help_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit) as excinfo:
        main([])  # a subcommand is required
    assert excinfo.value.code == EXIT_USAGE


def test_run_without_quiet_flags_prints_progress_and_paths(workdir, capsys):
    """Regression: the global flags are optional, so nothing may assume they are set."""
    config_path = write_config(workdir)
    assert main(["run", "-c", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "→ run:" in out
    assert "generated" in out
    assert "artifacts in" in out
    assert "report.md" in out


def test_preview_prints_examples(workdir, capsys):
    config_path = write_config(workdir)
    assert main(["preview", "-c", str(config_path), "-n", "1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "INSTRUCTION:" in out and "RESPONSE:" in out
    assert "prompts.py" in out

    assert main(["preview", "-c", str(config_path), "-n", "1", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["messages"][-1]["role"] == "assistant"
