"""Config merging, interpolation, overrides, validation and run directories."""

from __future__ import annotations

import pytest
import yaml

from sft_forge.config import (
    DEFAULTS,
    ConfigError,
    apply_overrides,
    deep_merge,
    interpolate_env,
    load_config,
    parse_override,
    resolve_path,
    resolve_run_dir,
    save_resolved_config,
    validate_config,
)


def test_deep_merge_is_recursive_and_non_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 9}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_interpolate_env_with_defaults(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert interpolate_env("${MY_KEY}") == "secret"
    assert interpolate_env("${MISSING_KEY:-fallback}") == "fallback"
    assert interpolate_env("${MISSING_KEY}") == ""
    assert interpolate_env({"a": ["${MY_KEY}", 2]}) == {"a": ["secret", 2]}


def test_parse_and_apply_overrides():
    assert parse_override("a.b=3") == ("a.b", 3)
    assert parse_override("a.b=true") == ("a.b", True)
    assert parse_override("a.b=[1, 2]") == ("a.b", [1, 2])
    assert parse_override("a.b=hello world") == ("a.b", "hello world")
    with pytest.raises(ConfigError):
        parse_override("no_equals")

    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c.d=3", "e=4"])
    assert cfg == {"a": {"b": 2, "c": {"d": 3}}, "e": 4}


def test_load_config_defaults_require_api_key():
    with pytest.raises(ConfigError, match="api_key"):
        load_config(None)
    cfg = load_config(None, require_api_key=False)
    assert cfg["run"]["name"] == DEFAULTS["run"]["name"]
    assert cfg["_meta"]["config_path"] is None


def test_load_config_reads_yaml_and_applies_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc123")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "from-file"},
                "provider": {"kind": "openai_compat", "api_key": "${TEST_KEY}"},
                "seeds": {"source": "inline", "inline": ["a topic"]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path, ["generation.concurrency=3"])
    assert cfg["run"]["name"] == "from-file"
    assert cfg["provider"]["api_key"] == "abc123"
    assert cfg["generation"]["concurrency"] == 3
    # unspecified keys still come from the defaults
    assert cfg["export"]["formats"] == DEFAULTS["export"]["formats"]


def test_missing_config_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_empty_api_key_gets_actionable_error():
    with pytest.raises(ConfigError) as excinfo:
        load_config(None, ["provider.api_key=", "provider.kind=openai_compat"])
    assert "export OPENAI_API_KEY" in str(excinfo.value) or "provider.api_key" in str(excinfo.value)


def test_mock_provider_needs_no_key():
    cfg = load_config(None, ["provider.kind=mock"])
    assert cfg["provider"]["kind"] == "mock"


@pytest.mark.parametrize(
    "override,message",
    [
        ("provider.kind=bogus", "provider.kind"),
        ("generation.concurrency=0", "concurrency"),
        ("generation.temperature=5", "temperature"),
        ("judge.min_overall=9", "min_overall"),
        ("judge.on_error=maybe", "on_error"),
        ("filters.dedup.threshold=2", "threshold"),
        ("tasks.types=[]", "tasks.types"),
        ("export.formats=[bogus]", "export format"),
        ("export.splits={train: 0.5}", "sum to 1.0"),
        ("budget.max_usd=-1", "max_usd"),
        ("seeds.source=nope", "seeds.source"),
    ],
)
def test_validation_errors(tmp_path, override, message):
    inline_seeds = tmp_path / "seeds.jsonl"
    inline_seeds.write_text('{"topic": "x"}\n', encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        # provider.kind=mock first, then the override under test (last one wins)
        load_config(None, ["provider.kind=mock", override], require_api_key=False)
    assert message in str(excinfo.value)


def test_validate_returns_warnings_for_empty_inline():
    cfg = deep_merge(DEFAULTS, {"seeds": {"source": "inline", "inline": []}})
    warnings = validate_config(cfg, require_api_key=False)
    assert any("inline is empty" in w for w in warnings)


def test_seeds_file_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="seeds.file not found"):
        load_config(None, ["seeds.source=file", "seeds.file=nope.jsonl", "provider.kind=mock"])


def test_resolve_path_prefers_config_dir(tmp_path):
    cfg_dir = tmp_path / "conf"
    cfg_dir.mkdir()
    (cfg_dir / "seeds.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = {"_meta": {"config_dir": str(cfg_dir)}}
    assert resolve_path(cfg, "seeds.jsonl") == cfg_dir / "seeds.jsonl"


def test_run_dir_and_resolved_config(tmp_path):
    cfg = load_config(None, ["provider.kind=mock", f"run.output_dir={tmp_path}"])
    run_dir = resolve_run_dir(cfg)
    assert run_dir == tmp_path / cfg["run"]["name"]

    cfg["provider"]["api_key"] = "super-secret-value"
    save_resolved_config(cfg, run_dir)
    saved = (run_dir / "run_config.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in saved
    assert "redacted" in saved


def test_saving_the_resolved_config_does_not_mutate_the_live_config(tmp_path):
    """Regression: redaction used to leak into the in-memory config and break auth."""
    cfg = load_config(None, ["provider.kind=openai_compat", "provider.api_key=sk-real-key"])
    run_dir = resolve_run_dir(cfg, str(tmp_path))

    save_resolved_config(cfg, run_dir)

    assert cfg["provider"]["api_key"] == "sk-real-key", "the client would authenticate with the redaction placeholder"
    saved = (run_dir / "run_config.json").read_text(encoding="utf-8")
    assert "sk-real-key" not in saved
    assert "***redacted***" in saved
