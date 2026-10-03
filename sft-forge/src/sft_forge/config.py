"""Configuration: YAML + environment interpolation + CLI overrides + validation.

The config file is the contract for a data-generation run. Everything downstream reads a
single merged dict, so behaviour is reproducible from ``run_config.json`` alone.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .utils import get_logger, read_json, utcnow_iso, write_json

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised for invalid or incomplete configuration, with an actionable message."""


DEFAULTS: dict[str, Any] = {
    "run": {
        "name": "sft-run",
        "output_dir": "runs",
        "seed": 17,
        "resume": True,
    },
    "provider": {
        "kind": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "api_key": "${OPENAI_API_KEY}",
        "timeout_s": 120,
        "max_retries": 5,
        "rate_limit_per_minute": 300,
    },
    "models": {
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "prices": {},
    },
    "generation": {
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 1200,
        "concurrency": 8,
        "seeds_per_topic": 6,
        "response_candidates": 1,
        "require_json": True,
        "system_guidance": "",
        "extra_body": {},
    },
    "seeds": {
        "source": "taxonomy",
        "file": "",
        "inline": [],
        "domains": [],
        "limit": None,
        "shuffle": True,
        "expansion": {
            "enabled": False,
            "per_domain": 8,
            "domains": [],
            "model": None,
            "temperature": 1.0,
        },
    },
    "tasks": {
        "types": [
            "open_qa",
            "explanation",
            "howto",
            "analysis",
            "brainstorming",
            "planning",
            "summarization",
            "rewriting",
            "classification",
            "extraction",
            "debugging",
            "coding",
        ],
        "difficulties": ["easy", "medium", "hard"],
    },
    "judge": {
        "enabled": True,
        "model": None,
        "sample_rate": 1.0,
        "min_overall": 4.0,
        "temperature": 0.0,
        "max_tokens": 600,
        "on_error": "reject",
        "rewrite": {
            "enabled": True,
            "max_attempts": 1,
            "min_overall": 3.5,
        },
    },
    "filters": {
        "dedup": {
            "enabled": True,
            "near": True,
            "template": True,
            "threshold": 0.85,
            "ngram": 5,
        },
        "length": {
            "min_instruction_chars": 8,
            "max_instruction_chars": 4000,
            "min_response_chars": 40,
            "max_response_chars": 12000,
            "min_response_words": 12,
            "max_response_words": 1200,
        },
        "quality": {
            "min_unique_token_ratio": 0.22,
            "max_ngram_repeat_ratio": 0.4,
            "max_instruction_response_overlap": 0.9,
        },
        "language": {
            "min_english_ratio": 0.02,
            "max_non_ascii_ratio": 0.5,
        },
        "safety": {
            "blocklist": [
                "social security number",
                "credit card number",
                "here is how to make a bomb",
            ],
            "regex_denylist": [],
        },
    },
    "cache": {
        "enabled": True,
    },
    "budget": {
        "max_requests": None,
        "max_tokens": None,
        "max_usd": None,
    },
    "export": {
        "formats": ["chatml", "alpaca"],
        "include_scores": True,
        "include_system": True,
        "splits": None,
        "write_rejected": True,
        "compact_jsonl": True,
        "hub": {
            "repo_id": None,
            "formats": ["chatml"],
            "private": True,
        },
    },
}

KNOWN_EXPORT_FORMATS = ("chatml", "alpaca", "sharegpt", "pairs")


# ---------------------------------------------------------------------------------------
# merging / interpolation
# ---------------------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, others replace)."""
    result: dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def interpolate_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Replace ``${VAR}`` / ``${VAR:-fallback}`` in strings. Missing vars become ``""``."""
    env = os.environ if env is None else env

    def sub(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in env and env[name] != "":
            return env[name]
        return fallback if fallback is not None else ""

    if isinstance(value, str):
        return _ENV_RE.sub(sub, value)
    if isinstance(value, Mapping):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    return value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``--set a.b.c=value`` string into ``(["a","b","c"], value)``."""
    if "=" not in text:
        raise ConfigError(f"--set expects key=value, got {text!r}")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"--set expects a non-empty key, got {text!r}")
    if not raw.strip():
        value: Any = ""
    else:
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        if isinstance(value, str):
            value = interpolate_env(value)
    return key, value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply a list of ``key=value`` overrides (dotted paths) to a config dict."""
    result = dict(cfg)
    for item in overrides:
        if isinstance(item, (tuple, list)) and len(item) == 2:  # already-parsed pair
            key, value = item  # type: ignore[misc]
        else:
            key, value = parse_override(str(item))
        cursor: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | None = None,
    *,
    require_api_key: bool = True,
) -> dict[str, Any]:
    """Load, merge, interpolate and validate a run config.

    ``path=None`` yields the built-in defaults (handy for tests and ``sft-forge demo``).
    """
    user_cfg: dict[str, Any] = {}
    base_dir = Path.cwd()
    if path:
        cfg_path = Path(path).expanduser()
        if not cfg_path.exists():
            raise ConfigError(f"config file not found: {cfg_path}")
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{cfg_path}: invalid YAML ({exc})") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{cfg_path}: top level must be a mapping")
        user_cfg = dict(loaded)
        base_dir = cfg_path.resolve().parent

    cfg = deep_merge(DEFAULTS, user_cfg)
    cfg = interpolate_env(cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
        cfg = interpolate_env(cfg)

    cfg["_meta"] = {
        "config_path": str(Path(path).expanduser()) if path else None,
        "config_dir": str(base_dir),
        "loaded_at": utcnow_iso(),
    }
    validate_config(cfg, require_api_key=require_api_key)
    return cfg


def resolve_path(cfg: Mapping[str, Any], value: str) -> Path:
    """Resolve a possibly-relative path against the config file's directory."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    base = Path(cfg.get("_meta", {}).get("config_dir") or Path.cwd())
    candidate = base / p
    return candidate if candidate.exists() else (Path.cwd() / p)


# ---------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------


def _num(cfg: Mapping[str, Any], path: str) -> Any:
    cursor: Any = cfg
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def validate_config(cfg: Mapping[str, Any], *, require_api_key: bool = True) -> list[str]:
    """Validate a merged config. Returns warnings; raises :class:`ConfigError` on errors."""
    errors: list[str] = []
    warnings: list[str] = []

    name = _num(cfg, "run.name")
    if not isinstance(name, str) or not name.strip():
        errors.append("run.name must be a non-empty string")

    kind = _num(cfg, "provider.kind")
    if kind not in ("openai_compat", "mock"):
        errors.append(f"provider.kind must be 'openai_compat' or 'mock' (got {kind!r})")
    elif kind == "openai_compat":
        base_url = _num(cfg, "provider.base_url") or ""
        if not str(base_url).startswith(("http://", "https://")):
            errors.append(f"provider.base_url must be an http(s) URL (got {base_url!r})")
        if require_api_key and not _num(cfg, "provider.api_key"):
            errors.append(
                "provider.api_key is empty. Export the environment variable your config "
                "references (e.g. `export OPENAI_API_KEY=...`), or run offline with "
                "`--set provider.kind=mock`."
            )

    if not _num(cfg, "models.generator"):
        errors.append("models.generator must be set (the model that writes the answers)")

    concurrency = _num(cfg, "generation.concurrency")
    if not isinstance(concurrency, int) or concurrency < 1:
        errors.append("generation.concurrency must be an integer >= 1")

    for key, low, high in (
        ("generation.temperature", 0.0, 2.0),
        ("generation.top_p", 0.0, 1.0),
        ("judge.temperature", 0.0, 2.0),
        ("judge.sample_rate", 0.0, 1.0),
    ):
        value = _num(cfg, key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not (low <= float(value) <= high):
            errors.append(f"{key} must be a number between {low} and {high} (got {value!r})")

    if int(_num(cfg, "generation.seeds_per_topic") or 0) < 1:
        errors.append("generation.seeds_per_topic must be >= 1")

    types = _num(cfg, "tasks.types") or []
    if not isinstance(types, list) or not types:
        errors.append("tasks.types must be a non-empty list")
    difficulties = _num(cfg, "tasks.difficulties") or []
    if not isinstance(difficulties, list) or not difficulties:
        errors.append("tasks.difficulties must be a non-empty list")

    source = _num(cfg, "seeds.source")
    if source not in ("taxonomy", "file", "inline"):
        errors.append(f"seeds.source must be 'taxonomy', 'file' or 'inline' (got {source!r})")
    elif source == "file":
        file_value = _num(cfg, "seeds.file")
        if not file_value:
            errors.append("seeds.file is required when seeds.source == 'file'")
        elif not resolve_path(cfg, str(file_value)).exists():
            errors.append(f"seeds.file not found: {file_value}")
    elif source == "inline" and not _num(cfg, "seeds.inline"):
        warnings.append("seeds.source is 'inline' but seeds.inline is empty")

    on_error = _num(cfg, "judge.on_error")
    if on_error is not None and str(on_error).lower() not in ("reject", "keep"):
        errors.append(f"judge.on_error must be 'reject' or 'keep' (got {on_error!r})")

    expansion = _num(cfg, "seeds.expansion")
    if expansion is not None and not isinstance(expansion, Mapping):
        errors.append("seeds.expansion must be a mapping like {enabled: true, per_domain: 8}")

    min_overall = _num(cfg, "judge.min_overall")
    if min_overall is not None and not (0 <= float(min_overall) <= 5):
        errors.append("judge.min_overall must be between 0 and 5")
    if not (0 <= float(_num(cfg, "filters.dedup.threshold") or 0) <= 1):
        errors.append("filters.dedup.threshold must be between 0 and 1")

    formats = _num(cfg, "export.formats") or []
    bad = [f for f in formats if f not in KNOWN_EXPORT_FORMATS]
    if bad:
        errors.append(
            f"unknown export format(s) {bad}; choose from {list(KNOWN_EXPORT_FORMATS)}"
        )
    if not formats:
        errors.append("export.formats must not be empty")

    splits = _num(cfg, "export.splits")
    if splits:
        if not isinstance(splits, Mapping):
            errors.append("export.splits must be a mapping like {train: 0.9, test: 0.1}")
        else:
            total = sum(float(v) for v in splits.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"export.splits must sum to 1.0 (got {total:.3f})")

    for key in ("budget.max_requests", "budget.max_tokens"):
        value = _num(cfg, key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            errors.append(f"{key} must be a positive integer or null")
    value = _num(cfg, "budget.max_usd")
    if value is not None and (not isinstance(value, (int, float)) or value <= 0):
        errors.append("budget.max_usd must be a positive number or null")

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    for warning in warnings:
        get_logger().warning("config: %s", warning)
    return warnings


# ---------------------------------------------------------------------------------------
# run directories
# ---------------------------------------------------------------------------------------


def resolve_run_dir(cfg: Mapping[str, Any], run_dir: str | None = None, *, create: bool = True) -> Path:
    """Return ``<output_dir>/<run.name>`` (or an explicit ``run_dir``), creating it."""
    if run_dir:
        path = Path(run_dir).expanduser()
    else:
        output_dir = Path(str(_num(cfg, "run.output_dir") or "runs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = resolve_path(cfg, str(output_dir))
        path = output_dir / str(_num(cfg, "run.name") or "sft-run")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_resolved_config(cfg: Mapping[str, Any], run_dir: Path) -> Path:
    """Persist the resolved config next to the run's artifacts, with secrets redacted.

    The config is deep-copied first: redaction must never touch the caller's dict, or the
    client would end up authenticating with the redacted placeholder.
    """
    redacted = _redact(copy.deepcopy(dict(cfg)))
    redacted["_meta"] = {**redacted.get("_meta", {}), "resolved_at": utcnow_iso()}
    return write_json(run_dir / "run_config.json", redacted)


def load_run_config(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a previously saved resolved config from a run directory."""
    data = read_json(Path(run_dir) / "run_config.json")
    if data is None:
        raise ConfigError(f"no run_config.json found in {run_dir}")
    return data


def _redact(cfg: dict[str, Any], keys: tuple[str, ...] = ("api_key", "authorization", "token")) -> dict[str, Any]:
    """Return ``cfg`` with secret-looking string values replaced (mutates the given copy)."""
    for key, value in list(cfg.items()):
        if isinstance(value, dict):
            cfg[key] = _redact(value, keys)
        elif isinstance(value, str) and any(k in key.lower() for k in keys) and value:
            cfg[key] = "***redacted***"
    return cfg
