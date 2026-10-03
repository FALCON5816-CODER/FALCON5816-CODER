"""Seed construction: *what* the dataset should cover.

A seed is a coarse specification (topic + domain + task type + difficulty + audience).
The generation stage turns each seed into several concrete user requests, so seed
diversity is what ultimately sets the ceiling on dataset diversity.

Three sources: the bundled taxonomy, a JSONL file, or inline YAML entries. Optionally an
LLM can expand a list of domains into fresh topics, which is how you move past the
bundled list once it starts to feel repetitive.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .client import BudgetExceeded, LLMClient
from .jsonx import as_list, loads_lenient
from .prompts import SEED_EXPANSION_SYSTEM, STYLE_HINTS, seed_expansion_messages
from .schemas import Seed
from .utils import get_logger, read_jsonl

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TAXONOMY = DATA_DIR / "general_taxonomy.jsonl"


# ---------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------


def load_taxonomy(path: str | Path | None = None) -> list[Seed]:
    """Load the bundled (or custom) taxonomy file as seeds."""
    source = Path(path) if path else DEFAULT_TAXONOMY
    if not source.exists():
        raise FileNotFoundError(f"taxonomy file not found: {source}")
    seeds = [Seed.from_dict(row) for row in read_jsonl(source)]
    for seed in seeds:
        seed.source = "taxonomy"
    return seeds


def seeds_from_file(path: str | Path) -> list[Seed]:
    rows = read_jsonl(path)
    seeds: list[Seed] = []
    for i, row in enumerate(rows):
        if "topic" not in row:
            raise ValueError(f"{path}:{i + 1}: seed records need a 'topic' field")
        seed = Seed.from_dict(row)
        seed.source = "file"
        seeds.append(seed)
    return seeds


def seeds_from_inline(rows: Iterable[Any]) -> list[Seed]:
    seeds: list[Seed] = []
    for row in rows or []:
        if isinstance(row, str):
            seed = Seed(topic=row, source="inline")
        elif isinstance(row, Mapping):
            seed = Seed.from_dict({**row, "source": "inline"})
        else:
            raise ValueError(f"inline seeds must be strings or mappings, got {type(row).__name__}")
        seeds.append(seed)
    return seeds


# ---------------------------------------------------------------------------------------
# LLM expansion
# ---------------------------------------------------------------------------------------


def expand_seeds_with_llm(
    client: LLMClient,
    *,
    model: str,
    domains: Sequence[str],
    per_domain: int,
    task_types: Sequence[str],
    temperature: float = 1.0,
) -> list[Seed]:
    """Ask the model for fresh topics in the given domains.

    Failures are non-fatal: a domain that cannot be expanded is skipped with a warning, so
    a single bad response never kills a run.
    """
    expanded: list[Seed] = []
    for domain in domains:
        messages = seed_expansion_messages(domain, per_domain, task_types)
        try:
            completion = client.chat(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=1500,
                response_format={"type": "json_object"},
                hint="seed_expansion",
            )
            payload = loads_lenient(completion.text)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - best-effort enrichment
            get_logger().warning("seed expansion failed for %r: %s", domain, exc)
            continue

        for item in as_list(payload, "topics", "seeds", "items"):
            if not isinstance(item, Mapping):
                continue
            topic = str(item.get("topic") or item.get("title") or "").strip()
            if not topic:
                continue
            expanded.append(
                Seed(
                    topic=topic,
                    domain=str(item.get("domain") or domain),
                    task_type=str(item.get("task_type") or "open_qa"),
                    difficulty=str(item.get("difficulty") or "medium"),
                    audience=str(item.get("audience") or "general audience"),
                    constraints=[str(c) for c in (item.get("constraints") or [])],
                    source="llm_expansion",
                )
            )
    return expanded


# ---------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------


def _reconcile_task_types(seeds: Sequence[Seed], cfg: Mapping[str, Any]) -> None:
    """Force every seed onto a task type the config allows (cycling deterministically)."""
    allowed = [str(t) for t in (cfg.get("tasks", {}).get("types") or [])]
    difficulties = [str(d) for d in (cfg.get("tasks", {}).get("difficulties") or [])]
    if not allowed:
        return
    for i, seed in enumerate(seeds):
        if seed.task_type not in allowed:
            seed.task_type = allowed[i % len(allowed)]
        if difficulties and seed.difficulty not in difficulties:
            seed.difficulty = difficulties[i % len(difficulties)]


def dedupe_seeds(seeds: Iterable[Seed]) -> list[Seed]:
    seen: set[str] = set()
    out: list[Seed] = []
    for seed in seeds:
        key = seed.topic.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(seed)
    return out


def build_seeds(cfg: Mapping[str, Any], client: LLMClient | None = None) -> list[Seed]:
    """Assemble the seed set for a run according to ``seeds.*`` config."""
    seeds_cfg = dict(cfg.get("seeds") or {})
    source = seeds_cfg.get("source", "taxonomy")
    log = get_logger()

    if source == "taxonomy":
        seeds = load_taxonomy(seeds_cfg.get("file") or DEFAULT_TAXONOMY)
    elif source == "file":
        from .config import resolve_path

        seeds = seeds_from_file(resolve_path(cfg, str(seeds_cfg.get("file"))))
    elif source == "inline":
        seeds = seeds_from_inline(seeds_cfg.get("inline") or [])
    else:  # pragma: no cover - validation catches this earlier
        raise ValueError(f"unsupported seeds.source: {source!r}")

    domains = [str(d) for d in (seeds_cfg.get("domains") or [])]
    if domains:
        lowered = {d.lower() for d in domains}
        matching = [s for s in seeds if s.domain.lower() in lowered]
        if not matching and source == "taxonomy":
            log.warning("no bundled seeds matched domains=%s; they will be generated", domains)
        seeds = matching

    expansion = dict(seeds_cfg.get("expansion") or {})
    if expansion.get("enabled"):
        if client is None:
            log.warning("seeds.expansion.enabled is true but no client was supplied; skipping")
        else:
            expand_domains = [str(d) for d in (expansion.get("domains") or domains or ["general knowledge"])]
            added = expand_seeds_with_llm(
                client,
                model=str(expansion.get("model") or (cfg.get("models") or {}).get("generator")),
                domains=expand_domains,
                per_domain=int(expansion.get("per_domain", 8)),
                task_types=[str(t) for t in (cfg.get("tasks", {}).get("types") or ["open_qa"])]
                if isinstance(cfg.get("tasks"), Mapping)
                else ["open_qa"],
                temperature=float(expansion.get("temperature", 1.0)),
            )
            log.info("seed expansion added %d topics", len(added))
            seeds = seeds + added

    seeds = dedupe_seeds(seeds)
    _reconcile_task_types(seeds, cfg)

    if seeds_cfg.get("shuffle"):
        rng = random.Random(int((cfg.get("run") or {}).get("seed", 17)))
        rng.shuffle(seeds)

    limit = seeds_cfg.get("limit")
    if limit:
        seeds = seeds[: int(limit)]

    log.info("prepared %d seeds (source=%s)", len(seeds), source)
    return seeds


def style_hint(index: int) -> str:
    """Deterministic style hint so generated instructions vary in framing."""
    return STYLE_HINTS[index % len(STYLE_HINTS)]


__all__ = [
    "DEFAULT_TAXONOMY",
    "SEED_EXPANSION_SYSTEM",
    "Seed",
    "build_seeds",
    "dedupe_seeds",
    "expand_seeds_with_llm",
    "load_taxonomy",
    "seeds_from_file",
    "seeds_from_inline",
    "style_hint",
]
