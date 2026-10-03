"""Pre-flight cost estimate.

Synthetic data generation is a metered activity: before you point a config at a paid
endpoint, you should know roughly how many calls it implies and what it will cost. The
numbers here are deliberately conservative estimates from typical token shapes, not exact
predictions — the point is to catch a 100x mistake before it happens.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .client import DEFAULT_PRICES
from .config import resolve_path
from .seeds import build_seeds

#: (prompt_tokens, completion_tokens) per call, by task.
TOKEN_ASSUMPTIONS: dict[str, tuple[int, int]] = {
    "seed_expansion": (300, 400),
    "instructions": (430, 320),
    "response": (280, 520),
    "judge": (950, 220),
    "rewrite": (1100, 520),
}

#: share of judged answers that get rewritten and re-judged when rewriting is enabled
REWRITE_RATE = 0.2


def _price(prices: Mapping[str, Mapping[str, float]], model: str) -> dict[str, float]:
    merged = {**DEFAULT_PRICES, **{k: dict(v) for k, v in prices.items()}}
    return dict(merged.get(model) or merged.get(model.split("/")[-1]) or {"in": 0.0, "out": 0.0})


def _count_seeds(cfg: Mapping[str, Any]) -> tuple[int, str]:
    """Count seeds without touching the network (expansion is estimated separately)."""
    seeds_cfg = dict(cfg.get("seeds") or {})
    source = str(seeds_cfg.get("source", "taxonomy"))
    if source == "file" and seeds_cfg.get("file"):
        from .utils import read_jsonl

        path = resolve_path(cfg, str(seeds_cfg["file"]))
        rows = read_jsonl(path) if path.exists() else []
        topics = [str(r.get("topic", "")) for r in rows]
    else:
        seeds = build_seeds({**cfg, "seeds": {**seeds_cfg, "expansion": {"enabled": False}}})
        topics = [s.topic for s in seeds]

    expansion = dict(seeds_cfg.get("expansion") or {})
    domains = [str(d) for d in (seeds_cfg.get("domains") or [])]
    extra = 0
    if expansion.get("enabled"):
        expansion_domains = expansion.get("domains") or domains or ["general knowledge"]
        extra = int(expansion.get("per_domain", 8)) * len(expansion_domains)
    limit = seeds_cfg.get("limit")
    total = len(topics) + extra
    if limit:
        total = min(total, int(limit))
    return total, source


def estimate_run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate calls, tokens and spend for a config."""
    gen = dict(cfg.get("generation") or {})
    models = dict(cfg.get("models") or {})
    judge_cfg = dict(cfg.get("judge") or {})
    provider_cfg = dict(cfg.get("provider") or {})
    budget_cfg = dict(cfg.get("budget") or {})
    prices = dict(models.get("prices") or {})

    generator_model = str(models.get("generator") or "")
    judge_model = str(judge_cfg.get("model") or models.get("judge") or generator_model)

    seed_count, seed_source = _count_seeds(cfg)
    per_seed = int(gen.get("seeds_per_topic", 6))
    candidates = max(1, int(gen.get("response_candidates", 1)))
    instruction_count = seed_count * per_seed
    judge_enabled = bool(judge_cfg.get("enabled", True))
    sample_rate = float(judge_cfg.get("sample_rate", 1.0)) if judge_enabled else 0.0
    rewrite_on = bool((judge_cfg.get("rewrite") or {}).get("enabled", True)) and judge_enabled

    calls: dict[str, float] = {
        "instructions": float(seed_count),
        "response": float(instruction_count * candidates),
    }
    if judge_enabled:
        calls["judge"] = float(instruction_count * sample_rate * (1 + (REWRITE_RATE if rewrite_on else 0)))
    if rewrite_on:
        calls["rewrite"] = float(instruction_count * sample_rate * REWRITE_RATE)
    if (dict(cfg.get("seeds") or {}).get("expansion") or {}).get("enabled"):
        expansion = dict((cfg.get("seeds") or {}).get("expansion") or {})
        domains = expansion.get("domains") or (cfg.get("seeds") or {}).get("domains") or ["general knowledge"]
        calls["seed_expansion"] = float(len(domains))

    judge_tasks = {"judge"}
    breakdown: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost = 0.0
    for task, count in calls.items():
        prompt_tokens, completion_tokens = TOKEN_ASSUMPTIONS.get(task, (300, 300))
        model = judge_model if task in judge_tasks else generator_model
        price = _price(prices, model)
        cost = (
            count * prompt_tokens / 1_000_000 * price.get("in", 0.0)
            + count * completion_tokens / 1_000_000 * price.get("out", 0.0)
        )
        tokens = int(count * (prompt_tokens + completion_tokens))
        total_tokens += tokens
        total_cost += cost
        breakdown.append(
            {
                "task": task,
                "calls": round(count, 1),
                "model": model,
                "prompt_tokens": int(count * prompt_tokens),
                "completion_tokens": int(count * completion_tokens),
                "total_tokens": tokens,
                "usd": round(cost, 4),
            }
        )

    rate = provider_cfg.get("rate_limit_per_minute")
    concurrency = int(gen.get("concurrency", 8)) or 1
    total_calls = sum(calls.values())
    minutes = None
    if rate:
        minutes = round(total_calls / float(rate), 1)

    estimate = {
        "seeds": seed_count,
        "seed_source": seed_source,
        "instructions": instruction_count,
        "calls": {k: round(v, 1) for k, v in calls.items()},
        "total_calls": round(total_calls, 1),
        "total_tokens": total_tokens,
        "estimated_usd": round(total_cost, 4),
        "estimated_minutes": minutes,
        "concurrency": concurrency,
        "rate_limit_per_minute": rate,
        "breakdown": breakdown,
        "budget": {k: v for k, v in budget_cfg.items() if v is not None},
        "warnings": _warnings(total_cost, total_calls, minutes, budget_cfg, judge_enabled, candidates),
    }
    return estimate


def _warnings(
    cost: float,
    calls: float,
    minutes: float | None,
    budget: Mapping[str, Any],
    judge_enabled: bool,
    candidates: int,
) -> list[str]:
    out: list[str] = []
    max_usd = budget.get("max_usd")
    if max_usd is not None and cost > float(max_usd):
        out.append(
            f"estimated cost {_money(cost)} exceeds budget.max_usd={max_usd}; "
            "the run will stop early and stay resumable"
        )
    max_requests = budget.get("max_requests")
    if max_requests is not None and calls > float(max_requests):
        out.append(f"estimated calls {calls:,.0f} exceed budget.max_requests={max_requests}")
    if minutes is not None and minutes > 120:
        out.append(f"at the configured rate limit this takes ~{minutes / 60:.1f} hours")
    if not judge_enabled:
        out.append("judging is disabled: quality control falls back to deterministic filters only")
    if candidates > 1:
        out.append(f"response_candidates={candidates} multiplies answer generation and judging cost")
    return out


def _money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def render_estimate(estimate: Mapping[str, Any]) -> str:
    lines = [
        "cost estimate (rough — token shapes are assumptions, not measurements)",
        "",
        f"  seeds:            {estimate['seeds']:,} ({estimate['seed_source']})",
        f"  instructions:     {estimate['instructions']:,}",
        f"  total model calls:{estimate['total_calls']:>12,.0f}",
        f"  total tokens:     {estimate['total_tokens']:>12,}",
        f"  estimated spend:  {_money(float(estimate['estimated_usd'])):>12}",
    ]
    if estimate.get("estimated_minutes"):
        lines.append(f"  at rate limit:    {estimate['estimated_minutes']:>12,.1f} min")
    lines.append("")
    lines.append(f"  {'task':<16}{'calls':>10}  {'model':<26}{'tokens':>12}{'usd':>10}")
    for row in estimate["breakdown"]:
        lines.append(
            f"  {row['task']:<16}{row['calls']:>10,.0f}  {row['model']:<26}"
            f"{row['total_tokens']:>12,}{row['usd']:>10.4f}"
        )
    if estimate.get("warnings"):
        lines.append("")
        for warning in estimate["warnings"]:
            lines.append(f"  ! {warning}")
    return "\n".join(lines)


__all__ = ["TOKEN_ASSUMPTIONS", "estimate_run", "render_estimate"]
