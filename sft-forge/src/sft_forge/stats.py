"""Dataset statistics.

The point of these numbers is to answer the questions you ask before training:
is coverage balanced, are answers the right length, did quality actually improve, and
what is the data costing me?
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record
from .utils import word_count

_PERCENTILES = (5, 25, 50, 75, 95)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for p in _PERCENTILES:
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = round(ordered[idx], 2)
    out["mean"] = round(sum(ordered) / len(ordered), 2)
    out["max"] = round(ordered[-1], 2)
    return out


def _distribution(values: Iterable[str], top: int | None = None) -> dict[str, int]:
    counts = Counter(values)
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    if top is None:
        return ordered
    return dict(list(ordered.items())[:top])


def _gini(counts: Sequence[int]) -> float:
    """0 = perfectly even coverage, →1 = everything concentrated in a few buckets."""
    values = sorted(v for v in counts if v > 0)
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    cumulative = 0.0
    for i, value in enumerate(values, start=1):
        cumulative += i * value
    return round((2 * cumulative) / (len(values) * total) - (len(values) + 1) / len(values), 4)


def compute_stats(records: Sequence[Record], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute a full statistics bundle for a list of records."""
    accepted = [r for r in records if r.status == STATUS_ACCEPTED]
    rejected = [r for r in records if r.status == STATUS_REJECTED]

    def lens(rows: Sequence[Record]) -> dict[str, float]:
        return _percentiles([word_count(r.response) for r in rows])

    scores = [r.scores for r in records if r.scores]
    overall = [float(s["overall"]) for s in scores if s.get("overall") is not None]

    reason_counts: dict[str, int] = {}
    for record in rejected:
        for reason in record.reject_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    domain_counts = _distribution(r.domain for r in accepted)
    topic_counts = _distribution(str(r.meta.get("topic") or "unknown") for r in accepted)
    task_counts = _distribution(r.task_type for r in accepted)
    difficulty_counts = _distribution(r.difficulty for r in accepted)

    stats: dict[str, Any] = {
        "counts": {
            "generated": len(records),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accept_rate": round(len(accepted) / len(records), 4) if records else 0.0,
            "rewritten": sum(1 for r in records if r.meta.get("rewritten")),
            "judged": len(scores),
        },
        "response_words": {"accepted": lens(accepted), "rejected": lens(rejected)},
        "instruction_words": _percentiles([word_count(r.instruction) for r in accepted]),
        "characters": {
            "accepted_total": sum(len(r.response) for r in accepted),
            "tokens_estimate": sum(int(len(r.response) / 4) for r in accepted),
        },
        "scores": {
            "overall": _percentiles(overall),
            "by_dimension": {
                dim: _percentiles([float(s[dim]) for s in scores if dim in s])
                for dim in ("correctness", "helpfulness", "clarity", "instruction_following", "safety")
            },
        },
        "coverage": {
            "domains": len(domain_counts),
            "topics": len(topic_counts),
            "domain_distribution": domain_counts,
            "task_type_distribution": task_counts,
            "difficulty_distribution": difficulty_counts,
            "top_topics": _distribution(
                (str(r.meta.get("topic") or "unknown") for r in accepted), top=10
            ),
            "domain_balance_gini": _gini(list(domain_counts.values())),
        },
        "rejections": {"total": len(rejected), "reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1]))},
    }
    if extra:
        stats["run"] = extra
    return stats


def stats_summary_lines(stats: dict[str, Any]) -> list[str]:
    """Short human-readable summary used by the CLI."""
    counts = stats.get("counts", {})
    lines = [
        f"generated {counts.get('generated', 0)}  accepted {counts.get('accepted', 0)} "
        f"({counts.get('accept_rate', 0) * 100:.1f}%)  rejected {counts.get('rejected', 0)}",
    ]
    words = (stats.get("response_words") or {}).get("accepted") or {}
    if words:
        lines.append(f"answer length: median {words.get('p50', 0):.0f} words (p5 {words.get('p5', 0):.0f}, p95 {words.get('p95', 0):.0f})")
    scores = (stats.get("scores") or {}).get("overall") or {}
    if scores:
        lines.append(f"judge score: mean {scores.get('mean', 0):.2f}/5 (p5 {scores.get('p5', 0):.2f})")
    coverage = stats.get("coverage") or {}
    if coverage:
        lines.append(
            f"coverage: {coverage.get('domains', 0)} domains, {coverage.get('topics', 0)} topics, "
            f"balance gini {coverage.get('domain_balance_gini', 0):.2f}"
        )
    run = stats.get("run") or {}
    if run.get("tokens"):
        lines.append(f"spend: {run['tokens']} tokens, ~${run.get('estimated_usd', 0):.2f}")
    return lines


__all__ = ["compute_stats", "stats_summary_lines"]
