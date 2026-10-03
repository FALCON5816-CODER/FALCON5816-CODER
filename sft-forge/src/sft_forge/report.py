"""Human-readable run report (``report.md``).

A dataset you cannot inspect is a dataset you cannot trust. The report answers, in one
screen: what was generated, how much survived, why the rest was dropped, how balanced the
coverage is, and what it cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schemas import Record
from .stats import compute_stats
from .utils import truncate

BAR_WIDTH = 24


def _bar(value: float, maximum: float, width: int = BAR_WIDTH) -> str:
    if maximum <= 0:
        return ""
    filled = int(round(width * value / maximum))
    return "█" * max(0, min(width, filled))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_report(
    records: Sequence[Record],
    cfg: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    client_stats: Mapping[str, Any] | None = None,
    timings: Mapping[str, float] | None = None,
    sample_count: int = 5,
) -> str:
    stats = compute_stats(records, extra=dict(client_stats or {}))
    counts = stats["counts"]
    coverage = stats["coverage"]
    scores = stats["scores"]
    run_cfg = dict(cfg.get("run") or {})
    models = dict(cfg.get("models") or {})
    judge_cfg = dict(cfg.get("judge") or {})

    out: list[str] = []
    out.append(f"# Dataset report — {run_cfg.get('name', 'run')}")
    out.append("")
    out.append(
        f"Generated {counts['generated']} candidate examples; **{counts['accepted']} accepted** "
        f"({counts['accept_rate'] * 100:.1f}%) and {counts['rejected']} dropped."
    )
    out.append("")

    out.append("## Run configuration")
    out.append("")
    out.append(
        _table(
            ["Setting", "Value"],
            [
                ["Generator model", models.get("generator")],
                ["Judge model", judge_cfg.get("model") or models.get("judge")],
                ["Judge threshold", judge_cfg.get("min_overall")],
                ["Seeds", (cfg.get("seeds") or {}).get("source")],
                ["Instructions per seed", (cfg.get("generation") or {}).get("seeds_per_topic")],
                ["Concurrency", (cfg.get("generation") or {}).get("concurrency")],
                ["Config hash", (manifest or {}).get("config_hash")],
                ["Dataset fingerprint", (manifest or {}).get("dataset_fingerprint")],
            ],
        )
    )
    out.append("")

    if timings:
        total = sum(timings.values())
        out.append("## Stage timings")
        out.append("")
        out.append(
            _table(
                ["Stage", "Seconds", "Share"],
                [
                    [name, f"{seconds:.1f}", f"{seconds / total * 100:.0f}%" if total else "-"]
                    for name, seconds in timings.items()
                ],
            )
        )
        out.append("")

    if client_stats:
        cache = client_stats.get("cache") or {}
        spend = [
            ["Model calls", client_stats.get("requests")],
            ["Tokens", client_stats.get("tokens")],
            ["Estimated cost", f"${float(client_stats.get('estimated_usd') or 0):.4f}"],
            ["Errors", client_stats.get("errors")],
            [
                "Cache hit rate",
                f"{cache.get('hit_rate', 0) * 100:.0f}% ({cache.get('hits', 0)} hits)" if cache else "disabled",
            ],
        ]
        out.append("## Spend and reliability")
        out.append("")
        out.append(_table(["Metric", "Value"], spend))
        out.append("")

    out.append("## Coverage")
    out.append("")
    out.append(
        f"Domains: **{coverage['domains']}**, topics: **{coverage['topics']}**, "
        f"balance gini: **{coverage['domain_balance_gini']:.2f}** (lower is more even)."
    )
    out.append("")
    domain_dist = coverage["domain_distribution"]
    top_domains = list(domain_dist.items())[:12]
    if top_domains:
        maximum = max(domain_dist.values())
        out.append(
            _table(
                ["Domain", "Accepted", ""],
                [[name, n, _bar(n, maximum)] for name, n in top_domains],
            )
        )
        out.append("")
    for label, key in (("Task types", "task_type_distribution"), ("Difficulty", "difficulty_distribution")):
        dist = coverage[key]
        if dist:
            out.append(
                _table(
                    [label, "Accepted", ""],
                    [[name, n, _bar(n, max(dist.values()))] for name, n in dist.items()],
                )
            )
            out.append("")

    out.append("## Answer quality")
    out.append("")
    words = stats["response_words"]
    out.append(
        _table(
            ["Metric", "Accepted", "Rejected"],
            [
                ["Median words", words["accepted"].get("p50"), words["rejected"].get("p50")],
                ["p5 words", words["accepted"].get("p5"), words["rejected"].get("p5")],
                ["p95 words", words["accepted"].get("p95"), words["rejected"].get("p95")],
            ],
        )
    )
    out.append("")
    if scores["overall"]:
        out.append(
            _table(
                ["Dimension", "Mean", "p5"],
                [
                    [dim, values.get("mean"), values.get("p5")]
                    for dim, values in scores["by_dimension"].items()
                    if values
                ],
            )
        )
        out.append("")

    reasons = stats["rejections"]["reasons"]
    if reasons:
        maximum = max(reasons.values())
        out.append("## Why examples were dropped")
        out.append("")
        out.append(
            _table(
                ["Reason", "Count", ""],
                [[reason, n, _bar(n, maximum)] for reason, n in list(reasons.items())[:12]],
            )
        )
        out.append("")

    accepted = [r for r in records if r.is_accepted]
    if accepted and sample_count:
        out.append("## Random sample of accepted examples")
        out.append("")
        step = max(1, len(accepted) // sample_count)
        for record in accepted[::step][:sample_count]:
            out.append(f"### `{record.id}` — {record.domain} / {record.task_type} / {record.difficulty}")
            out.append("")
            out.append(f"**Instruction:** {truncate(record.instruction, 500)}")
            out.append("")
            out.append(f"**Response:** {truncate(record.response, 700)}")
            if record.scores:
                out.append("")
                out.append(f"_overall {record.scores.get('overall')}/5 — {truncate(str((record.judge or {}).get('rationale', '')), 220)}_")
            out.append("")

    out.append("---")
    out.append("")
    out.append(
        "Next steps: spot-check the sample above, open `rejected.jsonl` to see what the "
        "filters removed, and tune `judge.min_overall` / `filters.*` before scaling up the run."
    )
    return "\n".join(out) + "\n"


def write_report(
    run_dir: str | Path,
    records: Sequence[Record],
    cfg: Mapping[str, Any],
    **kwargs: Any,
) -> Path:
    from .utils import atomic_write_text

    path = Path(run_dir) / "report.md"
    atomic_write_text(path, render_report(records, cfg, **kwargs))
    return path


__all__ = ["render_report", "write_report"]
