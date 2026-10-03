"""Stage 2-3: turn seeds into concrete instructions, then into answers.

* **Instructions** — one call per seed asks for several self-contained user requests. The
  seed's topic, difficulty, audience and constraints are injected into the prompt, and a
  rotating style hint keeps the phrasing from collapsing into one template.
* **Responses** — one call per instruction. When ``generation.response_candidates > 1``
  several answers are produced and the judge later keeps the best one (best-of-n).

Both stages fall back gracefully: an unparseable instruction response degrades to a
deterministic template, and a failed response call is retried by the client, then skipped
and counted rather than killing the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import BudgetExceeded, LLMClient
from .jsonx import as_list, loads_lenient
from .prompts import GENERATOR_SYSTEM, instruction_messages, response_messages
from .schemas import STATUS_GENERATED, Record, Seed
from .utils import get_logger, short_id

_TEMPLATES: dict[str, list[str]] = {
    "open_qa": [
        "What should someone know about {topic}?",
        "Why does {topic} matter in practice?",
        "What is the most common misconception about {topic}?",
        "When does {topic} stop being worth the effort?",
        "How has your understanding of {topic} changed over time?",
    ],
    "explanation": [
        "Explain {topic} to a {audience}.",
        "Give a clear, plain-language explanation of {topic}.",
        "Break {topic} down into its core ideas.",
        "Why does {topic} work the way it does?",
        "Explain {topic} using one concrete analogy.",
    ],
    "howto": [
        "How would you approach {topic} step by step?",
        "What is a practical process for {topic}?",
        "Walk a {audience} through getting started with {topic}.",
        "What checklist would you use for {topic}?",
        "How do you avoid the usual mistakes when doing {topic}?",
    ],
    "analysis": [
        "Compare the main options for {topic} and recommend one.",
        "What are the trade-offs involved in {topic}?",
        "Where does the common advice about {topic} break down?",
        "Which factors should drive a decision about {topic}?",
        "Argue both sides of {topic} and then take a position.",
    ],
    "planning": [
        "Draft a plan for {topic} for a {audience}.",
        "How should a {audience} plan for {topic}?",
        "What would a 30-day plan for {topic} look like?",
        "Design a lightweight plan for {topic} with milestones.",
        "What would you do first, second and last for {topic}?",
    ],
    "brainstorming": [
        "Give me several ideas for {topic}.",
        "Brainstorm approaches to {topic}.",
        "What are some unconventional angles on {topic}?",
        "List ideas for improving outcomes around {topic}.",
        "What could a small team try this quarter for {topic}?",
    ],
    "summarization": [
        "Summarise the essentials of {topic} in a short brief.",
        "Condense {topic} into its key points.",
        "Give me a five-bullet summary of {topic}.",
        "Turn {topic} into a one-paragraph brief for a {audience}.",
        "What are the three things worth remembering about {topic}?",
    ],
    "rewriting": [
        "Rewrite a note about {topic} so a {audience} can act on it.",
        "Improve this draft about {topic}.",
        "Make this text about {topic} clearer and shorter.",
        "Turn this rough note about {topic} into a polished message.",
        "Rewrite this update about {topic} for a non-expert reader.",
    ],
    "classification": [
        "Classify the typical cases involved in {topic} and explain the labels.",
        "How would you label examples of {topic}?",
        "Design a labelling scheme for {topic} and apply it to two examples.",
        "What categories should be used to sort instances of {topic}?",
        "Given a short message about {topic}, classify its intent and urgency.",
    ],
    "extraction": [
        "How would you extract structured information about {topic}?",
        "Pull the key fields out of a document about {topic}.",
        "Describe a robust extraction pipeline for records about {topic}.",
        "Which fields matter most when extracting data about {topic}?",
        "How do you handle messy inputs when extracting {topic}?",
    ],
    "debugging": [
        "Something involving {topic} is going wrong intermittently. How do you debug it?",
        "What usually causes failures in {topic}?",
        "How would you isolate the cause of a problem with {topic}?",
        "What would you check first when {topic} misbehaves?",
        "Diagnose why {topic} works in testing but fails in production.",
    ],
    "coding": [
        "Write a small Python utility related to {topic}.",
        "Implement and explain a function for {topic}.",
        "Show a minimal working example of {topic} in Python.",
        "How would you test code that handles {topic}?",
        "Refactor a naive implementation of {topic} for clarity.",
    ],
    "math": [
        "Work through a quantitative example involving {topic}.",
        "Set up and solve a simple {topic} problem.",
        "Show the equations behind {topic} and explain each step.",
        "Estimate an order-of-magnitude answer for a {topic} question.",
        "Where do people go wrong when reasoning quantitatively about {topic}?",
    ],
}


@dataclass
class WorkItem:
    """One instruction, tied back to the seed that produced it."""

    instruction: str
    seed: Seed
    difficulty: str = "medium"
    id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "wi_" + short_id(self.seed.id, self.instruction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.id,
            "instruction": self.instruction,
            "difficulty": self.difficulty,
            "seed": self.seed.to_dict(),
            "usage": self.usage,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> WorkItem:
        seed = Seed.from_dict(row.get("seed") or {"topic": row.get("instruction", "")})
        return cls(
            instruction=str(row.get("instruction", "")),
            seed=seed,
            difficulty=str(row.get("difficulty") or seed.difficulty),
            id=str(row.get("item_id") or ""),
            usage=dict(row.get("usage") or {}),
        )


def instruction_worker_factory(client: LLMClient, cfg: Mapping[str, Any]):
    """Build the per-seed worker used by the instructions stage."""
    gen = dict(cfg.get("generation") or {})
    model = str((cfg.get("models") or {}).get("generator") or "")
    per_seed = int(gen.get("seeds_per_topic", 6))
    temperature = float(gen.get("temperature", 0.9))
    top_p = gen.get("top_p")
    max_tokens = int(gen.get("max_tokens", 1200))
    require_json = bool(gen.get("require_json", True))
    extra = dict(gen.get("extra_body") or {})
    log = get_logger()

    def worker(seed: Seed) -> list[dict[str, Any]]:
        from .seeds import style_hint

        hints = [style_hint(i) for i in range(per_seed)]
        messages = instruction_messages(seed, per_seed, style_hints=hints)
        status = "ok"
        usage: dict[str, Any] = {}
        try:
            completion = client.chat(
                messages,
                model=model,
                temperature=temperature,
                top_p=float(top_p) if top_p is not None else None,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if require_json else None,
                hint="instructions",
                extra=extra or None,
            )
            usage = dict(completion.usage)
            payload = loads_lenient(completion.text)
            parsed = list(as_list(payload, "instructions", "items", "requests"))
        except BudgetExceeded:
            raise  # stop the stage cleanly rather than degrading every remaining seed
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing the stage
            log.warning("instruction generation failed for seed %s: %s", seed.id, exc)
            parsed, status = [], "error"

        instructions: list[tuple[str, str]] = []
        for item in parsed:
            if isinstance(item, str):
                text, difficulty = item, seed.difficulty
            elif isinstance(item, Mapping):
                text = str(item.get("instruction") or item.get("request") or "").strip()
                difficulty = str(item.get("difficulty") or seed.difficulty)
            else:
                continue
            text = text.strip()
            if text:
                instructions.append((text, difficulty))

        # de-duplicate within the seed, then top up with templates if the model under-delivered
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for text, difficulty in instructions:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                unique.append((text, difficulty))
        if len(unique) < per_seed:
            unique.extend((text, seed.difficulty) for text in fallback_instructions(seed, per_seed))

        # a second pass keeps the topped-up list free of repeats too
        deduped: list[tuple[str, str]] = []
        seen = set()
        for text, difficulty in unique:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((text, difficulty))

        rows: list[dict[str, Any]] = []
        for i, (text, difficulty) in enumerate(deduped[:per_seed]):
            item = WorkItem(
                instruction=text,
                seed=seed,
                difficulty=difficulty if difficulty in ("easy", "medium", "hard") else seed.difficulty,
                usage=usage if i == 0 else {},
            )
            row = item.to_dict()
            row["generation_status"] = status
            rows.append(row)
        return rows

    return worker


def fallback_instructions(seed: Seed, count: int) -> list[str]:
    """Deterministic templates so a seed still contributes when the model misbehaves.

    Wording is varied across a per-task pool; if more items are requested than the pool
    holds, repeats are disambiguated rather than emitted identically.
    """
    templates = _TEMPLATES.get(seed.task_type) or _TEMPLATES["open_qa"]
    out: list[str] = []
    for i in range(count):
        template = templates[i % len(templates)]
        text = template.format(topic=seed.topic, audience=seed.audience).strip()
        if i >= len(templates):
            variant = i // len(templates) + 1
            text = f"{text.rstrip('?.!')} (variant {variant})"
        out.append(text)
    return out


def response_worker_factory(client: LLMClient, cfg: Mapping[str, Any]):
    """Build the per-instruction worker used by the responses stage."""
    gen = dict(cfg.get("generation") or {})
    model = str((cfg.get("models") or {}).get("generator") or "")
    temperature = float(gen.get("temperature", 0.9))
    top_p = gen.get("top_p")
    max_tokens = int(gen.get("max_tokens", 1200))
    candidates = max(1, int(gen.get("response_candidates", 1)))
    guidance = str(gen.get("system_guidance") or "")
    extra = dict(gen.get("extra_body") or {})
    seed_value = int((cfg.get("run") or {}).get("seed", 17))

    def worker(item: WorkItem) -> dict[str, Any]:
        record = Record(
            instruction=item.instruction,
            seed_id=item.seed.id,
            domain=item.seed.domain,
            task_type=item.seed.task_type,
            difficulty=item.difficulty,
            audience=item.seed.audience,
        )
        record.system = f"{GENERATOR_SYSTEM}\n\n{guidance}" if guidance else GENERATOR_SYSTEM
        record.meta.update(
            {
                "source": "sft-forge",
                "generator_model": model,
                "topic": item.seed.topic,
                "item_id": item.id,
                "seed_source": item.seed.source,
            }
        )
        texts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for i in range(candidates):
            completion = client.chat(
                response_messages(record, extra_guidance=guidance),
                model=model,
                temperature=temperature,
                top_p=float(top_p) if top_p is not None else None,
                max_tokens=max_tokens,
                hint="response",
                seed=seed_value + i,
                extra=extra or None,
            )
            texts.append((completion.text or "").strip())
            for key in usage:
                usage[key] += int(completion.usage.get(key, 0))

        record.response = texts[0] if texts else ""
        record.usage = usage
        record.status = STATUS_GENERATED
        if len(texts) > 1:
            record.meta["candidates"] = texts
        return record.to_dict()

    return worker


def records_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[Record]:
    return [Record.from_dict(row) for row in rows]


__all__ = [
    "WorkItem",
    "fallback_instructions",
    "instruction_worker_factory",
    "records_from_rows",
    "response_worker_factory",
]
