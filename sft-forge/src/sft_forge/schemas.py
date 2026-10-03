"""Typed records that flow through the pipeline.

Everything serialises to plain dicts so artifacts stay human-readable JSONL and can be
diffed, grepped, and hand-edited between stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .utils import short_id, utcnow_iso

SCHEMA_VERSION = "1.0"

STATUS_GENERATED = "generated"
STATUS_FILTERED = "filtered"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

ROLES = ("system", "user", "assistant")


@dataclass
class Seed:
    """One unit of diversity: *what* the model should be trained on.

    A seed is intentionally coarse (a topic + task type + difficulty + audience). The
    generator expands it into concrete instructions.
    """

    topic: str
    domain: str = "general"
    task_type: str = "open_qa"
    difficulty: str = "medium"
    audience: str = "general audience"
    constraints: list[str] = field(default_factory=list)
    language: str = "English"
    id: str = ""
    source: str = "taxonomy"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "seed_" + short_id(
                self.topic, self.domain, self.task_type, self.difficulty, self.audience
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "domain": self.domain,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "audience": self.audience,
            "constraints": list(self.constraints),
            "language": self.language,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Seed:
        known = {
            "id",
            "topic",
            "domain",
            "task_type",
            "difficulty",
            "audience",
            "constraints",
            "language",
            "source",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("topic", "")
        constraints = kwargs.get("constraints") or []
        if isinstance(constraints, str):
            constraints = [constraints]
        kwargs["constraints"] = list(constraints)
        return cls(**kwargs)

    def describe(self) -> str:
        bits = [
            f"Topic: {self.topic}",
            f"Domain: {self.domain}",
            f"Task type: {self.task_type}",
            f"Difficulty: {self.difficulty}",
            f"Audience: {self.audience}",
            f"Language: {self.language}",
        ]
        if self.constraints:
            bits.append("Constraints: " + "; ".join(self.constraints))
        return "\n".join(bits)


@dataclass
class Record:
    """A single SFT training example plus all provenance needed to audit it."""

    instruction: str
    response: str = ""
    system: str | None = None
    id: str = ""
    seed_id: str = ""
    domain: str = "general"
    task_type: str = "open_qa"
    difficulty: str = "medium"
    audience: str = "general audience"
    status: str = STATUS_GENERATED
    scores: dict[str, float] | None = None
    judge: dict[str, Any] | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)
    rewrite_of: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = utcnow_iso()
        if not self.id:
            self.id = "ex_" + short_id(self.instruction, self.system or "")

    # -- derived views ---------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """OpenAI/HF chat-format message list (system first when present)."""
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.instruction})
        msgs.append({"role": "assistant", "content": self.response})
        return msgs

    @property
    def overall_score(self) -> float | None:
        if not self.scores:
            return None
        value = self.scores.get("overall")
        return float(value) if value is not None else None

    @property
    def is_accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    def reject(self, reason: str) -> Record:
        self.status = STATUS_REJECTED
        if reason not in self.reject_reasons:
            self.reject_reasons.append(reason)
        return self

    # -- serialisation ---------------------------------------------------------------

    def to_dict(self, include_messages: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "seed_id": self.seed_id,
            "domain": self.domain,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "audience": self.audience,
            "system": self.system,
            "instruction": self.instruction,
            "response": self.response,
            "status": self.status,
            "scores": self.scores,
            "judge": self.judge,
            "filters": self.filters,
            "reject_reasons": self.reject_reasons,
            "rewrite_of": self.rewrite_of,
            "usage": self.usage,
            "meta": self.meta,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if include_messages:
            data["messages"] = self.messages
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Record:
        payload = {k: v for k, v in data.items() if k != "messages"}
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in allowed}
        payload.setdefault("instruction", "")
        return cls(**payload)

    def export_view(
        self, fmt: str, include_scores: bool = True, include_system: bool = True
    ) -> dict[str, Any]:
        """Render into a training format. See :mod:`sft_forge.export` for the list."""
        system = self.system if include_system else None
        if fmt == "chatml":
            messages = self.messages
            if not include_system and system is None:
                messages = [m for m in messages if m["role"] != "system"]
            out: dict[str, Any] = {"messages": messages}
        elif fmt == "alpaca":
            out = {
                "instruction": self.instruction,
                "input": "",
                "output": self.response,
            }
            if system:
                out["system"] = system
        elif fmt == "sharegpt":
            convo = [{"from": "human", "value": self.instruction}]
            if system:
                convo.insert(0, {"from": "system", "value": system})
            convo.append({"from": "gpt", "value": self.response})
            out = {"conversations": convo}
        elif fmt == "pairs":
            prompt = self.instruction
            if system:
                prompt = f"{system}\n\n{prompt}"
            out = {"prompt": prompt, "completion": self.response}
        else:
            raise ValueError(f"unknown export format: {fmt!r}")

        if include_scores:
            meta: dict[str, Any] = {
                "id": self.id,
                "seed_id": self.seed_id,
                "domain": self.domain,
                "task_type": self.task_type,
                "difficulty": self.difficulty,
                "source": self.meta.get("source", "sft-forge"),
            }
            if self.scores:
                meta["scores"] = self.scores
            out["metadata"] = meta
        return out
