"""Deterministic offline provider.

Two jobs:

1. Let ``sft-forge run -c configs/demo_mock.yaml`` exercise the whole pipeline with no
   API key, so the workflow is testable end-to-end (including in CI).
2. Give the test-suite a stable, network-free backend.

It inspects the ``hint`` the pipeline attaches to each call and returns something shaped
like a real answer for that task.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Sequence
from typing import Any

from ..utils import approx_tokens
from .base import Completion, Message, Provider

_FIELD_RE = re.compile(
    r"^\s*[-*]?\s*(Topic|Domain|Task type|Difficulty|Audience|Language):\s*(.+)$", re.MULTILINE
)

_INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
    "open_qa": [
        "What are the practical trade-offs people should know about {topic}?",
        "Why does {topic} matter in practice, and where does it break down?",
        "What is the most common misconception about {topic}?",
    ],
    "explanation": [
        "Explain {topic} to a {audience} in plain language.",
        "Give me a short, intuition-first explanation of {topic}.",
        "Explain how {topic} works, using one concrete analogy.",
    ],
    "howto": [
        "Walk me through how to get started with {topic}, step by step.",
        "What is a sensible checklist for {topic} aimed at a {audience}?",
        "Describe a reliable procedure for {topic}, including common failure modes.",
    ],
    "summarization": [
        "Summarise the key ideas behind {topic} in five bullet points.",
        "Condense the essentials of {topic} into a one-paragraph brief for a {audience}.",
    ],
    "rewriting": [
        "Rewrite this note about {topic} so a {audience} can act on it: 'we should probably look at {topic} at some point, it seems important and there are risks'.",
        "Make this update about {topic} clearer and more concise: 'as discussed, {topic} is being reviewed and next steps will follow shortly'.",
    ],
    "brainstorming": [
        "List eight ideas for improving outcomes around {topic}.",
        "Brainstorm approaches to {topic} that a small team could try this quarter.",
    ],
    "analysis": [
        "Compare the main approaches to {topic} and recommend one for a {audience}.",
        "Analyse the risks and benefits of {topic}, and state what you would do.",
    ],
    "planning": [
        "Draft a 30-day plan to make progress on {topic} for a {audience}.",
        "Design a lightweight plan for {topic}, with milestones and success metrics.",
    ],
    "classification": [
        "Given a short support message about {topic}, classify its intent and urgency, and explain the label.",
        "Define a labelling scheme for {topic} and apply it to two example items.",
    ],
    "extraction": [
        "Show how to extract the key structured fields from a document about {topic}.",
        "Describe a robust extraction pipeline for records about {topic}, including edge cases.",
    ],
    "debugging": [
        "A system using {topic} intermittently returns wrong results. How would you debug it?",
        "What are the likely root causes when {topic} behaves inconsistently in production?",
    ],
    "coding": [
        "Write a small, well-documented Python utility that helps with {topic}.",
        "Implement a function related to {topic} and explain its complexity.",
    ],
    "math": [
        "Work through a quantitative example involving {topic}, showing each step.",
        "Set up the equations behind a simple {topic} problem and solve them.",
    ],
}

_RESPONSE_BODY = (
    "{topic} is easiest to reason about once you separate what is stable from what changes "
    "with context. In practice, three things matter most for a {audience}: the goal you are "
    "optimising, the constraints you cannot move, and the feedback loop that tells you "
    "whether the choice was right.\n\n"
    "**Key points**\n"
    "1. Start from the decision you are actually making — not from the tool that is trendy.\n"
    "2. Make the smallest change that produces observable signal, then iterate.\n"
    "3. Write down the trade-offs you accepted, so the next person can revisit them.\n\n"
    "A worked example: suppose you are applying {topic} under a tight budget. The cheap first "
    "step is to measure the current baseline, because most disagreements here are about "
    "unmeasured assumptions rather than about technique. Once the baseline is visible, the "
    "highest-leverage option is usually the one that shortens the feedback loop.\n\n"
    "**Common pitfalls**: optimising a proxy that drifts from the real goal, treating a "
    "one-off success as a general rule, and skipping the check that would falsify your "
    "assumption early. For a {audience}, the practical takeaway is to keep the first version "
    "small, instrumented, and reversible."
)

_DEEP_DIVE = (
    "\n\n**Where this gets harder.** The advice above assumes the goal is stable and the "
    "feedback is fast. When {topic} is applied where neither holds — long horizons, noisy "
    "signals, or a decision that is hard to reverse — the balance shifts toward reversible "
    "steps and explicit assumptions. Ask what evidence would change your mind, and how long "
    "it would take to obtain, before committing further resources."
)

_CODE_BODY = """```python
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Item:
    name: str
    weight: float


def rank(items: Iterable[Item], *, threshold: float = 0.0) -> list[Item]:
    \"\"\"Return items above `threshold`, heaviest first.

    {topic_note}
    \"\"\"
    return sorted((i for i in items if i.weight > threshold), key=lambda i: -i.weight)
```

**Why this shape**: the dataclass keeps callers honest about the fields they pass, the
generator body keeps memory flat for large inputs, and the explicit `threshold` makes the
filtering rule testable. Complexity is `O(n log n)` from the sort and `O(n)` extra space."""


class MockProvider(Provider):
    """Offline stand-in for a chat model."""

    name = "mock"

    def __init__(self, *, latency_s: float = 0.0, strict_rate: float = 1.0):
        self.latency_s = latency_s
        #: share of judged items that should pass the default quality bar
        self.strict_rate = strict_rate

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _fields(text: str) -> dict[str, str]:
        return {m.group(1).lower().replace(" ", "_"): m.group(2).strip() for m in _FIELD_RE.finditer(text or "")}

    @staticmethod
    def _rng(text: str, salt: str = "") -> random.Random:
        seed = int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:12], 16)
        return random.Random(seed)

    def _usage(self, messages: Sequence[Message], text: str) -> dict[str, int]:
        prompt_tokens = sum(approx_tokens(m.get("content", "")) for m in messages)
        completion_tokens = approx_tokens(text)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    # -- behaviours ---------------------------------------------------------------------

    def _seed_expansion(self, prompt: str, rng: random.Random) -> str:
        domains = re.findall(r"^\s*-\s*(.+)$", prompt, re.MULTILINE) or ["general knowledge"]
        topics = []
        for i in range(3):
            base = domains[i % len(domains)].strip().strip('",')
            topics.append(
                {
                    "topic": f"{base} — {rng.choice(['choosing between options', 'avoiding common failures', 'measuring impact', 'scaling up safely'])}",
                    "task_type": rng.choice(["open_qa", "explanation", "howto", "analysis", "planning"]),
                    "difficulty": rng.choice(["easy", "medium", "hard"]),
                }
            )
        return json.dumps(topics, ensure_ascii=False)

    def _instructions(self, prompt: str, rng: random.Random) -> str:
        fields = self._fields(prompt)
        topic = fields.get("topic", "the subject")
        task_type = fields.get("task_type", "open_qa")
        audience = fields.get("audience", "general audience")
        templates = _INSTRUCTION_TEMPLATES.get(task_type, _INSTRUCTION_TEMPLATES["open_qa"])
        pool = templates[:]
        rng.shuffle(pool)
        count = 2
        match = re.search(r"exactly (\d+) instructions", prompt)
        if match:
            count = int(match.group(1))
        out = [
            {
                "instruction": pool[i % len(pool)].format(topic=topic, audience=audience),
                "difficulty": fields.get("difficulty", "medium"),
                "tags": [task_type],
            }
            for i in range(max(1, min(count, len(pool))))
        ]
        return json.dumps(out, ensure_ascii=False)

    def _response(self, prompt: str, rng: random.Random) -> str:
        fields = self._fields(prompt)
        topic = fields.get("topic", "the topic")
        topic = topic[0].upper() + topic[1:] if topic else topic
        audience = fields.get("audience", "general audience")
        task_type = fields.get("task_type", "open_qa")
        difficulty = fields.get("difficulty", "medium")
        if task_type == "coding":
            return _CODE_BODY.format(topic_note=f"Applied to {topic}.", topic=topic)

        body = _RESPONSE_BODY.format(topic=topic, audience=audience)
        if difficulty == "hard":
            body += "\n\n" + _DEEP_DIVE.format(topic=topic.lower())
        elif difficulty == "easy":
            body = body.split("**Common pitfalls**")[0].strip()
        return body

    def _judge(self, prompt: str, rng: random.Random) -> str:
        roll = rng.random()
        base = 5 if roll > 0.25 else 4 if roll > 0.08 else 3
        penalty = rng.random()
        scores = {
            "correctness": base,
            "helpfulness": base if penalty > 0.15 else max(3, base - 1),
            "clarity": min(5, base if penalty > 0.3 else base + 1),
            "safety": 5 if penalty > 0.05 else 3,
            "format": 5 if penalty > 0.2 else 4,
        }
        overall = round(sum(scores.values()) / len(scores), 2)
        verdict = "accept" if overall >= 4.0 else "revise" if overall >= 3.0 else "reject"
        return json.dumps(
            {
                "scores": scores,
                "overall": overall,
                "verdict": verdict,
                "rationale": (
                    "Offline mock judgement: the answer is coherent and on-topic; "
                    f"strongest on clarity, weakest on {'helpfulness' if penalty <= 0.15 else 'format'}."
                ),
                "issues": [] if verdict == "accept" else ["could be more concrete about trade-offs"],
            },
            ensure_ascii=False,
        )

    def _rewrite(self, prompt: str, rng: random.Random) -> str:
        fields = self._fields(prompt)
        topic = fields.get("topic", "the topic")
        audience = fields.get("audience", "general audience")
        return _RESPONSE_BODY.format(topic=topic, audience=audience)

    # -- entry point --------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str = "mock-model",
        temperature: float = 1.0,  # noqa: ARG002
        top_p: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        response_format: dict[str, Any] | None = None,  # noqa: ARG002
        seed: int | None = None,
        hint: str | None = None,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Completion:
        started = time.perf_counter()
        prompt = "\n".join(m.get("content", "") for m in messages)
        rng = self._rng(prompt, salt=str(seed) + (hint or ""))

        if hint == "seed_expansion":
            text = self._seed_expansion(prompt, rng)
        elif hint == "instructions":
            text = self._instructions(prompt, rng)
        elif hint == "judge":
            text = self._judge(prompt, rng)
        elif hint == "rewrite":
            text = self._rewrite(prompt, rng)
        else:
            text = self._response(prompt, rng)

        if self.latency_s:
            time.sleep(self.latency_s)
        return Completion(
            text=text,
            model=model or "mock-model",
            usage=self._usage(messages, text),
            finish_reason="stop",
            latency_s=time.perf_counter() - started,
            provider=self.name,
        )
