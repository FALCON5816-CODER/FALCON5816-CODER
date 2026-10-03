"""Every prompt used by the pipeline lives here.

Keeping prompts in one module makes them reviewable and diffable — prompt changes are the
most consequential change you can make to a synthetic-data pipeline, so they should be
easy to find and compare.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .schemas import Record, Seed

# ---------------------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You are a rigorous, helpful assistant. You answer requests directly and completely, "
    "in clear prose, without hedging, filler, or meta-commentary about being an AI. "
    "When a request is ambiguous you make a reasonable assumption, state it briefly, and "
    "answer anyway."
)

INSTRUCTION_WRITER_SYSTEM = (
    "You write realistic user requests for an instruction-following dataset. "
    "Each request must be self-contained, natural, and answerable without extra context. "
    "Vary sentence structure, length, and framing across items. Never mention datasets, "
    "training, prompts, or evaluations. Respond with JSON only."
)

JUDGE_SYSTEM = (
    "You are a demanding but fair evaluator of assistant answers. You apply the rubric "
    "literally, you penalise vague or padded answers, and you never reward length for its "
    "own sake. You reply with a single JSON object and nothing else."
)

REWRITER_SYSTEM = (
    "You improve draft answers so they fully satisfy the request. You keep what is correct, "
    "fix what is wrong or vague, and add the specifics a reader needs. You output only the "
    "improved answer — no preamble, no notes about what you changed."
)

SEED_EXPANSION_SYSTEM = (
    "You design topic taxonomies for building instruction datasets. You produce concrete, "
    "non-overlapping topics with real substance. Respond with JSON only."
)

JSON_OBJECT_FORMAT = {"type": "json_object"}

# ---------------------------------------------------------------------------------------
# Seed expansion
# ---------------------------------------------------------------------------------------


def seed_expansion_messages(domain: str, count: int, task_types: Sequence[str]) -> list[dict[str, str]]:
    task_list = ", ".join(task_types)
    user = (
        f"Domain: {domain}\n"
        f"Produce {count} distinct, concrete seed topics for this domain.\n\n"
        "Requirements:\n"
        "- Each topic must be specific enough to write about (not a vague category).\n"
        f"- Spread them across task types: {task_list}.\n"
        "- Cover different subtopics, audiences, and difficulty levels.\n"
        "- Avoid near-duplicates of each other.\n\n"
        'Return a JSON object: {"topics": [{"topic": "...", "task_type": "...", '
        '"difficulty": "easy|medium|hard", "audience": "..."}]}'
    )
    return [
        {"role": "system", "content": SEED_EXPANSION_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------------------
# Instruction writing
# ---------------------------------------------------------------------------------------

_SPECIFICITY_BY_DIFFICULTY = {
    "easy": "a straightforward request a competent non-expert could answer in a short paragraph",
    "medium": "a request that needs solid domain knowledge and a structured answer",
    "hard": "a request that requires expert judgement, nuance, or reasoning through trade-offs",
}


def instruction_messages(seed: Seed, count: int, style_hints: Sequence[str] = ()) -> list[dict[str, str]]:
    difficulty_note = _SPECIFICITY_BY_DIFFICULTY.get(seed.difficulty, _SPECIFICITY_BY_DIFFICULTY["medium"])
    constraints = ""
    if seed.constraints:
        constraints = "\nMust satisfy these constraints:\n" + "\n".join(f"- {c}" for c in seed.constraints)
    styles = ""
    if style_hints:
        styles = "\nVary the framing across items, e.g. " + ", ".join(f'"{s}"' for s in style_hints) + "."

    user = (
        f"{seed.describe()}\n\n"
        f"Write exactly {count} distinct user requests (the 'instruction' an assistant would receive).\n\n"
        "Rules:\n"
        f"- Each request should be {difficulty_note}.\n"
        "- Self-contained: no dangling pronouns, no references to earlier conversation.\n"
        "- Natural human phrasing: some may be questions, some task-style ('draft…', 'compare…').\n"
        "- Do not include the answer, or any instructions about formatting the reply as JSON.\n"
        "- Never mention training data, models, prompts, or the word 'assistant'.\n"
        f"{styles}{constraints}\n\n"
        'Return JSON: {"instructions": [{"instruction": "...", "difficulty": "easy|medium|hard"}]}'
    )
    return [
        {"role": "system", "content": INSTRUCTION_WRITER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------------------

_LENGTH_GUIDE = {
    "easy": "Answer in 80-200 words. No headings.",
    "medium": "Answer in 150-400 words. Use short paragraphs; headings or a short list if it helps.",
    "hard": "Answer in 250-600 words. Be precise, address trade-offs, and use structure where it aids clarity.",
}


def response_messages(record: Record, *, extra_guidance: str = "") -> list[dict[str, str]]:
    guide = _LENGTH_GUIDE.get(record.difficulty, _LENGTH_GUIDE["medium"])
    system = GENERATOR_SYSTEM
    if extra_guidance:
        system = f"{system}\n\n{extra_guidance}"

    user = (
        "Complete the following request as a high-quality assistant would.\n\n"
        "<request>\n"
        f"{record.instruction}\n"
        "</request>\n\n"
        "Requirements:\n"
        f"- Difficulty: {record.difficulty}\n"
        f"- Answer length: {guide}\n"
        "- Lead with the direct answer, then support it with detail or an example.\n"
        "- Be concrete: name specific steps, numbers, or cases instead of generalities.\n"
        "- Use Markdown only where it genuinely improves readability.\n"
        "- If the request is impossible or ill-posed, say so plainly and explain what you would do instead.\n"
        "- Do not restate the question, do not mention these requirements, and do not add a closing offer "
        "of further help."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------------------

RUBRIC: dict[str, str] = {
    "correctness": "Is the answer factually and logically sound? 5 = no errors; 3 = minor imprecision; 1 = wrong.",
    "helpfulness": "Does it actually resolve the request with enough specifics to act on? 5 = fully actionable; 1 = useless.",
    "clarity": "Is it well organised and easy to follow? 5 = crisp; 1 = confusing.",
    "instruction_following": "Does it respect the request's constraints, format, and scope?",
    "safety": "Is it free of harmful instructions, private data, and unsupported professional advice?",
}


def judge_messages(record: Record) -> list[dict[str, str]]:
    rubric_text = "\n".join(f"- {name}: {desc}" for name, desc in RUBRIC.items())
    user = (
        "Evaluate the candidate answer against the rubric. Score each dimension 1-5 (integers).\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        f"<request>\n{record.instruction}\n</request>\n\n"
        f"<candidate_answer>\n{record.response}\n</candidate_answer>\n\n"
        "Calibration: most competent answers land at 4. Reserve 5 for answers that are "
        "accurate, specific, and complete. Use 3 when something important is missing or vague, "
        "and 1-2 for answers that are wrong, unsafe, or ignore the request.\n\n"
        'Return JSON: {"scores": {"correctness": n, "helpfulness": n, "clarity": n, '
        '"instruction_following": n, "safety": n}, "overall": <mean of the five, 1-5>, '
        '"verdict": "accept" | "revise" | "reject", "rationale": "<one or two sentences>", '
        '"issues": ["<specific defect>", ...]}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def rewrite_messages(record: Record) -> list[dict[str, str]]:
    judge = record.judge or {}
    issues = judge.get("issues") or []
    issues_text = "\n".join(f"- {i}" for i in issues) or "- General vagueness or missing detail."
    rationale = judge.get("rationale") or ""
    user = (
        "Rewrite the draft answer so it fully satisfies the request.\n\n"
        f"<request>\n{record.instruction}\n</request>\n\n"
        f"<draft_answer>\n{record.response}\n</draft_answer>\n\n"
        f"Reviewer notes:\n{issues_text}\n"
        + (f"Reviewer summary: {rationale}\n" if rationale else "")
        + "\nKeep everything that is already correct. Fix each issue specifically — replace vague "
        "claims with concrete ones, correct errors, and drop padding. Match the depth the request "
        "implies. Output only the improved answer."
    )
    return [
        {"role": "system", "content": REWRITER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------

STYLE_HINTS = (
    "a direct question",
    "a request to compare options",
    "a task phrased as a command",
    "a request with an explicit constraint",
    "a request from a beginner's point of view",
    "a request that asks for a recommendation",
    "a troubleshooting request",
    "a request for a plan or checklist",
)


def json_schema_hint() -> str:
    """Human-readable reminder appended to prompts when a response format is unsupported."""
    return json.dumps({"type": "object", "required": ["instructions"]})
