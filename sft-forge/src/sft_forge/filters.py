"""Stage 5: cheap deterministic filters and near-duplicate removal.

LLM judging catches "is this answer good?"; these filters catch everything a judge is bad
at: degenerate length, looping text, prompt-copying, refusals, boilerplate, unsafe strings,
language drift, and near-duplicates of examples you already have.

Every outcome is recorded on the record (``record.filters``) so you can audit exactly why
something was dropped, and tune thresholds from evidence rather than guesswork.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record
from .utils import (
    english_ratio,
    ngram_repeat_ratio,
    non_ascii_ratio,
    normalize_for_dedup,
    unique_token_ratio,
    words,
)

# ---------------------------------------------------------------------------------------
# outcome plumbing
# ---------------------------------------------------------------------------------------


@dataclass
class FilterOutcome:
    name: str
    ok: bool
    reason: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "ok": self.ok}
        if self.reason:
            out["reason"] = self.reason
        if self.detail:
            out["detail"] = self.detail
        return out


REFUSAL_PATTERNS = (
    re.compile(r"^\s*i(?:'m| am) (?:sorry|unable|not able)", re.I),
    re.compile(r"^\s*i can(?:'t|not) (?:help|assist|provide)", re.I),
    re.compile(r"^\s*as an ai (?:language )?model", re.I),
    re.compile(r"^\s*i (?:must|have to) decline", re.I),
)


# ---------------------------------------------------------------------------------------
# per-record filters
# ---------------------------------------------------------------------------------------


def filter_length(record: Record, cfg: Mapping[str, Any]) -> FilterOutcome:
    """``cfg`` is the ``filters.length`` section."""
    c = dict(cfg or {})
    instruction, response = record.instruction.strip(), record.response.strip()
    if len(instruction) < int(c.get("min_instruction_chars", 8)):
        return FilterOutcome("length", False, "instruction_too_short", f"{len(instruction)} chars")
    if len(instruction) > int(c.get("max_instruction_chars", 4000)):
        return FilterOutcome("length", False, "instruction_too_long", f"{len(instruction)} chars")
    if not response:
        return FilterOutcome("length", False, "empty_response")
    if len(response) < int(c.get("min_response_chars", 40)):
        return FilterOutcome("length", False, "response_too_short", f"{len(response)} chars")
    if len(response) > int(c.get("max_response_chars", 12000)):
        return FilterOutcome("length", False, "response_too_long", f"{len(response)} chars")
    n_words = len(words(response))
    if n_words < int(c.get("min_response_words", 12)):
        return FilterOutcome("length", False, "response_too_few_words", f"{n_words} words")
    if n_words > int(c.get("max_response_words", 1200)):
        return FilterOutcome("length", False, "response_too_many_words", f"{n_words} words")
    return FilterOutcome("length", True)


def filter_quality(record: Record, cfg: Mapping[str, Any]) -> FilterOutcome:
    """``cfg`` is the ``filters.quality`` section."""
    c = dict(cfg or {})
    instruction, response = record.instruction, record.response

    for pattern in REFUSAL_PATTERNS:
        if pattern.search(response) and len(response) < 400:
            return FilterOutcome("quality", False, "refusal_like_response")

    uniqueness = unique_token_ratio(response)
    if uniqueness < float(c.get("min_unique_token_ratio", 0.22)):
        return FilterOutcome("quality", False, "low_lexical_diversity", f"{uniqueness:.2f}")

    repeat = ngram_repeat_ratio(response, n=4)
    if repeat > float(c.get("max_ngram_repeat_ratio", 0.4)):
        return FilterOutcome("quality", False, "repetitive_text", f"{repeat:.2f}")

    instr_tokens = set(words(instruction))
    resp_tokens = set(words(response))
    if instr_tokens and resp_tokens:
        overlap = len(instr_tokens & resp_tokens) / len(instr_tokens)
        if overlap > float(c.get("max_instruction_response_overlap", 0.9)) and len(resp_tokens) < 3 * len(instr_tokens):
            return FilterOutcome("quality", False, "response_parrots_instruction", f"{overlap:.2f}")

    # Placeholder / unfinished generations
    if response.rstrip().endswith(("...", "…")) and len(response) < 200:
        return FilterOutcome("quality", False, "truncated_response")
    if "{{" in response or "[insert" in response.lower():
        return FilterOutcome("quality", False, "unfilled_placeholder")

    return FilterOutcome("quality", True)


def filter_language(record: Record, cfg: Mapping[str, Any]) -> FilterOutcome:
    """``cfg`` is the ``filters.language`` section."""
    c = dict(cfg or {})
    min_english = float(c.get("min_english_ratio", 0.02))
    max_non_ascii = float(c.get("max_non_ascii_ratio", 0.5))
    ratio = english_ratio(record.response)
    if len(words(record.response)) > 25 and ratio < min_english:
        return FilterOutcome("language", False, "not_english_enough", f"stopword ratio {ratio:.3f}")
    non_ascii = non_ascii_ratio(record.response)
    if non_ascii > max_non_ascii:
        return FilterOutcome("language", False, "excessive_non_ascii", f"{non_ascii:.2f}")
    return FilterOutcome("language", True)


def filter_safety(record: Record, cfg: Mapping[str, Any]) -> FilterOutcome:
    """``cfg`` is the ``filters.safety`` section."""
    c = dict(cfg or {})
    haystack = f"{record.instruction}\n{record.response}".lower()
    for phrase in c.get("blocklist") or []:
        needle = str(phrase).strip().lower()
        if needle and needle in haystack:
            return FilterOutcome("safety", False, "blocklisted_phrase", str(phrase))
    for pattern in c.get("regex_denylist") or []:
        try:
            if re.search(str(pattern), haystack, re.I):
                return FilterOutcome("safety", False, "denylist_regex", str(pattern))
        except re.error:  # pragma: no cover - user-supplied pattern
            continue
    return FilterOutcome("safety", True)


FILTERS = {
    "length": filter_length,
    "quality": filter_quality,
    "language": filter_language,
    "safety": filter_safety,
}


# ---------------------------------------------------------------------------------------
# near-duplicate detection (minhash + LSH)
# ---------------------------------------------------------------------------------------

_MERSENNE = (1 << 61) - 1


def _ngram_ids(text: str, n: int, limit_tokens: int = 400) -> set[int]:
    tokens = words(text)[:limit_tokens]
    ids = [zlib.crc32(tok.encode()) for tok in tokens]
    if len(ids) < n:
        return {zlib.crc32(normalize_for_dedup(text).encode())} if text.strip() else set()
    out: set[int] = set()
    for i in range(len(ids) - n + 1):
        chunk = ids[i : i + n]
        value = 0
        for cid in chunk:
            value = (value * 1000003 + cid) % _MERSENNE
        out.add(value)
    return out


class MinHashDeduper:
    """LSH-banded MinHash for near-duplicate detection on the instruction text."""

    def __init__(self, *, num_perm: int = 64, bands: int = 8, threshold: float = 0.85, ngram: int = 5):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = max(1, num_perm // bands)
        self.num_perm = self.rows * self.bands
        self.threshold = threshold
        self.ngram = ngram
        # deterministic permutation coefficients
        self._coeffs = [
            ((i * 2654435761) % _MERSENNE or 1, (i * 40503 + 12345) % _MERSENNE) for i in range(self.num_perm)
        ]
        self._buckets: dict[tuple[int, int], list[str]] = {}
        self._grams: dict[str, set[int]] = {}
        self._exact: dict[str, str] = {}
        self.duplicates = 0

    def signature(self, grams: set[int]) -> tuple[int, ...]:
        if not grams:
            return tuple([0] * self.num_perm)
        sig = []
        for a, b in self._coeffs:
            sig.append(min(((a * g + b) % _MERSENNE) for g in grams))
        return tuple(sig)

    def find_duplicate(self, key: str, text: str) -> str | None:
        """Return the id of an existing near-duplicate, or ``None`` (and register ``key``)."""
        normalized = normalize_for_dedup(text)
        if not normalized:
            return None

        exact = self._exact.get(normalized)
        if exact is not None:
            self.duplicates += 1
            return exact

        grams = _ngram_ids(text, self.ngram)
        signature = self.signature(grams)
        candidates: set[str] = set()
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows]
            bucket_key = (band, hash(chunk))
            candidates.update(self._buckets.get(bucket_key, ()))

        for other in candidates:
            other_grams = self._grams.get(other)
            if not other_grams:
                continue
            if _jaccard(grams, other_grams) >= self.threshold:
                self.duplicates += 1
                return other

        # register
        self._exact[normalized] = key
        self._grams[key] = grams
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows]
            self._buckets.setdefault((band, hash(chunk)), []).append(key)
        return None


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


# ---------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------


def instruction_skeleton(record: Record) -> str | None:
    """Instruction text with the seed's topic words removed.

    Synthetic generators reuse the same question template across topics ("What should
    someone know about X?"). Word-shingle overlap cannot see that, because the only
    differing token *is* the topic. Removing the topic leaves the template behind, so two
    examples built from the same template collapse onto one key. Returns ``None`` when the
    topic is unknown or the leftover skeleton is too short to be meaningful.
    """
    topic = str((record.meta or {}).get("topic") or "").strip()
    instruction = normalize_for_dedup(record.instruction)
    if not topic or not instruction:
        return None
    topic_words = set(words(topic))
    skeleton = [w for w in instruction.split() if w not in topic_words]
    if len(skeleton) < 4:
        return None
    joined = " ".join(skeleton)
    # only useful when the topic actually appeared in the instruction
    return joined if joined != instruction else None


def deduplicate(records: Sequence[Record], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Reject exact duplicates, near-duplicates and template duplicates. Returns stats."""
    c = dict(cfg.get("dedup") or {})
    if not c.get("enabled", True):
        return {"enabled": False, "duplicates": 0}

    deduper = MinHashDeduper(
        threshold=float(c.get("threshold", 0.85)),
        ngram=int(c.get("ngram", 5)),
    )
    near = bool(c.get("near", True))
    template = bool(c.get("template", True))
    near_duplicates = 0
    template_duplicates = 0
    seen_skeletons: dict[str, str] = {}

    for record in records:
        if "near_duplicate_instruction" in record.reject_reasons:
            continue
        duplicate_of = deduper.find_duplicate(record.id, record.instruction)
        if duplicate_of is not None and near:
            near_duplicates += 1
            record.reject("near_duplicate_instruction")
            record.filters.append(
                FilterOutcome(
                    "dedup", False, "near_duplicate_instruction", f"duplicate_of={duplicate_of}"
                ).to_dict()
            )
            continue  # no need to also flag the template form

        if template:
            skeleton = instruction_skeleton(record)
            if skeleton:
                first = seen_skeletons.get(skeleton)
                if first is not None:
                    template_duplicates += 1
                    record.reject("template_duplicate_instruction")
                    record.filters.append(
                        FilterOutcome(
                            "dedup",
                            False,
                            "template_duplicate_instruction",
                            f"same template as {first}",
                        ).to_dict()
                    )
                    continue
                seen_skeletons[skeleton] = record.id

    return {
        "enabled": True,
        "duplicates": near_duplicates + template_duplicates,
        "near_duplicates": near_duplicates,
        "template_duplicates": template_duplicates,
        "threshold": deduper.threshold,
        "ngram": deduper.ngram,
    }


def run_filters(record: Record, cfg: Mapping[str, Any]) -> list[FilterOutcome]:
    """Run every per-record filter and record the outcomes on the record."""
    filters_cfg = dict(cfg.get("filters") or {})
    outcomes: list[FilterOutcome] = []
    for name, fn in FILTERS.items():
        section = dict(filters_cfg.get(name) or {})
        if not section.get("enabled", True):
            continue
        outcome = fn(record, section)
        outcomes.append(outcome)
        record.filters.append(outcome.to_dict())
        if not outcome.ok and outcome.reason:
            record.reject(outcome.reason)
    return outcomes


def apply_filters(records: Sequence[Record], cfg: Mapping[str, Any]) -> tuple[list[Record], dict[str, Any]]:
    """Run all filters (including cross-record dedup) and finalise each record's status."""
    filters_cfg = dict(cfg.get("filters") or {})
    for record in records:
        run_filters(record, cfg)

    dedup_stats = deduplicate(records, filters_cfg)

    accepted = 0
    reason_counts: dict[str, int] = {}
    for record in records:
        if record.status != STATUS_REJECTED:
            record.status = STATUS_ACCEPTED
            accepted += 1
        for reason in record.reject_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    stats = {
        "total": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "reject_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "dedup": dedup_stats,
    }
    return list(records), stats


def rejection_summary(records: Iterable[Record], limit: int = 15) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.reject_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


__all__ = [
    "FILTERS",
    "FilterOutcome",
    "instruction_skeleton",
    "MinHashDeduper",
    "apply_filters",
    "deduplicate",
    "filter_language",
    "filter_length",
    "filter_quality",
    "filter_safety",
    "rejection_summary",
    "run_filters",
]
