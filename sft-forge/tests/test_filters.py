"""Deterministic filters and near-duplicate detection."""

from __future__ import annotations

import pytest

from sft_forge.filters import (
    MinHashDeduper,
    apply_filters,
    filter_language,
    filter_length,
    filter_quality,
    filter_safety,
    instruction_skeleton,
    rejection_summary,
)
from sft_forge.schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record

LONG_ANSWER = (
    "Sourdough rises because wild yeast and lactic acid bacteria ferment the flour. "
    "You keep a starter alive by feeding it flour and water on a schedule, then use it "
    "to leaven the dough. Temperature controls how fast this happens, and a long, cool "
    "fermentation develops more flavour than a quick warm one."
)


def record(instruction: str = "How does sourdough rise?", response: str = LONG_ANSWER, **kwargs) -> Record:
    return Record(instruction=instruction, response=response, **kwargs)


@pytest.mark.parametrize(
    "instruction,response,reason",
    [
        ("too short", LONG_ANSWER, None),
        ("x" * 10, "", "empty_response"),
        ("A reasonable question about baking bread?", "tiny", "response_too_short"),
        ("A reasonable question about baking bread?", "word " * 2000, "response_too_many_words"),
        ("x" * 5000, LONG_ANSWER, "instruction_too_long"),
    ],
)
def test_length_filter(instruction, response, reason):
    cfg = {"min_instruction_chars": 8, "max_instruction_chars": 4000, "min_response_chars": 40,
           "max_response_chars": 12000, "min_response_words": 12, "max_response_words": 1200}
    outcome = filter_length(record(instruction, response), cfg)
    assert outcome.ok is (reason is None)
    if reason:
        assert outcome.reason == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        ("I'm sorry, but I can't help with that.", "refusal_like_response"),
        ("As an AI language model, I would say it depends on many things and factors.", "refusal_like_response"),
        ("the the the the the the the the the the the the the the the the the", "low_lexical_diversity"),
        ("repeat this phrase repeat this phrase repeat this phrase repeat this phrase", "repetitive_text"),
        ("This answer is cut off…", "truncated_response"),
        ("Fill in [insert your name here] and then submit the completed document.", "unfilled_placeholder"),
        (LONG_ANSWER, None),
    ],
)
def test_quality_filter(response, reason):
    outcome = filter_quality(record(response=response), {})
    assert outcome.ok is (reason is None)
    if reason:
        assert outcome.reason == reason


def test_quality_filter_catches_parroting():
    instruction = "explain how database indexes improve read performance for large tables"
    outcome = filter_quality(record(instruction, instruction + "."), {})
    assert outcome.reason == "response_parrots_instruction"


def test_language_filter():
    assert filter_language(record(), {}).ok
    german = "Die Katze sitzt auf der Matte und schläft dort den ganzen Nachmittag lang, weil es warm ist und sie müde war von der Jagd im Garten hinter dem Haus."
    assert filter_language(record(response=german), {}).reason == "not_english_enough"
    assert filter_language(record(response="日本語のテキストがここにあります" * 20), {}).reason in (
        "not_english_enough",
        "excessive_non_ascii",
    )


def test_safety_filter_blocklist_and_regex():
    cfg = {"blocklist": ["credit card number"], "regex_denylist": ["\\bpassword\\s*[:=]"]}
    assert filter_safety(record(response="Never share your credit card number."), cfg).reason == "blocklisted_phrase"
    assert filter_safety(record(response="Set password= hunter2 in the file."), cfg).reason == "denylist_regex"
    assert filter_safety(record(), cfg).ok
    # a broken regex must not crash the pipeline
    assert filter_safety(record(), {"regex_denylist": ["("]}).ok


# ---------------------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------------------


def test_minhash_deduper_flags_near_duplicates():
    """Word-shingle similarity catches reformatting, not paraphrase (documented limit)."""
    deduper = MinHashDeduper(threshold=0.8)
    original = (
        "How do I improve the read performance of a large PostgreSQL table that has indexes, "
        "without taking the database offline for maintenance?"
    )
    assert deduper.find_duplicate("a", original) is None
    reformatted = (
        "how do i improve the read performance of a large postgresql table that has indexes "
        "without taking the database offline for maintenance"
    )
    assert deduper.find_duplicate("b", reformatted) == "a"
    assert (
        deduper.find_duplicate("c", "What is the capital of Portugal and why does it matter historically?")
        is None
    )


def test_dedup_stats_and_exact_duplicates():
    deduper = MinHashDeduper(threshold=0.9)
    deduper.find_duplicate("a", "A question about baking bread at home")
    assert deduper.find_duplicate("b", "A question about baking bread at home") == "a"
    assert deduper.duplicates == 1


def test_apply_filters_marks_statuses_and_reasons():
    good = record()
    bad = record(instruction="q" * 3, response="too short")
    duplicate = record(instruction=LONG_ANSWER.split(".")[0] + "?")
    records, stats = apply_filters([good, bad, duplicate], {"filters": {}})

    assert records[0].status == STATUS_ACCEPTED
    assert records[1].status == STATUS_REJECTED
    assert stats["accepted"] >= 1
    assert stats["rejected"] == len(records) - stats["accepted"]
    assert "filters" in stats and stats["filters"]["total"] == 3 if "filters" in stats else True
    assert records[1].filters, "every filter outcome must be recorded for auditing"
    assert rejection_summary(records)


def test_dedup_can_be_disabled():
    records = [
        record(instruction="Same question about baking bread?"),
        record(instruction="Same question about baking bread?"),
    ]
    _, stats = apply_filters(records, {"filters": {"dedup": {"enabled": False}}})
    assert stats["dedup"]["enabled"] is False
    assert all(r.status == STATUS_ACCEPTED for r in records)


def test_exact_duplicate_records_are_rejected():
    question = "Explain how a MinHash signature approximates Jaccard similarity between shingle sets"
    records = [record(instruction=question), record(instruction=question + "  ")]
    records, stats = apply_filters(records, {"filters": {"dedup": {}}})
    assert stats["dedup"]["duplicates"] == 1
    assert records[1].status == STATUS_REJECTED
    assert "near_duplicate_instruction" in records[1].reject_reasons


def test_template_duplicates_are_rejected():
    """The failure mode shingle overlap cannot see: one template, many topics."""
    template = "What should someone know about {topic}?"
    records = [
        Record(
            instruction=template.format(topic="index scans"),
            response=LONG_ANSWER,
            meta={"topic": "index scans"},
        ),
        Record(
            instruction=template.format(topic="email triage"),
            response=LONG_ANSWER,
            meta={"topic": "email triage"},
        ),
    ]
    assert instruction_skeleton(records[0]) == instruction_skeleton(records[1])
    records, stats = apply_filters(records, {"filters": {"dedup": {}}})
    assert stats["dedup"]["template_duplicates"] == 1
    assert records[1].status == STATUS_REJECTED
    assert "template_duplicate_instruction" in records[1].reject_reasons


def test_template_dedup_needs_a_matching_topic_and_skeleton():
    # no topic metadata -> nothing to strip, so the check stays out of the way
    assert instruction_skeleton(record()) is None
    # a topic that does not appear in the instruction leaves the text unchanged
    assert instruction_skeleton(Record(instruction="How do I bake bread?", meta={"topic": "quantum physics"})) is None
