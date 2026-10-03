"""Forgiving JSON parsing — the single most common source of pipeline crashes."""

from __future__ import annotations

import pytest

from sft_forge.jsonx import (
    as_list,
    find_all_json_spans,
    find_json_span,
    loads_lenient,
    loads_or_none,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("```json\n{\"a\": 1}\n```", {"a": 1}),
        ('Sure! Here you go:\n{"a": 1}\nHope that helps.', {"a": 1}),
        ('{"a": 1,}', {"a": 1}),  # trailing comma
        ('{"a": "it\u2019s fine"}', {"a": "it’s fine"}),
        ('{"a": True, "b": None}', {"a": True, "b": None}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('[{"x": "}"}]', [{"x": "}"}]),  # brace inside a string
        ('{"nested": {"deep": [1, {"k": "v"}]}}', {"nested": {"deep": [1, {"k": "v"}]}}),
    ],
)
def test_loads_lenient_recovers_payloads(text, expected):
    assert loads_lenient(text) == expected


def test_loads_lenient_raises_on_garbage():
    with pytest.raises(ValueError):
        loads_lenient("no json at all")
    with pytest.raises(ValueError):
        loads_lenient("")
    assert loads_or_none("still not json") is None


def test_find_json_span_handles_nesting_and_strings():
    text = 'prefix {"a": "}{", "b": [1, 2]} suffix'
    start, end = find_json_span(text)
    assert text[start:end] == '{"a": "}{", "b": [1, 2]}'


def test_find_all_json_spans():
    spans = find_all_json_spans('one {"a": 1} two [2, 3] three')
    assert spans == ['{"a": 1}', "[2, 3]"]


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([1, 2], [1, 2]),
        ({"items": [1, 2]}, [1, 2]),
        ({"data": [{"x": 1}]}, [{"x": 1}]),
        ({"instructions": ["a"]}, ["a"]),
        ({"single": "object"}, [{"single": "object"}]),
        (None, []),
        ("nope", []),
    ],
)
def test_as_list_coercions(payload, expected):
    assert as_list(payload) == expected
