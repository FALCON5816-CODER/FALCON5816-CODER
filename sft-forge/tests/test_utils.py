"""Text helpers, hashing, and JSONL IO."""

from __future__ import annotations

import json

import pytest

from sft_forge.utils import (
    RateLimiter,
    append_jsonl,
    atomic_write_text,
    english_ratio,
    load_env_file,
    ngram_repeat_ratio,
    normalize_for_dedup,
    normalize_ws,
    read_jsonl,
    short_id,
    stable_hash,
    unique_token_ratio,
    word_count,
    write_jsonl,
)


def test_hash_and_ids_are_stable_and_sensitive():
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("b", "a")
    assert len(short_id("x")) == 12
    assert short_id("x") != short_id("y")


def test_normalisation():
    assert normalize_ws("  a\n\n b  ") == "a b"
    assert normalize_for_dedup("**Hello**, World!") == "hello world"
    assert normalize_for_dedup("```code``` ignored") == "ignored"


def test_text_metrics():
    assert word_count("one two three") == 3
    assert unique_token_ratio("a a a a") == pytest.approx(0.25)
    assert ngram_repeat_ratio("one two three four one two three four") == pytest.approx(0.2)
    assert ngram_repeat_ratio("loop " * 40) > 0.9
    assert ngram_repeat_ratio("short") == 0.0
    assert english_ratio("the cat is on the mat and it is fine") > 0.3
    assert english_ratio("") == 0.0


def test_jsonl_roundtrip_and_atomicity(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [{"a": 1, "unicode": "café"}, {"b": [1, 2, 3]}]
    assert write_jsonl(path, rows) == 2
    assert read_jsonl(path) == rows
    assert append_jsonl(path, [{"c": 3}]) == 1
    assert len(read_jsonl(path)) == 3
    # the temp file used for atomic writes must not be left behind
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_jsonl_gzip_and_errors(tmp_path):
    import gzip

    path = tmp_path / "out.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"a": 1}) + "\n")
    assert read_jsonl(path) == [{"a": 1}]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl(bad)
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")


def test_atomic_write_text(tmp_path):
    path = tmp_path / "note.txt"
    atomic_write_text(path, "hello")
    assert path.read_text() == "hello"


def test_rate_limiter_disabled_is_a_noop():
    limiter = RateLimiter(None)
    limiter.acquire()  # must not raise or sleep


def test_rate_limiter_throttles():
    import time

    limiter = RateLimiter(rate_per_minute=6000)  # 100/s, capacity 10
    start = time.monotonic()
    for _ in range(15):
        limiter.acquire()
    assert time.monotonic() - start < 1.0


def test_load_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nFOO=bar\nQUOTED="baz"\n\n', encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    assert load_env_file(env) == 2
    import os

    assert os.environ["FOO"] == "bar"
    assert os.environ["QUOTED"] == "baz"
    # existing values win
    monkeypatch.setenv("FOO", "already")
    load_env_file(env)
    assert os.environ["FOO"] == "already"
