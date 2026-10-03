"""Stage 4: LLM-as-judge scoring, best-of-n selection, and one-shot repair.

The judge applies a five-dimension rubric (correctness, helpfulness, clarity,
instruction-following, safety) and returns a JSON verdict. What happens next is policy:

* ``overall >= judge.min_overall`` → the record survives to the filter stage,
* ``judge.rewrite.min_overall <= overall < judge.min_overall`` → one rewrite attempt, then
  re-judge (records we can salvage are worth far more than records we discard),
* below that, or a failed/unparseable judgement with ``judge.on_error: reject`` → dropped.

Judging is the expensive stage, so ``judge.sample_rate`` lets you judge a deterministic
sample of the data instead of everything.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .client import BudgetExceeded, LLMClient
from .jsonx import loads_lenient
from .prompts import judge_messages, rewrite_messages
from .schemas import STATUS_ACCEPTED, STATUS_GENERATED, STATUS_REJECTED, Record
from .utils import get_logger, short_id

JUDGE_DIMENSIONS = ("correctness", "helpfulness", "clarity", "instruction_following", "safety")

#: Keep judge prompts bounded — the tail of a very long answer rarely changes the verdict.
MAX_JUDGED_CHARS = 6000


def parse_judgement(text: str) -> dict[str, Any]:
    """Parse a judge reply into ``{scores, overall, verdict, rationale, issues}``.

    Raises :class:`ValueError` when nothing usable can be recovered.
    """
    payload = loads_lenient(text)
    if not isinstance(payload, Mapping):
        raise ValueError("judge output was not a JSON object")

    raw_scores = payload.get("scores") or {}
    scores: dict[str, float] = {}
    if isinstance(raw_scores, Mapping):
        for key, value in raw_scores.items():
            try:
                scores[str(key)] = max(0.0, min(5.0, float(value)))
            except (TypeError, ValueError):
                continue
    if not scores:
        for dim in JUDGE_DIMENSIONS:
            if dim in payload:
                try:
                    scores[dim] = max(0.0, min(5.0, float(payload[dim])))
                except (TypeError, ValueError):
                    continue
    if not scores:
        raise ValueError("judge output contained no numeric scores")

    overall = payload.get("overall")
    try:
        overall_value = float(overall) if overall is not None else sum(scores.values()) / len(scores)
    except (TypeError, ValueError):
        overall_value = sum(scores.values()) / len(scores)
    overall_value = max(0.0, min(5.0, overall_value))

    issues = payload.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]

    return {
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "overall": round(overall_value, 3),
        "verdict": str(payload.get("verdict") or "").lower() or None,
        "rationale": str(payload.get("rationale") or "").strip(),
        "issues": [str(i).strip() for i in issues if str(i).strip()][:8],
    }


def sampled(record_id: str, sample_rate: float) -> bool:
    """Deterministic sampling so a rerun judges exactly the same records."""
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    bucket = int(short_id(record_id, n=8), 16) / float(16**8)
    return bucket < sample_rate


class Judge:
    """Judge + rewriter bound to a client and a config."""

    def __init__(self, client: LLMClient, cfg: Mapping[str, Any]):
        judge_cfg = dict(cfg.get("judge") or {})
        models = dict(cfg.get("models") or {})
        self.gen_client = client
        self.enabled = bool(judge_cfg.get("enabled", True))
        self.model = str(judge_cfg.get("model") or models.get("judge") or models.get("generator") or "")
        self.temperature = float(judge_cfg.get("temperature", 0.0))
        self.max_tokens = int(judge_cfg.get("max_tokens", 600))
        self.sample_rate = float(judge_cfg.get("sample_rate", 1.0))
        self.min_overall = float(judge_cfg.get("min_overall", 4.0))
        self.on_error = str(judge_cfg.get("on_error", "reject")).lower()
        rewrite_cfg = dict(judge_cfg.get("rewrite") or {})
        self.rewrite_enabled = bool(rewrite_cfg.get("enabled", True))
        self.rewrite_attempts = max(0, int(rewrite_cfg.get("max_attempts", 1)))
        self.rewrite_min_overall = float(rewrite_cfg.get("min_overall", max(3.0, self.min_overall - 1.0)))
        self._lock = threading.Lock()
        self.rejected_for_error = 0

    # -- single record ------------------------------------------------------------------

    def _score_once(self, record: Record, text: str) -> dict[str, Any]:
        completion = self.gen_client.chat(
            judge_messages(record),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            hint="judge",
            seed=0,
        )
        judgement = parse_judgement(completion.text)
        judgement["model"] = self.model
        judgement["usage"] = dict(completion.usage)
        judgement["cached"] = completion.cached
        return judgement

    def score(self, record: Record, text: str | None = None) -> dict[str, Any] | None:
        """Score ``record`` (or the supplied alternative text). Returns ``None`` on failure."""
        original = record.response
        if text is not None:
            record.response = text
        try:
            return self._score_once(record, text or original)
        except BudgetExceeded:
            raise  # a budget cap is a control signal: stop the stage, keep it resumable
        except Exception as exc:  # noqa: BLE001 - judged failure is data, not a crash
            get_logger().warning("judge failed for %s: %s", record.id, exc)
            return None
        finally:
            record.response = original

    def best_of(self, record: Record, candidates: Sequence[str]) -> tuple[str, dict[str, Any] | None]:
        """Score every candidate concurrently and return the winner (ties → earliest)."""
        scored: list[tuple[float, int, str, dict[str, Any] | None]] = []

        def run(index: int, text: str) -> tuple[int, str, dict[str, Any] | None]:
            return index, text, self.score(record, text)

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(candidates)))) as pool:
            for index, text, judgement in pool.map(lambda pair: run(*pair), list(enumerate(candidates))):
                overall = float((judgement or {}).get("overall", -1.0))
                scored.append((overall, index, text, judgement))

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_overall, _, best_text, best_judgement = scored[0]
        record.meta["candidate_scores"] = [
            {"index": idx, "overall": round(score, 3)} for score, idx, _, _ in sorted(scored, key=lambda s: s[1])
        ]
        if best_overall < 0:
            return candidates[0], None
        return best_text, best_judgement

    # -- record pipeline ----------------------------------------------------------------

    def process(self, record: Record) -> Record:
        """Judge (and if needed rewrite) one record, updating it in place."""
        if not self.enabled or not self.model:
            record.status = STATUS_GENERATED
            return record

        if not sampled(record.id, self.sample_rate):
            record.meta["judge_skipped"] = "sampled_out"
            record.status = STATUS_GENERATED
            return record

        candidates = record.meta.get("candidates") or []
        judgement: dict[str, Any] | None = None
        if len(candidates) > 1:
            winner, judgement = self.best_of(record, candidates)
            record.response = winner
        else:
            judgement = self.score(record)

        if judgement is None:
            record.meta["judge_error"] = True
            if self.on_error == "keep":
                record.status = STATUS_GENERATED
            else:
                record.reject("judge_error")
            return record

        record.judge = {k: v for k, v in judgement.items() if k != "usage"}
        record.scores = {**judgement["scores"], "overall": judgement["overall"]}
        attempts = 0

        # Salvage path: one rewrite when the answer is close but not good enough.
        while (
            self.rewrite_enabled
            and attempts < self.rewrite_attempts
            and judgement["overall"] < self.min_overall
            and judgement["overall"] >= self.rewrite_min_overall
        ):
            attempts += 1
            record = self._rewrite(record, attempt=attempts)
            if record.status == STATUS_REJECTED:
                break
            judgement = self.score(record)
            if judgement is None:
                record.meta["judge_error"] = True
                if self.on_error == "keep":
                    break
                record.reject("judge_error")
                return record
            record.judge = {k: v for k, v in judgement.items() if k != "usage"}
            record.scores = {**judgement["scores"], "overall": judgement["overall"]}

        record.meta["judge_attempts"] = attempts
        if record.scores.get("overall", 0.0) >= self.min_overall:
            record.status = STATUS_ACCEPTED
        else:
            record.reject("judge_below_threshold")
        return record

    def _rewrite(self, record: Record, *, attempt: int) -> Record:
        usage = dict(record.usage or {})
        try:
            completion = self.gen_client.chat(
                rewrite_messages(record),
                model=str(record.meta.get("generator_model") or self.model),
                temperature=min(1.0, self.temperature + 0.3),
                max_tokens=1200,
                hint="rewrite",
                seed=attempt,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            get_logger().warning("rewrite failed for %s: %s", record.id, exc)
            record.reject("rewrite_failed")
            return record

        text = (completion.text or "").strip()
        if len(text) < 40:
            record.reject("rewrite_empty")
            return record

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(usage.get(key, 0)) + int(completion.usage.get(key, 0))
        record.meta.setdefault("rejected_draft", str(record.response)[:4000])
        record.response = text
        record.usage = usage
        record.meta["rewritten"] = True
        record.rewrite_of = record.rewrite_of or record.id
        record.id = f"{record.rewrite_of}-r{attempt}"
        record.status = STATUS_GENERATED
        return record


def judge_worker_factory(client: LLMClient, cfg: Mapping[str, Any]):
    """Per-record worker for the judging stage; returns a serialised record row."""
    judge = Judge(client, cfg)

    def worker(row: Mapping[str, Any]) -> dict[str, Any]:
        record = Record.from_dict(row)
        record.status = STATUS_GENERATED if record.status != STATUS_REJECTED else record.status
        processed = judge.process(record)
        return processed.to_dict()

    worker.judge = judge  # type: ignore[attr-defined]  # exposed for tests/reporting
    return worker


__all__ = ["JUDGE_DIMENSIONS", "Judge", "judge_worker_factory", "parse_judgement", "sampled"]
