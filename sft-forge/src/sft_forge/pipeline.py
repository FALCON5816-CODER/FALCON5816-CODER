"""Stage orchestration: seeds → instructions → responses → judge → filter → export.

Two design commitments make this usable on real budgets:

**Everything is checkpointed.** Each stage writes ``<stage>.jsonl`` incrementally, keeps a
cursor of completed items, and only marks itself ``done`` when it finishes. Kill the
process, fix your prompt, and re-run: completed work is reused and only the missing tail is
regenerated. Combined with the response cache, iterating is cheap.

**Failures are data, not crashes.** A worker that raises writes an error row, is counted,
and does not take down the run. A budget cap stops the run cleanly and leaves it resumable.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import prompts as prompts_module
from .cache import ResponseCache
from .client import Budget, BudgetExceeded, LLMClient
from .config import resolve_run_dir, save_resolved_config
from .export import export_records
from .filters import apply_filters
from .generate import (
    WorkItem,
    instruction_worker_factory,
    records_from_rows,
    response_worker_factory,
)
from .judge import judge_worker_factory
from .providers import build_provider
from .report import write_report
from .schemas import Record, Seed
from .seeds import build_seeds
from .stats import compute_stats
from .utils import (
    atomic_write_text,
    get_logger,
    read_json,
    read_jsonl,
    short_id,
    stable_hash,
    utcnow_iso,
    write_json,
    write_jsonl,
)

STAGES = ("seeds", "instructions", "responses", "judge", "filter", "export")

try:  # tqdm is a hard dependency, but the pipeline should never hard-fail on display
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


#: Yielded by the worker wrapper when a budget cap is hit. The item is *not* written and
#: the cursor is *not* advanced, so a resumed run retries exactly that item.
_BUDGET_STOP = object()


def _progress(iterable: Iterable[Any], total: int, desc: str, enabled: bool) -> Iterable[Any]:
    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit="item", dynamic_ncols=True, leave=False)


# ---------------------------------------------------------------------------------------
# stage fingerprints
# ---------------------------------------------------------------------------------------


def _prompt_fingerprint() -> str:
    """Hash the prompt module, so editing a prompt invalidates the stages that use it."""
    try:
        return short_id(Path(prompts_module.__file__).read_text(encoding="utf-8"), n=10)
    except OSError:  # pragma: no cover - source always exists in practice
        return "unknown"


def _seeds_key(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seeds": cfg.get("seeds"),
        "run_seed": (cfg.get("run") or {}).get("seed"),
        "tasks": cfg.get("tasks"),
    }


def _generation_key(cfg: Mapping[str, Any]) -> dict[str, Any]:
    generation = dict(cfg.get("generation") or {})
    return {
        "generator": (cfg.get("models") or {}).get("generator"),
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "max_tokens": generation.get("max_tokens"),
        "seeds_per_topic": generation.get("seeds_per_topic"),
        "require_json": generation.get("require_json"),
        "system_guidance": generation.get("system_guidance"),
        "extra_body": generation.get("extra_body"),
    }


FINGERPRINT_KEYS = ("seeds", "instructions", "responses", "judge")


def stage_fingerprint(cfg: Mapping[str, Any], stage: str) -> str:
    """Hash of exactly the inputs that determine a stage's output.

    Resuming is only safe when the inputs are unchanged; this is what makes that check
    precise. Editing a prompt, a temperature or the judge threshold recomputes the affected
    stages (and only those); everything else is reused from the checkpoints.
    """
    generation = {
        "generation": _generation_key(cfg),
        "candidates": (cfg.get("generation") or {}).get("response_candidates"),
        "prompts": _prompt_fingerprint(),
    }
    if stage == "seeds":
        payload: Any = _seeds_key(cfg)
    elif stage == "instructions":
        payload = {**_seeds_key(cfg), **generation}
    elif stage == "responses":
        # a stage's output depends on the stage before it, so fingerprints chain
        payload = {"upstream": stage_fingerprint(cfg, "instructions"), **generation}
    elif stage == "judge":
        payload = {
            "upstream": stage_fingerprint(cfg, "responses"),
            "judge": cfg.get("judge"),
            "judge_model": (cfg.get("models") or {}).get("judge"),
            **generation,
        }
    else:  # pragma: no cover - the filter stage is recomputed every run
        payload = {}
    return stable_hash(stage, payload)[:16]


# ---------------------------------------------------------------------------------------
# resumable stage execution
# ---------------------------------------------------------------------------------------


@dataclass
class StageReport:
    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    items_total: int = 0
    items_processed: int = 0
    resumed_items: int = 0
    errors: int = 0
    seconds: float = 0.0
    resumed: bool = False
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "items_total": self.items_total,
            "items_processed": self.items_processed,
            "resumed_items": self.resumed_items,
            "errors": self.errors,
            "seconds": round(self.seconds, 2),
            "resumed_from_checkpoint": self.resumed,
            "stopped_early": self.stopped_early,
            "rows": len(self.rows),
        }


class StageRunner:
    """Runs one pipeline stage with checkpointing, resume, and bounded concurrency."""

    def __init__(self, run_dir: str | Path, name: str, fingerprint: str | None = None):
        self.run_dir = Path(run_dir)
        self.name = name
        self.fingerprint = fingerprint
        self.path = self.run_dir / f"{name}.jsonl"
        self.partial_path = self.run_dir / f"{name}.jsonl.partial"
        self.cursor_path = self.run_dir / f"{name}.cursor.json"
        self.done_path = self.run_dir / f"{name}.done.json"
        self.stopped_early = False

    # -- checkpoint helpers -------------------------------------------------------------

    def is_done(self) -> bool:
        """True when a completed checkpoint exists *and* its inputs still match."""
        if not (self.done_path.exists() and self.path.exists()):
            return False
        if self.fingerprint is None:
            return True
        stored = (read_json(self.done_path, default={}) or {}).get("fingerprint")
        return stored == self.fingerprint

    def mark_done(self, *, items: int, rows: int, errors: int) -> None:
        write_json(
            self.done_path,
            {
                "stage": self.name,
                "items": items,
                "rows": rows,
                "errors": errors,
                "fingerprint": self.fingerprint,
                "finished_at": utcnow_iso(),
            },
        )

    def invalidate(self) -> None:
        """Drop a stale checkpoint whose inputs no longer match the config."""
        for path in (self.done_path, self.cursor_path, self.partial_path):
            path.unlink(missing_ok=True)

    def _load_cursor(self) -> int:
        data = read_json(self.cursor_path, default=None)
        if not isinstance(data, Mapping):
            return 0
        if self.fingerprint is not None and data.get("fingerprint") not in (None, self.fingerprint):
            return 0  # previous partial run used different inputs
        try:
            return max(0, int(data.get("items_done", 0)))
        except (TypeError, ValueError):
            return 0

    def _save_cursor(self, items_done: int) -> None:
        payload = {"items_done": int(items_done), "fingerprint": self.fingerprint}
        atomic_write_text(self.cursor_path, _dump(payload) + "\n")

    # -- main entry point ---------------------------------------------------------------

    def run(
        self,
        items: Sequence[Any],
        worker: Callable[[Any], Any],
        *,
        concurrency: int = 1,
        resume: bool = True,
        show_progress: bool = True,
    ) -> StageReport:
        report = StageReport(name=self.name, items_total=len(items))
        started = time.perf_counter()

        if resume and self.is_done():
            stored = read_json(self.done_path, default={}) or {}
            if stored.get("items") != len(items):
                get_logger().info(
                    "stage %s: checkpoint has %s items but %s are queued, recomputing",
                    self.name,
                    stored.get("items"),
                    len(items),
                )
                self.invalidate()
            else:
                rows = read_jsonl(self.path)
                report.rows = rows
                report.resumed = True
                report.items_processed = len(items)
                report.resumed_items = len(items)
                report.seconds = time.perf_counter() - started
                get_logger().info("stage %s: resumed from checkpoint (%d rows)", self.name, len(rows))
                return report

        if resume and (self.done_path.exists() or self.cursor_path.exists()) and not self.is_done():
            get_logger().info("stage %s: inputs changed, recomputing", self.name)
            self.invalidate()

        done_items = self._load_cursor() if resume else 0
        existing_rows: list[dict[str, Any]] = []
        if done_items and self.partial_path.exists():
            existing_rows = read_jsonl(self.partial_path)
        else:
            done_items = 0
            self.partial_path.unlink(missing_ok=True)

        pending = list(items[done_items:])
        report.items_processed = done_items
        report.resumed_items = done_items
        report.rows = existing_rows
        if done_items:
            get_logger().info(
                "stage %s: resuming at item %d/%d (%d rows already written)",
                self.name,
                done_items,
                len(items),
                len(existing_rows),
            )

        lock = threading.Lock()
        mode = "a" if existing_rows else "w"

        def safe_worker(item: Any) -> Any:
            try:
                return worker(item)
            except BudgetExceeded as exc:
                self.stopped_early = True
                get_logger().warning("stage %s: %s", self.name, exc)
                return _BUDGET_STOP
            except Exception as exc:  # noqa: BLE001 - one bad item must not kill the stage
                with lock:
                    report.errors += 1
                get_logger().warning("stage %s: item failed: %s", self.name, exc)
                return {"_error": f"{type(exc).__name__}: {exc}"}

        written_items = done_items
        if pending:
            with open(self.partial_path, mode, encoding="utf-8") as fh, ThreadPoolExecutor(
                max_workers=max(1, int(concurrency))
            ) as pool:
                    stream = pool.map(safe_worker, pending)
                    for rows in _progress(stream, len(pending), self.name, show_progress):
                        if rows is _BUDGET_STOP:
                            pool.shutdown(wait=False, cancel_futures=True)
                            break
                        batch = rows if isinstance(rows, list) else [rows]
                        for row in batch:
                            fh.write(_dump(row))
                            fh.write("\n")
                        fh.flush()
                        written_items += 1
                        report.items_processed = written_items
                        report.rows.append(batch)
                        self._save_cursor(written_items)

        flat: list[dict[str, Any]] = []
        for entry in report.rows:
            if isinstance(entry, list):
                flat.extend(entry)
            else:  # rows restored from a checkpoint arrive already flattened
                flat.append(entry)

        complete = (not self.stopped_early) and written_items >= len(items)
        report.stopped_early = not complete
        if complete:
            if self.partial_path.exists():
                os.replace(self.partial_path, self.path)
            self.mark_done(items=written_items, rows=len(flat), errors=report.errors)
            self.cursor_path.unlink(missing_ok=True)
        else:
            # Something interrupted the stage (budget cap, or a worker that could not be
            # retried): keep the partial file *and* the cursor so the next run continues here.
            self.stopped_early = True
            get_logger().warning(
                "stage %s incomplete after %d/%d items; checkpoint kept for resume",
                self.name,
                written_items,
                len(items),
            )
        report.rows = flat

        report.seconds = time.perf_counter() - started
        get_logger().info(
            "stage %s: %d items, %d rows, %d errors in %.1fs",
            self.name,
            report.items_processed,
            len(report.rows),
            report.errors,
            report.seconds,
        )
        return report


def _dump(row: Any) -> str:
    import json

    return json.dumps(row, ensure_ascii=False)


def _rows_without_errors(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if "_error" in row:
            continue
        out.append(dict(row))
    return out


# ---------------------------------------------------------------------------------------
# client construction
# ---------------------------------------------------------------------------------------


def build_client(cfg: Mapping[str, Any], run_dir: str | Path, *, transport: Any = None) -> LLMClient:
    """Create the LLM client (provider + cache + budget + rate limit) for a run."""
    provider_cfg = dict(cfg.get("provider") or {})
    provider = build_provider(provider_cfg, transport=transport)
    cache_cfg = dict(cfg.get("cache") or {})
    cache_path = run_dir if cache_cfg.get("enabled", True) else None
    cache = (
        ResponseCache(Path(run_dir) / "cache.sqlite" if cache_path else None, enabled=bool(cache_path))
        if cache_path
        else None
    )
    budget_cfg = dict(cfg.get("budget") or {})
    budget = Budget(
        max_requests=budget_cfg.get("max_requests"),
        max_tokens=budget_cfg.get("max_tokens"),
        max_usd=budget_cfg.get("max_usd"),
    )
    prices = dict((cfg.get("models") or {}).get("prices") or {})
    return LLMClient(
        provider,
        cache=cache,
        rate_limit_per_minute=provider_cfg.get("rate_limit_per_minute"),
        budget=budget,
        prices=prices,
        seed=int((cfg.get("run") or {}).get("seed", 17)),
    )


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------


@dataclass
class RunResult:
    run_dir: Path
    stats: dict[str, Any]
    manifest: dict[str, Any]
    stages: dict[str, Any]
    client_stats: dict[str, Any]
    stopped_early: bool = False
    paths: dict[str, str] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        from .stats import stats_summary_lines

        lines = [f"run directory: {self.run_dir}"]
        lines.extend(stats_summary_lines(self.stats))
        if self.client_stats:
            lines.append(
                f"model calls: {self.client_stats.get('requests', 0)} "
                f"({self.client_stats.get('errors', 0)} errors, cache hits "
                f"{(self.client_stats.get('cache') or {}).get('hits', 0)})"
            )
        if self.stopped_early:
            lines.append("STOPPED EARLY (budget) — re-run to continue from the checkpoint")
        return lines


def run_pipeline(
    cfg: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
    resume: bool | None = None,
    show_progress: bool = True,
    client: LLMClient | None = None,
    transport: Any = None,
) -> RunResult:
    """Execute the full workflow and write every artifact into ``run_dir``."""
    log = get_logger()
    started = time.perf_counter()
    run_path = resolve_run_dir(cfg, str(run_dir) if run_dir else None)
    resume = bool((cfg.get("run") or {}).get("resume", True)) if resume is None else resume
    save_resolved_config(cfg, run_path)

    owns_client = client is None
    active_client = client or build_client(cfg, run_path, transport=transport)
    concurrency = int((cfg.get("generation") or {}).get("concurrency", 8))
    timings: dict[str, float] = {}
    stages: dict[str, Any] = {}

    log.info("run directory: %s", run_path)
    log.info("provider: %s | generator: %s", active_client.provider.name, (cfg.get("models") or {}).get("generator"))

    try:
        # -- stage 1: seeds -------------------------------------------------------------
        t0 = time.perf_counter()
        seeds_runner = StageRunner(run_path, "seeds", fingerprint=stage_fingerprint(cfg, "seeds"))
        if resume and seeds_runner.is_done():
            seed_rows = read_jsonl(seeds_runner.path)
            stages["seeds"] = {"rows": len(seed_rows), "resumed_from_checkpoint": True}
        else:
            seeds = build_seeds(cfg, active_client)
            seed_rows = [s.to_dict() for s in seeds]
            write_jsonl(seeds_runner.path, seed_rows)
            seeds_runner.mark_done(items=len(seed_rows), rows=len(seed_rows), errors=0)
            stages["seeds"] = {"rows": len(seed_rows), "resumed_from_checkpoint": False}
        seeds = [Seed.from_dict(row) for row in seed_rows]
        timings["seeds"] = time.perf_counter() - t0

        if not seeds:
            raise ValueError(
                "no seeds produced — check seeds.source/domains/limit in the config "
                "(or whether every seed was filtered out)"
            )

        # -- stage 2: instructions ------------------------------------------------------
        t0 = time.perf_counter()
        instructions = StageRunner(
            run_path, "instructions", fingerprint=stage_fingerprint(cfg, "instructions")
        ).run(
            seeds,
            instruction_worker_factory(active_client, cfg),
            concurrency=concurrency,
            resume=resume,
            show_progress=show_progress,
        )
        timings["instructions"] = time.perf_counter() - t0
        stages["instructions"] = instructions.to_dict()
        instruction_rows = _rows_without_errors(instructions.rows)

        # -- stage 3: responses ---------------------------------------------------------
        t0 = time.perf_counter()
        work_items = [WorkItem.from_row(row) for row in instruction_rows]
        responses = StageRunner(
            run_path, "responses", fingerprint=stage_fingerprint(cfg, "responses")
        ).run(
            work_items,
            response_worker_factory(active_client, cfg),
            concurrency=concurrency,
            resume=resume,
            show_progress=show_progress,
        )
        timings["responses"] = time.perf_counter() - t0
        stages["responses"] = responses.to_dict()
        response_rows = _rows_without_errors(responses.rows)

        # -- stage 4: judging -----------------------------------------------------------
        t0 = time.perf_counter()
        if (cfg.get("judge") or {}).get("enabled", True):
            judged = StageRunner(
                run_path, "judge", fingerprint=stage_fingerprint(cfg, "judge")
            ).run(
                response_rows,
                judge_worker_factory(active_client, cfg),
                concurrency=max(1, min(concurrency, 8)),
                resume=resume,
                show_progress=show_progress,
            )
            judged_rows = _rows_without_errors(judged.rows)
            stages["judge"] = judged.to_dict()
        else:
            judged_rows = response_rows
            stages["judge"] = {"enabled": False, "rows": len(judged_rows)}
            (run_path / "judge.jsonl").unlink(missing_ok=True)
            (run_path / "judge.done.json").unlink(missing_ok=True)
        timings["judge"] = time.perf_counter() - t0

        # -- stage 5: filters -----------------------------------------------------------
        t0 = time.perf_counter()
        records = records_from_rows(judged_rows)
        records, filter_stats = apply_filters(records, cfg)
        write_jsonl(run_path / "filtered.jsonl", (r.to_dict() for r in records))
        write_json(
            run_path / "filter_stats.json",
            {**filter_stats, "at": utcnow_iso()},
        )
        stages["filter"] = filter_stats
        timings["filter"] = time.perf_counter() - t0

        # -- stage 6: stats, report, export ---------------------------------------------
        t0 = time.perf_counter()
        client_stats = active_client.stats()
        stats = compute_stats(records, extra=client_stats)
        write_json(run_path / "stats.json", stats)

        manifest = export_records(
            records,
            cfg,
            run_path,
            extra_manifest={
                "stages": stages,
                "timings_s": {k: round(v, 2) for k, v in timings.items()},
                "client": client_stats,
                "provider": {"kind": (cfg.get("provider") or {}).get("kind")},
                "wall_clock_s": round(time.perf_counter() - started, 2),
            },
        )
        write_report(
            run_path,
            records,
            cfg,
            manifest=manifest,
            client_stats=client_stats,
            timings=timings,
        )
        timings["export"] = time.perf_counter() - t0
        stages["export"] = {"rows": manifest["counts"]["accepted"], "formats": manifest["formats"]}

        result = RunResult(
            run_dir=run_path,
            stats=stats,
            manifest=manifest,
            stages=stages,
            client_stats=client_stats,
            stopped_early=any(
                stage.get("stopped_early") for stage in stages.values() if isinstance(stage, Mapping)
            ),
            paths=dict(manifest.get("paths") or {}),
        )
        write_json(run_path / "run_summary.json", {
            "run_dir": str(run_path),
            "summary": result.summary_lines(),
            "stages": stages,
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "finished_at": utcnow_iso(),
        })
        log.info("done in %.1fs → %s", time.perf_counter() - started, run_path)
        return result
    finally:
        if owns_client:
            active_client.close()


def preview(
    cfg: Mapping[str, Any],
    *,
    count: int = 3,
    client: LLMClient | None = None,
    transport: Any = None,
) -> list[Record]:
    """Generate a small unjudged sample without writing artifacts (prompt iteration)."""
    import tempfile

    owns_client = client is None
    with tempfile.TemporaryDirectory() as tmp:
        active_client = client or build_client(cfg, tmp, transport=transport)
        try:
            seeds = build_seeds(cfg, active_client)
            subset = seeds[: max(1, count)]
            instruction_worker = instruction_worker_factory(active_client, cfg)
            response_worker = response_worker_factory(active_client, cfg)
            records: list[Record] = []
            for seed in subset:
                for row in instruction_worker(seed)[:1]:
                    item = WorkItem.from_row(row)
                    records.append(Record.from_dict(response_worker(item)))
            return records
        finally:
            if owns_client:
                active_client.close()


__all__ = [
    "STAGES",
    "LLMClient",
    "RunResult",
    "StageReport",
    "StageRunner",
    "build_client",
    "preview",
    "run_pipeline",
]
