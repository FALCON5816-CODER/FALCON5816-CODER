"""Stage 6: write the training artifacts.

Outputs per run directory:

* ``final.jsonl`` — the accepted dataset in the configured format(s) + metadata,
* ``dataset_<format>.jsonl`` — one file per requested training format,
* ``rejected.jsonl`` — everything dropped, with the reasons (catnip for tuning),
* ``manifest.json`` — provenance: config hash, models, counts, spend, timings.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record
from .utils import get_logger, stable_hash, utcnow_iso, write_json, write_jsonl

TRUTHY = {"1", "true", "yes", "on"}


def _normalise_record(row: Any) -> Record:
    return row if isinstance(row, Record) else Record.from_dict(row)


def export_records(
    records: Sequence[Record | Mapping[str, Any]],
    cfg: Mapping[str, Any],
    run_dir: str | Path,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every artifact for a finished run. Returns the manifest."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    export_cfg = dict(cfg.get("export") or {})
    formats: list[str] = [str(f) for f in (export_cfg.get("formats") or ["chatml"])]
    include_scores = bool(export_cfg.get("include_scores", True))
    include_system = bool(export_cfg.get("include_system", True))

    parsed = [_normalise_record(r) for r in records]
    accepted = [r for r in parsed if r.status == STATUS_ACCEPTED]
    rejected = [r for r in parsed if r.status == STATUS_REJECTED]

    paths: dict[str, str] = {}

    # Canonical accepted dataset in the first configured format
    primary = formats[0]
    _write_dataset(run_path / "final.jsonl", accepted, primary, include_scores, include_system)
    paths["final"] = str(run_path / "final.jsonl")

    for fmt in formats:
        target = run_path / f"dataset_{fmt}.jsonl"
        _write_dataset(target, accepted, fmt, include_scores, include_system)
        paths[f"dataset_{fmt}"] = str(target)

    if export_cfg.get("write_rejected", True):
        rejected_path = run_path / "rejected.jsonl"
        count = write_jsonl(
            rejected_path,
            (
                {
                    "id": r.id,
                    "seed_id": r.seed_id,
                    "domain": r.domain,
                    "task_type": r.task_type,
                    "difficulty": r.difficulty,
                    "instruction": r.instruction,
                    "response": r.response,
                    "reject_reasons": r.reject_reasons,
                    "scores": r.scores,
                    "filters": r.filters,
                    "meta": r.meta,
                }
                for r in rejected
            ),
        )
        paths["rejected"] = str(rejected_path)
        rejected_count = count
    else:
        rejected_count = len(rejected)

    split_paths = _maybe_write_splits(run_path, accepted, formats, cfg, include_scores, include_system)
    paths.update(split_paths)

    manifest: dict[str, Any] = {
        "generated_at": utcnow_iso(),
        "run_name": (cfg.get("run") or {}).get("name"),
        "config_hash": stable_hash(
            {
                k: v
                for k, v in cfg.items()
                if k not in ("_meta", "provider")  # provider holds secrets + transport details
            }
        )[:16],
        "dataset_fingerprint": stable_hash([r.id for r in accepted])[:16],
        "schema_version": accepted[0].schema_version if accepted else None,
        "formats": formats,
        "counts": {"accepted": len(accepted), "rejected": rejected_count, "total": len(parsed)},
        "models": {
            "generator": (cfg.get("models") or {}).get("generator"),
            "judge": (cfg.get("judge") or {}).get("model") or (cfg.get("models") or {}).get("judge"),
        },
        "paths": paths,
        "notes": (
            "Synthetic SFT data. Review before training: synthetic answers can contain "
            "plausible-but-wrong claims. Keep the judge and filter configuration with the "
            "dataset so any future run is reproducible."
        ),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_json(run_path / "manifest.json", manifest)
    paths["manifest"] = str(run_path / "manifest.json")
    manifest["paths"] = paths
    return manifest


def _write_dataset(
    path: Path,
    records: Sequence[Record],
    fmt: str,
    include_scores: bool,
    include_system: bool = True,
) -> int:
    return write_jsonl(
        path,
        (r.export_view(fmt, include_scores=include_scores, include_system=include_system) for r in records),
    )


def _maybe_write_splits(
    run_path: Path,
    accepted: Sequence[Record],
    formats: Sequence[str],
    cfg: Mapping[str, Any],
    include_scores: bool,
    include_system: bool = True,
) -> dict[str, str]:
    splits = (cfg.get("export") or {}).get("splits")
    if not splits:
        return {}
    order = list(splits.items())
    total = sum(float(v) for _, v in order)
    if total <= 0:
        return {}

    rng = random.Random(int((cfg.get("run") or {}).get("seed", 17)))
    shuffled = list(accepted)
    rng.shuffle(shuffled)

    assignments: list[tuple[str, list[Record]]] = []
    cursor = 0
    for i, (name, weight) in enumerate(order):
        if i == len(order) - 1:
            chunk = shuffled[cursor:]
        else:
            size = int(round(len(shuffled) * float(weight) / total))
            chunk = shuffled[cursor : cursor + size]
            cursor += size
        assignments.append((str(name), chunk))

    paths: dict[str, str] = {}
    index: dict[str, int] = {}
    for name, chunk in assignments:
        index[name] = len(chunk)
        for fmt in formats:
            target = run_path / f"{name}_{fmt}.jsonl"
            write_jsonl(
                target,
                (
                    r.export_view(fmt, include_scores=include_scores, include_system=include_system)
                    for r in chunk
                ),
            )
            paths[f"split_{name}_{fmt}"] = str(target)
    write_json(run_path / "splits.json", index)
    paths["splits"] = str(run_path / "splits.json")
    return paths


def push_to_hub(  # pragma: no cover - requires the optional `hub` extra
    run_dir: str | Path,
    repo_id: str,
    *,
    formats: Sequence[str] = ("chatml",),
    split: str = "train",
    private: bool = True,
    token: str | None = None,
) -> str:
    """Optionally publish a run to the Hugging Face Hub (``pip install sft-forge[hub]``)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("publishing requires the hub extra: pip install 'sft-forge[hub]'") from exc

    run_path = Path(run_dir)
    manifest = {}
    try:
        from .utils import read_json

        manifest = read_json(run_path / "manifest.json", default={}) or {}
    except Exception:  # noqa: BLE001
        manifest = {}

    uploaded: list[str] = []
    for fmt in formats:
        pattern = f"dataset_{fmt}.jsonl" if split == "train" else f"{split}_{fmt}.jsonl"
        source = run_path / pattern
        if not source.exists():
            get_logger().warning("skipping upload: %s not found", source)
            continue
        dataset = load_dataset("json", data_files={split: str(source)}, split=split)
        dataset.push_to_hub(repo_id, token=token, private=private)
        uploaded.append(fmt)

    if uploaded:
        from .utils import atomic_write_text

        note = f"https://huggingface.co/datasets/{repo_id}\nformats: {', '.join(uploaded)}\n"
        if manifest:
            note += f"manifest: {manifest.get('dataset_fingerprint', 'n/a')}\n"
        atomic_write_text(run_path / "HUB_UPLOAD.txt", note)
    return repo_id


__all__ = ["export_records", "push_to_hub"]
