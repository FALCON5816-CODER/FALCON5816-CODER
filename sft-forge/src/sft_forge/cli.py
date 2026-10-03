"""Command line interface.

    sft-forge init my-dataset        # scaffold a config
    sft-forge cost -c config.yaml    # what will this run cost?
    sft-forge preview -c config.yaml # eyeball generated examples first
    sft-forge run -c config.yaml     # generate the dataset
    sft-forge stats runs/my-run      # inspect the result
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, load_config, resolve_run_dir
from .utils import get_logger, load_env_file, read_json, read_jsonl, setup_logging

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

STARTER_CONFIG = """\
# sft-forge run configuration — see README.md for every option.
run:
  name: {name}
  output_dir: runs
  seed: 17
  resume: true

provider:
  kind: openai_compat            # or 'mock' to run offline with no API key
  base_url: ${{OPENAI_BASE_URL:-https://api.openai.com/v1}}
  api_key: ${{OPENAI_API_KEY}}
  timeout_s: 120
  max_retries: 5
  rate_limit_per_minute: 300     # safe default; raise if your tier allows

models:
  generator: ${{SFT_FORGE_GENERATOR_MODEL:-gpt-4o-mini}}   # writes the answers
  judge: ${{SFT_FORGE_JUDGE_MODEL:-gpt-4o-mini}}           # scores them
  prices: {{}}                    # optional: {{my-model: {{in: 0.2, out: 0.8}}}} per 1M tokens

generation:
  temperature: 0.9
  top_p: 0.95
  max_tokens: 1200
  concurrency: 8
  seeds_per_topic: 6             # instructions written per seed
  response_candidates: 1         # >1 = best-of-n (costs more, judged picks the winner)

seeds:
  source: taxonomy               # taxonomy | file | inline
  file: ""
  inline: []
  domains: []                    # e.g. [technology & software, personal finance]
  limit: null                    # set to 20 for a cheap first run
  shuffle: true
  expansion:
    enabled: false               # let the model invent extra topics
    per_domain: 8

tasks:
  types: [open_qa, explanation, howto, analysis, brainstorming, planning,
          summarization, rewriting, classification, extraction, debugging, coding]
  difficulties: [easy, medium, hard]

judge:
  enabled: true
  sample_rate: 1.0               # 0.25 judges a deterministic quarter of the data
  min_overall: 4.0               # accept threshold on the 1-5 rubric mean
  on_error: reject               # reject | keep, when the judge reply is unparseable
  rewrite:
    enabled: true                # salvage near-miss answers instead of dropping them
    max_attempts: 1
    min_overall: 3.5

filters:
  dedup:
    enabled: true
    near: true
    threshold: 0.85
  length:
    min_response_words: 12
    max_response_words: 1200
  quality: {{}}
  language: {{}}
  safety:
    blocklist: ["social security number", "credit card number"]

budget:
  max_requests: null             # hard stops — the run checkpoints and stays resumable
  max_tokens: null
  max_usd: null

export:
  formats: [chatml, alpaca]      # chatml | alpaca | sharegpt | pairs
  include_scores: true
  splits: null                   # e.g. {{train: 0.9, test: 0.1}}
  write_rejected: true
"""


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _print(message: str = "") -> None:
    print(message)


def _fmt_money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def _load_cfg(args: argparse.Namespace, *, require_api_key: bool = True) -> dict[str, Any]:
    overrides = list(getattr(args, "set", None) or [])
    if getattr(args, "limit", None):
        overrides.append(f"seeds.limit={int(args.limit)}")
    if getattr(args, "run_name", None):
        overrides.append(f"run.name={args.run_name}")
    config_path = getattr(args, "config", None)
    if config_path is None and Path("config.yaml").exists():
        config_path = "config.yaml"
    return load_config(config_path, overrides, require_api_key=require_api_key)


def _attach_run_log(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "pipeline.log"
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path.resolve():
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return path


def _records_from_run(run_dir: Path) -> list[dict[str, Any]]:
    for name in ("filtered.jsonl", "final.jsonl", "judged.jsonl"):
        path = run_dir / name
        if path.exists():
            return read_jsonl(path)
    raise FileNotFoundError(f"no dataset found in {run_dir} (looked for filtered.jsonl/final.jsonl)")


# ---------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.directory or ".").expanduser()
    target.mkdir(parents=True, exist_ok=True)
    name = args.name or target.resolve().name or "sft-run"

    config_path = target / "config.yaml"
    if config_path.exists() and not args.force:
        _print(f"refusing to overwrite existing {config_path} (use --force)")
        return EXIT_ERROR
    config_path.write_text(STARTER_CONFIG.format(name=name), encoding="utf-8")

    env_path = target / ".env.example"
    if not env_path.exists():
        env_path.write_text(
            "OPENAI_API_KEY=sk-replace-me\n"
            "# OPENAI_BASE_URL=https://api.openai.com/v1\n"
            "# SFT_FORGE_GENERATOR_MODEL=gpt-4o-mini\n"
            "# SFT_FORGE_JUDGE_MODEL=gpt-4o-mini\n",
            encoding="utf-8",
        )

    seeds_path = target / "seeds.example.jsonl"
    if not seeds_path.exists():
        seeds_path.write_text(
            json.dumps(
                {
                    "topic": "explaining database indexes",
                    "domain": "technology & software",
                    "task_type": "explanation",
                    "difficulty": "medium",
                    "audience": "a junior developer",
                    "constraints": ["include one concrete example"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    _print(f"created {config_path}")
    _print(f"created {env_path}   (copy to .env and add your key)")
    _print(f"created {seeds_path}")
    _print("")
    _print("next steps:")
    _print(f"  1. cp {env_path} {target / '.env'} && edit it")
    _print(f"  2. sft-forge cost -c {config_path}      # estimate spend")
    _print(f"  3. sft-forge preview -c {config_path}   # see 3 sample examples")
    _print(f"  4. sft-forge run -c {config_path}       # generate the dataset")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = _load_cfg(args)
    except ConfigError as exc:
        _print(f"invalid config: {exc}")
        return EXIT_USAGE
    _print("config is valid ✓")
    if args.json:
        print(json.dumps(cfg, indent=2, default=str))
    return EXIT_OK


def cmd_cost(args: argparse.Namespace) -> int:
    from .estimate import estimate_run, render_estimate

    try:
        cfg = _load_cfg(args, require_api_key=False)
    except ConfigError as exc:
        _print(f"invalid config: {exc}")
        return EXIT_USAGE
    estimate = estimate_run(cfg)
    if args.json:
        print(json.dumps(estimate, indent=2, default=str))
    else:
        _print(render_estimate(estimate))
    return EXIT_OK


def cmd_preview(args: argparse.Namespace) -> int:
    from .pipeline import preview

    try:
        cfg = _load_cfg(args)
    except ConfigError as exc:
        _print(f"invalid config: {exc}")
        return EXIT_USAGE

    records = preview(cfg, count=args.count)
    if not records:
        _print("no examples generated — check the seeds section of your config")
        return EXIT_ERROR
    if args.json:
        print(json.dumps([r.to_dict(include_messages=True) for r in records], indent=2, ensure_ascii=False))
        return EXIT_OK
    for i, record in enumerate(records, start=1):
        _print("=" * 88)
        _print(f"[{i}] {record.domain} / {record.task_type} / {record.difficulty}")
        _print("-" * 88)
        _print(f"INSTRUCTION:\n{record.instruction.strip()}")
        _print("")
        _print(f"RESPONSE:\n{record.response.strip()[:1400]}")
        _print("")
    _print("=" * 88)
    _print("If the instructions or answers are not what you want, edit")
    _print("src/sft_forge/prompts.py (or generation.system_guidance) and preview again.")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    quiet = bool(getattr(args, "quiet", False))

    try:
        cfg = _load_cfg(args)
    except ConfigError as exc:
        _print(f"invalid config: {exc}")
        return EXIT_USAGE

    run_dir = resolve_run_dir(cfg, args.run_dir, create=True)
    _attach_run_log(run_dir)
    if not quiet:
        _print(f"→ run: {cfg['run']['name']}  ({run_dir})")
        _print(f"  generator: {cfg['models']['generator']}   judge: {cfg['judge'].get('model') or cfg['models'].get('judge')}")
        _print(f"  seeds: {cfg['seeds']['source']}   instructions/seed: {cfg['generation']['seeds_per_topic']}")

    try:
        result = run_pipeline(
            cfg,
            run_dir=run_dir,
            resume=None if args.resume else False,
            show_progress=not quiet,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _print("\ninterrupted — progress is checkpointed; re-run to continue")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        get_logger().exception("run failed")
        _print(f"run failed: {exc}")
        _print(f"see {run_dir / 'pipeline.log'} for details")
        return EXIT_ERROR

    if not quiet:
        _print("")
        for line in result.summary_lines():
            _print(f"  {line}")
        _print("")
    if args.json:
        print(json.dumps({"manifest": result.manifest, "stats": result.stats}, indent=2, default=str))
    elif not quiet:
        _print(f"artifacts in {run_dir}:")
        for label, path in sorted(result.paths.items()):
            _print(f"  {label:<22} {path}")
        _print("")
        _print(f"open {run_dir / 'report.md'} for the full breakdown")
    return EXIT_OK


def cmd_stats(args: argparse.Namespace) -> int:
    from .report import render_report
    from .schemas import Record
    from .stats import compute_stats, stats_summary_lines

    run_dir = Path(args.run_dir).expanduser()
    rows = _records_from_run(run_dir)
    records = [Record.from_dict(row) for row in rows]
    cfg = read_json(run_dir / "run_config.json", default={}) or {}
    stats = compute_stats(records)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return EXIT_OK
    if args.report:
        print(render_report(records, cfg))
        return EXIT_OK
    for line in stats_summary_lines(stats):
        _print(line)
    _print("")
    coverage = stats["coverage"]
    _print("domains:")
    for name, count in list(coverage["domain_distribution"].items())[:15]:
        _print(f"  {count:>5}  {name}")
    reasons = stats["rejections"]["reasons"]
    if reasons:
        _print("")
        _print("dropped examples:")
        for reason, count in list(reasons.items())[:10]:
            _print(f"  {count:>5}  {reason}")
    return EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    from .schemas import Record

    run_dir = Path(args.run_dir).expanduser()
    rows = _records_from_run(run_dir)
    records = [Record.from_dict(row) for row in rows]
    fmt = args.format
    out_path = Path(args.out).expanduser() if args.out else run_dir / f"dataset_{fmt}.jsonl"

    from .utils import write_jsonl

    kept = [r for r in records if r.is_accepted]
    count = write_jsonl(out_path, (r.export_view(fmt, include_scores=not args.no_metadata) for r in kept))
    _print(f"wrote {count} accepted examples → {out_path} ({fmt})")
    if args.include_rejected:
        rejected = [r for r in records if not r.is_accepted]
        rejected_path = out_path.with_name(out_path.stem + "_rejected.jsonl")
        n = write_jsonl(rejected_path, (r.export_view(fmt, include_scores=True) for r in rejected))
        _print(f"wrote {n} rejected examples → {rejected_path}")
    return EXIT_OK


def cmd_push(args: argparse.Namespace) -> int:
    from .export import push_to_hub

    run_dir = Path(args.run_dir).expanduser()
    try:
        repo_id = push_to_hub(
            run_dir,
            args.repo_id,
            formats=args.formats,
            split=args.split,
            private=not args.public,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        )
    except Exception as exc:  # noqa: BLE001
        _print(f"upload failed: {exc}")
        return EXIT_ERROR
    _print(f"pushed to https://huggingface.co/datasets/{repo_id}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------


def _add_logging_flags(parser: argparse.ArgumentParser) -> None:
    """Global flags. ``SUPPRESS`` lets a subcommand win only when it is actually passed."""
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="debug logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help="only warnings and errors"
    )
    parser.add_argument(
        "--env-file",
        default=argparse.SUPPRESS,
        help="path to a .env file (default: ./.env)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sft-forge",
        description="Generate high-quality SFT training data with an OpenAI-compatible API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  sft-forge init my-data && cd my-data\n"
            "  sft-forge cost -c config.yaml\n"
            "  sft-forge preview -c config.yaml --count 3\n"
            "  sft-forge run -c config.yaml --limit 20\n"
            "  sft-forge run -c config.yaml --set provider.kind=mock   # offline demo\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"sft-forge {__version__}")
    _add_logging_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    # Global flags are also accepted after the subcommand (`sft-forge run -c cfg -q`).
    common = argparse.ArgumentParser(add_help=False)
    _add_logging_flags(common)

    def _with_common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Re-attach the global flags to a subparser (SUPPRESS keeps the top-level value)."""
        for action in common._actions:  # noqa: SLF001 - argparse exposes no public API for this
            p._add_action(action)  # noqa: SLF001
        return p

    def add_config_flags(p: argparse.ArgumentParser, *, required: bool = False) -> None:
        p.add_argument("-c", "--config", required=required, help="path to the run config YAML")
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override any config value, e.g. --set generation.concurrency=16",
        )

    p_init = _with_common(sub.add_parser("init", help="scaffold a config, .env.example and seeds example"))
    p_init.add_argument("directory", nargs="?", default=".", help="where to create files (default: .)")
    p_init.add_argument("--name", default=None, help="run name (default: directory name)")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing config.yaml")
    p_init.set_defaults(func=cmd_init)

    p_validate = _with_common(sub.add_parser("validate", help="check a config without running anything"))
    add_config_flags(p_validate)
    p_validate.add_argument("--json", action="store_true", help="print the resolved config")
    p_validate.set_defaults(func=cmd_validate)

    p_cost = _with_common(sub.add_parser("cost", help="estimate requests, tokens and spend before running"))
    add_config_flags(p_cost)
    p_cost.add_argument("--json", action="store_true")
    p_cost.set_defaults(func=cmd_cost)

    p_preview = _with_common(sub.add_parser("preview", help="generate a few examples and print them"))
    add_config_flags(p_preview)
    p_preview.add_argument("-n", "--count", type=int, default=3, help="how many seeds to sample")
    p_preview.add_argument("--json", action="store_true")
    p_preview.set_defaults(func=cmd_preview)

    p_run = _with_common(sub.add_parser("run", help="run the full generation workflow"))
    add_config_flags(p_run)
    p_run.add_argument("--run-dir", default=None, help="explicit output directory")
    p_run.add_argument("--run-name", default=None, help="override run.name")
    p_run.add_argument("--limit", type=int, default=None, help="only use the first N seeds")
    p_run.add_argument("--no-resume", dest="resume", action="store_false", help="ignore checkpoints")
    p_run.add_argument("--json", action="store_true", help="print the manifest and stats as JSON")
    p_run.set_defaults(func=cmd_run, resume=True)

    p_stats = _with_common(sub.add_parser("stats", help="summarise a completed run"))
    p_stats.add_argument("run_dir", help="run directory, e.g. runs/general-sft")
    p_stats.add_argument("--json", action="store_true")
    p_stats.add_argument("--report", action="store_true", help="print the full markdown report")
    p_stats.set_defaults(func=cmd_stats)

    p_export = _with_common(sub.add_parser("export", help="re-export a run in another format"))
    p_export.add_argument("run_dir")
    p_export.add_argument("--format", default="chatml", choices=["chatml", "alpaca", "sharegpt", "pairs"])
    p_export.add_argument("--out", default=None, help="output path (default: <run_dir>/dataset_<format>.jsonl)")
    p_export.add_argument("--no-metadata", action="store_true", help="drop the metadata block")
    p_export.add_argument("--include-rejected", action="store_true", help="also write rejected examples")
    p_export.set_defaults(func=cmd_export)

    p_push = _with_common(sub.add_parser("push", help="upload a run to the Hugging Face Hub"))
    p_push.add_argument("run_dir")
    p_push.add_argument("--repo-id", required=True, help="e.g. yourname/my-sft-dataset")
    p_push.add_argument("--formats", nargs="+", default=["chatml"])
    p_push.add_argument("--split", default="train")
    p_push.add_argument("--public", action="store_true", help="default is private")
    p_push.set_defaults(func=cmd_push)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))
    load_env_file(getattr(args, "env_file", None))

    try:
        return int(args.func(args))
    except ConfigError as exc:
        _print(f"config error: {exc}")
        return EXIT_USAGE
    except FileNotFoundError as exc:
        _print(f"not found: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        _print("interrupted")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
