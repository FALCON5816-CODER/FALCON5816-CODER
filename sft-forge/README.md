# sft-forge

**A resumable workflow for generating SFT training data for LLMs.**

You point it at a topic taxonomy and an OpenAI-compatible endpoint; it produces a
filtered, de-duplicated, judge-scored instruction dataset in your choice of training
formats — plus a report telling you exactly what it kept, what it dropped, and why.

```
seeds ─▶ instructions ─▶ answers ─▶ LLM judge ─▶ deterministic filters ─▶ dataset
  ▲                                                    │
  └────────────── resumable checkpoints ───────────────┘
```

Unlike a script that calls an API in a `for` loop, this is built for the part that
actually takes time: **iterating**. Prompts change, judges disagree, budgets run out.
Every stage checkpoints, every model call is cached, and every decision is recorded on the
example it applies to, so a re-run costs almost nothing and you can always answer
"why is this example not in my dataset?".

---

## Contents

- [What you get](#what-you-get)
- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Commands](#commands)
- [Quality control](#quality-control)
- [Working with other providers](#working-with-other-providers)
- [Design notes and limits](#design-notes-and-limits)
- [Development](#development)

---

## What you get

| | |
|---|---|
| **6-stage pipeline** | seed selection → instruction writing → answer generation → LLM judging → deterministic filters → export |
| **Resumable** | kill it mid-run; re-running continues from the last completed item (`--no-resume` to start over) |
| **Cached** | identical model calls are served from a local SQLite cache, so prompt iteration is nearly free |
| **Budgeted** | hard caps on requests, tokens and dollars; the run stops cleanly and stays resumable |
| **Judged** | five-dimension rubric (correctness, helpfulness, clarity, instruction-following, safety) with an automatic rewrite pass for near-misses |
| **Filtered** | near-duplicate detection (MinHash + LSH), length, degeneracy, refusal, language and safety filters — each with an auditable reason code |
| **Multiple formats** | `chatml`, `alpaca`, `sharegpt`, `pairs`, optional train/val/test splits, optional HF Hub upload |
| **Cost-aware** | `sft-forge cost` estimates calls, tokens and dollars *before* you spend any |
| **Offline demo** | `provider.kind: mock` runs the whole workflow with no API key, for CI and for wiring up training code |

---

## Install

```bash
git clone https://github.com/FALCON5816-CODER/FALCON5816-CODER.git
cd FALCON5816-CODER/sft-forge
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # add ,hub for Hugging Face uploads
```

Requires Python 3.9+. The only runtime dependencies are `httpx`, `PyYAML` and `tqdm`.

## Quick start

### 1. Offline first (no API key, ~5 seconds)

```bash
sft-forge run -c configs/demo_mock.yaml
```

This exercises every stage against a deterministic mock provider and writes a real
dataset to `runs/demo-mock/`. Read `runs/demo-mock/report.md` to see what a run produces.

### 2. Point it at a real endpoint

```bash
cp .env.example .env          # then put your key in it
sft-forge cost -c configs/general.yaml        # estimate spend
sft-forge preview -c configs/general.yaml     # look at 3 examples
sft-forge run -c configs/general.yaml         # generate the dataset
```

A full default run is 144 seeds × 6 instructions ≈ 860 answers, judged and filtered — on
`gpt-4o-mini` that is roughly **$0.60–1.50**. Start with `--limit 20`:

```bash
sft-forge run -c configs/general.yaml --limit 20
```

### 3. Scaffold your own

```bash
sft-forge init my-dataset && cd my-dataset
$EDITOR config.yaml
sft-forge preview -c config.yaml
sft-forge run -c config.yaml
```

### 4. Train on the result

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="runs/general-sft/dataset_chatml.jsonl", split="train")
# each row: {"messages": [{"role": "system"|"user"|"assistant", "content": ...}], "metadata": {...}}

def to_tokens(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=2048)

tokenized = ds.map(lambda ex: to_tokens(ex, tokenizer), remove_columns=ds.column_names)
```

---

## How it works

| Stage | What happens | Artifact |
|---|---|---|
| 1. **Seeds** | Loads the bundled 144-seed taxonomy (26 domains, 13 task types), your own JSONL, or inline entries. Optionally asks the model for extra topics per domain. | `seeds.jsonl` |
| 2. **Instructions** | One call per seed writes *N* self-contained user requests, each with a rotated style hint and the seed's difficulty/audience/constraints. Unparseable replies fall back to deterministic templates, so no seed is silently lost. | `instructions.jsonl` |
| 3. **Answers** | One call per instruction. With `response_candidates > 1`, several answers are produced so the judge can pick the best. | `responses.jsonl` |
| 4. **Judging** | Five-dimension rubric on a 1–5 scale. Answers below the bar but close to it get one rewrite + re-judge instead of being discarded. `sample_rate` judges only a deterministic sample. | `judged.jsonl` |
| 5. **Filters** | Length, lexical diversity, repetition, instruction-parroting, refusals, placeholders, language drift, blocklists, and near-duplicate removal via MinHash + LSH. | `filtered.jsonl`, `filter_stats.json` |
| 6. **Export** | Training formats, optional splits, rejected-examples file, manifest, report. | `dataset_*.jsonl`, `manifest.json`, `report.md` |

Each stage writes `<stage>.jsonl` incrementally with a cursor file, and only marks itself
`done` when it completes. Interrupt a run — Ctrl-C, a budget cap, a crash — and the next one
resumes at the exact item it stopped on. Because responses are cached by full request hash,
even work that was in flight is usually free to redo.

Resume is *safe*, not blind: every checkpoint stores a fingerprint of the inputs that
produced it, and fingerprints chain from one stage to the next. So:

- change `judge.min_overall` → only judging re-runs,
- change `generation.temperature` → answers and judging re-run, seeds are reused,
- change `seeds.limit` or edit `prompts.py` → the affected stages (and everything
  downstream) recompute,
- change nothing → the whole run is a no-op that returns the same dataset.

---

## Configuration

One YAML file describes a run. See `configs/` for three complete examples and
`sft-forge init` for a commented starter. The full default config lives in
`src/sft_forge/config.py` (`DEFAULTS`), which doubles as the schema.

```yaml
run:        {name, output_dir, seed, resume}
provider:   {kind, base_url, api_key, timeout_s, max_retries, rate_limit_per_minute}
models:     {generator, judge, prices}
generation: {temperature, top_p, max_tokens, concurrency, seeds_per_topic,
             response_candidates, require_json, system_guidance, extra_body}
seeds:      {source, file, inline, domains, limit, shuffle, expansion{...}}
tasks:      {types, difficulties}
judge:      {enabled, model, sample_rate, min_overall, on_error, rewrite{...}}
filters:    {dedup, length, quality, language, safety}
cache:      {enabled}
budget:     {max_requests, max_tokens, max_usd}
export:     {formats, include_scores, include_system, splits, write_rejected, hub{...}}
```

**Environment interpolation** — `${VAR}` and `${VAR:-default}` are expanded everywhere,
and a `.env` file in the working directory is loaded automatically.

**Overrides** — any value can be overridden on the command line without editing the file:

```bash
sft-forge run -c config.yaml --set generation.concurrency=24 --set judge.min_overall=4.5
sft-forge run -c config.yaml --set seeds.domains='[personal finance, travel]'
```

Nothing is validated by luck: `sft-forge validate -c config.yaml` checks types, ranges,
thresholds, file paths, budget values and export formats, and tells you what to fix.

---

## Outputs

```
runs/general-sft/
├── report.md            # start here: scores, coverage, what was dropped and why, samples
├── manifest.json        # provenance: config hash, dataset fingerprint, models, spend
├── run_config.json      # the fully resolved config (API key redacted)
├── stats.json           # full distributions and percentiles
├── dataset_chatml.jsonl # accepted data, chat format
├── dataset_alpaca.jsonl # accepted data, alpaca format
├── final.jsonl          # canonical accepted dataset
├── rejected.jsonl       # every dropped example + its reason codes (tuning material)
├── filtered.jsonl       # every example with its status and filter audit trail
├── seeds.jsonl, instructions.jsonl, responses.jsonl, judged.jsonl   # stage checkpoints
├── cache.sqlite         # response cache (delete anytime)
└── pipeline.log
```

Every row carries the scores that let it through, so you can always trace a training
example back to the judge verdict that approved it.

An accepted row looks like this:

```json
{
  "messages": [
    {"role": "system", "content": "You are a rigorous, helpful assistant. …"},
    {"role": "user", "content": "What are the practical trade-offs of index scans…?"},
    {"role": "assistant", "content": "…"}
  ],
  "metadata": {
    "id": "ex_3f1a9c2b7d40",
    "domain": "technology & software",
    "task_type": "explanation",
    "difficulty": "medium",
    "scores": {"correctness": 5, "helpfulness": 4, "clarity": 5,
               "instruction_following": 5, "safety": 5, "overall": 4.8}
  }
}
```

---

## Commands

| Command | Purpose |
|---|---|
| `sft-forge init [dir]` | Scaffold `config.yaml`, `.env.example`, `seeds.example.jsonl` |
| `sft-forge validate -c cfg` | Check a config; `--json` prints the resolved version |
| `sft-forge cost -c cfg` | Estimate calls, tokens, dollars and wall-clock before spending |
| `sft-forge preview -c cfg -n 3` | Generate a few examples and print them — iterate on prompts cheaply |
| `sft-forge run -c cfg` | Run the workflow (`--limit`, `--no-resume`, `--run-dir`, `--json`) |
| `sft-forge stats runs/x` | Summarise a run (`--report` for the full markdown report) |
| `sft-forge export runs/x --format sharegpt` | Re-export without regenerating anything |
| `sft-forge push runs/x --repo-id you/name` | Upload to the Hugging Face Hub (needs the `hub` extra) |

Common flags: `-v/--verbose`, `-q/--quiet`, `--env-file`, `--set key=value`.

---

## Quality control

Synthetic data fails in predictable ways. Each one has a detector:

| Failure mode | Detector |
|---|---|
| Near-duplicate prompts | MinHash + LSH banding over 5-grams, Jaccard threshold (`filters.dedup`) |
| One question template reused across topics | topic-stripped skeleton comparison (`filters.dedup.template`) |
| Looping / degenerate text | 4-gram repeat ratio, unique-token ratio (`filters.quality`) |
| Prompt copied back as the answer | instruction/response token overlap heuristic |
| Refusals and "As an AI language model…" | pattern match on short answers |
| Truncated or placeholder output | terminator check, `{{`/`[insert` check |
| Language drift | stopword-ratio and non-ASCII checks (`filters.language`) |
| Unsafe content | phrase blocklist + user-supplied regex denylist (`filters.safety`) |
| Plausible-but-wrong answers | LLM judge with an explicit rubric and calibrating instructions |
| Answers that are close but weak | rewrite-and-re-judge salvage pass instead of discarding |

Tune from evidence: `rejected.jsonl` carries the reason codes, and
`report.md` shows the distribution, so raising `judge.min_overall` or tightening a
threshold is a decision you can make with numbers in front of you.

---

## Working with other providers

Anything that speaks the OpenAI chat-completions wire format works — set `base_url`:

```yaml
provider:
  kind: openai_compat
  base_url: https://api.groq.com/openai/v1          # Groq
  api_key: ${GROQ_API_KEY}
models:
  generator: llama-3.3-70b-versatile
  judge: llama-3.1-8b-instant
```

```yaml
provider:
  kind: openai_compat
  base_url: http://localhost:11434/v1               # Ollama (local, free)
  api_key: ollama
models:
  generator: qwen2.5:14b-instruct
  judge: qwen2.5:14b-instruct
```

Provider-specific knobs go in `generation.extra_body` (for example
`{top_k: 40, repetition_penalty: 1.1}`), and prices can be overridden in
`models.prices` so cost accounting stays honest.

**Seeds from your own documents.** Chunk your text into JSONL rows with a `topic` field
(a one-line description of what the chunk teaches) and set `seeds.source: file`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `provider.api_key is empty` | Export the variable your config references (`export OPENAI_API_KEY=...`), or run `--set provider.kind=mock` offline. `validate` tells you which key it wants. |
| `HTTP 429` warnings in `pipeline.log` | Retries are automatic with backoff. If it persists, lower `provider.rate_limit_per_minute` or `generation.concurrency`. |
| Almost everything is rejected for `judge_below_threshold` | The judge is doing its job. Read `rejected.jsonl`, then either lower `judge.min_overall` or strengthen `generation.system_guidance`. |
| Lots of `template_duplicate_instruction` | Your generator is reusing one phrasing. Raise `generation.temperature`, widen `tasks.types`, or set `seeds.expansion.enabled: true`. |
| `budget.max_usd` tripped mid-run | Nothing is lost — re-run with a raised cap and it continues from the checkpoint. Check `cost` first next time. |
| Answers are all about the same length | That is the prompt, not the pipeline. Change `_LENGTH_GUIDE` in `prompts.py`; the fingerprint will invalidate exactly the affected stages. |
| `stats` says "no dataset found" | Point it at a run directory (`runs/<run.name>`), not at the output root. |

---

## Design notes and limits

- **Synthetic data is not ground truth.** The judge is a model too; it shares the
  generator's blind spots and systematically overrates fluent, confident answers.
  Spot-check `report.md` samples before training, and keep a human-reviewed eval set.
- **Filters are heuristics.** The English-share check is a stopword ratio, not a language
  detector; duplicate detection is character/word based, not semantic. They are cheap
  pre-filters, deliberately tuned for recall, not final arbiters.
- **The judge sees text only.** Pass rate is a quality signal, not a certification.
- **Token estimates** in `cost` assume typical prompt/answer shapes; run
  `--limit 20` and read the actual spend in `stats.json` before scaling up.
- **Legality and licences** are your responsibility: check what the model provider's terms
  allow you to train on, and what your seed data permits.

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # unit + integration tests, no network required
ruff check src tests
```

Tests use the mock provider and `httpx.MockTransport`, so the suite never touches the
network. Contributions that add a provider, a filter, or a training format are welcome —
each is a small, well-isolated module.

## License

MIT — see [LICENSE](LICENSE).
