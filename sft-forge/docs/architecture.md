# Architecture

Why the pipeline is shaped the way it is, and where to extend it.

## The problem this solves

Generating SFT data looks like a `for` loop over prompts. In practice the expensive part is
*iteration*: the first pass never produces data you would train on. You change a prompt,
tighten the judge, notice the answers are all the same length, discover half of them are
duplicates — and each experiment costs money and wall-clock time.

So the design optimises for the loop, not the first run:

| Property | Mechanism |
|---|---|
| Re-running after a prompt change is cheap | response cache keyed by the full request hash |
| Re-running after a config change is *not* free but *is* targeted | per-stage input fingerprints |
| Interrupted work is never lost | per-stage checkpoints + cursor + partial file |
| Nothing silently disappears | every rejection is recorded with a reason code on the record |
| A runaway config cannot burn the budget | hard caps on requests / tokens / dollars |

## Stages

```
seeds ──▶ instructions ──▶ responses ──▶ judge ──▶ filter ──▶ export
  │            │               │            │          │         │
  │            │               │            │          │         └─ manifest, report, formats
  │            │               │            │          └─ dedup + heuristics (no model calls)
  │            │               │            └─ rubric, best-of-n, one rewrite, re-judge
  │            │               └─ N answers per instruction (best-of-n when configured)
  │            └─ N self-contained requests per seed, styled from the seed
  └─ bundled taxonomy | JSONL | inline | LLM expansion
```

Each arrow is a `StageRunner` invocation. A stage receives a list of items and a worker
function, and writes rows in item order.

## Checkpointing and resume

A stage owns four files in the run directory:

| File | Meaning |
|---|---|
| `<stage>.jsonl` | completed output (only exists once the stage finished) |
| `<stage>.jsonl.partial` | output written so far, flushed after every item |
| `<stage>.cursor.json` | how many items finished, plus the input fingerprint |
| `<stage>.done.json` | completion marker, item/row counts, input fingerprint |

A resumed run trusts a checkpoint only if **both** the stage fingerprint and the queued item
count match. Otherwise the stale files are discarded and the stage recomputes.

### Fingerprints

`stage_fingerprint(cfg, stage)` hashes exactly the inputs that determine that stage's
output, and fingerprints chain upstream: `responses` includes the `instructions`
fingerprint, `judge` includes the `responses` one. Editing:

* `judge.min_overall` → only the judge stage recomputes,
* `generation.temperature` → responses, judge and export recompute; seeds are reused,
* `seeds.limit` or `tasks.types` → everything recomputes,
* `prompts.py` → every model-facing stage recomputes (the prompt file is hashed).

### Failure semantics

* A worker raising an ordinary exception writes an `_error` row and is counted; the stage
  continues. Small failure rates are normal and should not cost you the run.
* `BudgetExceeded` is a **control signal**, not a data error. It propagates out of every
  worker (this is why the broad `except Exception` handlers in `generate`, `judge` and
  `seeds` re-raise it), stops the stage, and leaves the checkpoint resumable. The item that
  tripped the cap is *not* written, so a later run redoes it.
* `KeyboardInterrupt` leaves the same state as a budget stop.

## Data model

Everything is a plain dict on disk; in memory it is one of two dataclasses
(`sft_forge.schemas`):

* `Seed` — what to teach (topic, domain, task type, difficulty, audience, constraints).
  Content-addressed id, so the same seed in two runs is the same seed.
* `Record` — one training example plus provenance: `scores`, `judge`, `filters`,
  `reject_reasons`, `rewrite_of`, `usage`, `meta`.

`Record.export_view(fmt)` is the only place that knows about training formats, so adding a
format is a five-line change.

## Quality control layers

1. **Generation-time contract** — prompts demand self-contained requests and answers with a
   stated depth; JSON replies are parsed leniently (`jsonx`) and degrade to deterministic
   templates rather than being dropped.
2. **Judge** — five-dimension rubric, temperature 0, explicit calibration, JSON verdict.
   `best_of` scores several candidate answers and keeps the winner. Near-misses get one
   rewrite and a re-judge; only genuinely weak answers are discarded.
3. **Deterministic filters** — cheap, auditable, and tuned for recall: length, lexical
   diversity, repetition, instruction parroting, refusals, placeholders, language drift,
   blocklists/regex denylist.
4. **Deduplication** — exact, MinHash+LSH near-duplicate, and *template* duplicates. The
   last one matters most for synthetic data: a generator that reuses "What should someone
   know about X?" across topics produces examples that look different to a shingle-based
   comparison but are the same template. `instruction_skeleton()` strips the seed's topic
   words and compares what remains.

## Concurrency and rate limits

Stages run items through a `ThreadPoolExecutor` with `pool.map`, which preserves input order
while running `concurrency` requests in flight; results are written in order, which is what
makes a cursor meaningful. A token-bucket limiter (`utils.RateLimiter`) throttles globally,
and retries inside the provider use exponential backoff with jitter, honouring
`Retry-After`. Budget checks happen before each request, so with concurrency N the ceiling
can be overshot by at most N-1 in-flight calls.

## Extension points

| To add… | Do this |
|---|---|
| A provider | implement `Provider.chat` in `providers/`, register it in `build_provider` |
| A filter | write `filter_*(record, section) -> FilterOutcome` in `filters.py`, add to `FILTERS` |
| A training format | add a branch to `Record.export_view` and to `KNOWN_EXPORT_FORMATS` |
| Seed sources | extend `build_seeds` in `seeds.py` |
| Prompt changes | edit `prompts.py`; fingerprints and the cache pick the change up automatically |

Tests for all of these run offline: the `MockProvider` is a real `Provider` whose replies
depend only on the prompt and the `hint` label, and HTTP code is tested through
`httpx.MockTransport`.

## Known limits

* Similarity is lexical, not semantic. True paraphrase duplicates survive unless you lower
  `filters.dedup.threshold` or add an embedding filter.
* The judge is a model: it shares the generator's blind spots and overrates confident prose.
  Its scores are a filter, not a ground truth.
* Token estimates are assumptions, not measurements; verify with `--limit 20` before a big
  run.
* The `seeds` stage writes once and checkpoints; it is not itself parallelised (it is cheap
  except when LLM expansion is enabled, which is one call per domain).
