# Methodology

Read this before quoting any number in [RESULTS.md](RESULTS.md). A benchmark without a stated
methodology is marketing with extra steps.

## Hardware and environment

| | |
|---|---|
| GPU | 1x NVIDIA RTX 5090, 32 GB GDDR7 |
| Driver / CUDA | recorded per run -- see the JSON in `results/` |
| OS | Windows 11 (native) with WSL2 noted where it applies |
| Python | 3.11+ |

**Why single-GPU consumer hardware.** Datacenter numbers do not answer the question a team with
one GPU actually has: *can we serve this model on the box we own, at a latency a user will
tolerate, without renting anything?* That is the constraint `inferlab` is built around.

## What is measured, precisely

### TTFT -- time to first token

Clock starts when `POST /v1/chat/completions` is issued, and stops at the first **non-empty
content delta**, not the first SSE chunk.

Two consequences worth knowing:

- Engines emit a role-only delta (`{"delta":{"role":"assistant"}}`) before any token. Measuring to
  the first chunk instead of the first content token understates TTFT by one round trip.
- Because the requests in a concurrency wave are issued together, TTFT at concurrency > 1
  **includes queueing** behind the other requests in that wave. That is intentional: it is what a
  real user experiences. To get the isolated TTFT, read the concurrency=1 row.

### Decode throughput

```
decode_tps = completion_tokens / (total_latency - ttft)
```

`total_latency` includes connection teardown, so single-request decode numbers are very slightly
pessimistic. This is preferred over the alternative: an optimistic number that quietly depends on
where you stop the clock.

### End-to-end throughput

```
e2e_tps = completion_tokens / total_latency
```

Useful for capacity planning. Not useful as a UX number -- it hides prefill entirely.

### Token counting

Counted from the server's `usage` field when the engine reports one
(`stream_options.include_usage`). When it does not, `inferlab` falls back to counting SSE content
chunks and records `token_source: "chunk-count"` on the sample.

Chunk counting is an approximation -- an engine may emit one token per chunk, or merge tokens, or
split them -- so any row that relied on it is not directly comparable with a row that did not.
The source is stored per sample and surfaced in the JSON, not in the table, to keep the table
readable. Check it before comparing engines.

### Peak VRAM

`nvidia-smi --query-gpu=memory.used` polled every 100 ms on a daemon thread for the duration of
the run.

Peak VRAM is reported because a config can pass every timing test while leaving almost no
headroom. 500 MB free on a 32 GB card is not "it works", it is "it works until concurrency goes up
two steps". A row with `--vram` not set shows `-`, which means *unknown*, not *low*.

### Warmup

`--warmup N` (default 2) issues N sequential requests per workload that are discarded entirely.
They exist so that CUDA graph capture, kernel autotuning, memory-pool growth and page-table
warmup are charged to the warmup instead of to the first measured sample. Without this, the first
sample of every run looks like a hardware failure.

## Quality measurement

### Perplexity

Evaluated with `POST /v1/completions`, `echo=true`, `max_tokens=0`, `logprobs=1`, and the mean
per-token negative log-likelihood taken over the returned `token_logprobs`.

- The first token of each echo is unconditioned (its log-prob is `null`) and is skipped.
- The corpus is evaluated in sentence-aligned character windows because servers cap the prompt at
  `max_model_len`. Window boundaries therefore cost a little accuracy at the seam; the effect is
  small and identical across every config measured against the same corpus and window size, so
  the *deltas* are meaningful even where the absolute value is slightly pessimistic.
- Report the **delta against the BF16 reference**, never the absolute perplexity. Absolute
  perplexity varies with tokenizer and corpus; the delta is the number that answers "did this
  quantization hurt?".

### Task accuracy

Multiple-choice, scored by **option log-probability**: the prompt ends at `Answer:` and the
log-probability of each candidate letter at that position is compared.

This is deliberately not "ask the model to answer and parse the letter". Free-form parsing depends
on the model's verbosity, on stop tokens, and on how forgiving the parser is -- which means it
depends on the quantization, which is exactly the variable under test. Log-prob scoring is
deterministic, temperature-independent, and comparable across engines.

## Concurrency

The sweep is the point. Concurrency levels are issued as one simultaneous wave, so:

- it measures continuous batching and chunked prefill doing their job, and
- it exposes the configs that only work when nothing else is running.

Default sweep: `1, 8, 32`. The concurrency=1 row is a demo; the higher rows are a server.

## Sampling

`temperature=0.0` for every measured run. Results must not depend on sampling luck, and a
throughput number that changes between two runs of the same config is a throughput number nobody
should quote.

## Known sources of error

1. **Wall-clock noise.** Other GPU work during a run shows up as an outlier. Percentiles (`p50`,
   `p95`) are reported instead of means for this reason.
2. **Thermal / clock behaviour.** Sustained decode on a 32 GB card can pull clocks down. Long
   runs are more pessimistic than short ones.
3. **Cross-engine comparability.** Engines differ in how they schedule, how they treat
   `max_tokens`, and whether `prompt_tokens` includes the chat template. Compare rows with the
   same engine version and the same workload, and treat cross-engine gaps of under ~5% as noise.
4. **`token_source` drift.** See above. Check the JSON.
5. **First-run disk cache.** The first load of a model reads weights from disk. All measured runs
   follow the warmup, so this affects the warmup, not the numbers -- but it means a cold-start
   number is not in this table at all.

## Truncation is a silent failure mode

A sample with `finish_reason == "length"` did not complete. It is reported loudly rather than
averaged in, because the failure it hides is severe and invisible in a timing table.

The concrete evidence, from a measurement on a 14B model:

| Generation cap | GSM8K accuracy |
|---|---|
| 320 tokens | 76.6% |
| 1024 tokens | 95.3% |

Almost the entire gap was answers being cut off mid-reasoning and then scored as wrong -- not a
capability difference. A harness that reports timing and ignores `finish_reason` will happily
table a fast number for a config that never finished its answers.

Two consequences for this lab:

- `inferlab bench` records `finish_reason` per sample and prints a truncation warning naming the
  affected `(workload, concurrency)` pairs. Those rows are not comparable with completed rows.
- The per-workload `max_tokens` budgets in `inferlab/workloads.py` are deliberately generous. A
  budget small enough to cause truncation turns a capability question into a length question.

## What perplexity does and does not predict

Perplexity delta is a distributional measure, not a capability measure. The clearest data point
against treating it as a predictor, from a 14B Q6_K-versus-F16 comparison:

| Metric | F16 | Q6_K | Delta |
|---|---|---|---|
| Perplexity | 100.00% | 100.70% | +0.70% |
| MMLU | 0.770 | 0.772 | +0.002 |

A 0.7% perplexity shift produced *no measurable* task-score change. Perplexity that barely moves
does not license a claim that nothing changed -- and, more importantly, a large perplexity move is
the only case where ppl is a useful early warning. Treat it as a smoke detector for gross
quantization damage, and let the task-accuracy row answer whether the model still works.

This is why `inferlab quality` reports both, and why neither is presented as a substitute for the
other.

## Reproducing a row

Every result JSON carries the recipe name, engine, model, URL, concurrency levels, run counts,
sampling params and the git commit of the harness that produced it. To reproduce:

```bash
git checkout <harness_commit>
# start the backend with the launch command from backends/README.md
inferlab bench --config <backend-label> --url http://localhost:8000 --vram --json results/<label>.json
```

If two runs on the same commit disagree by more than 10%, suspect the environment before the
harness.
