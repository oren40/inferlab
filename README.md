# inferlab

**Reproducible LLM inference benchmarking & optimization lab for single-GPU Blackwell.**

Answers one question that vendor blogs avoid: *what is the actual latency / throughput / VRAM /
quality trade-off of each inference engine and each quantization format on the exact hardware I own?*

`inferlab` runs identical workloads across **vLLM, SGLang, TensorRT-LLM, llama.cpp**, on
**NVFP4 / FP8 / GGUF** weights, and -- critically -- measures the **quality cost** of every
optimization, not just the speed win. A 300 tok/s number is meaningless if the model forgot how to
reason.

- Hardware under test: **1x NVIDIA RTX 5090 (32 GB)**
- Measured: TTFT, decode tok/s, end-to-end throughput, peak VRAM, KV-cache efficiency,
  perplexity delta, task accuracy
- Output: machine-readable JSON + human-readable tables in [`docs/RESULTS.md`](docs/RESULTS.md)

---

## Results

> **Status:** awaiting first full run. The table below is the exact output format -- it is generated
> by `inferlab report` from `results/*.json`, never hand-edited. Every number carries the config
> that produced it (see [METHODOLOGY](docs/METHODOLOGY.md)).

| Engine | Backend label | Workload | Model | Conc | n | TTFT p50 (ms) | TTFT p95 (ms) | Decode p50 (tok/s) | Decode p95 (tok/s) | E2E p50 (tok/s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | vllm-nvfp4 | chat-short | -- | 1 | -- | -- | -- | -- | -- | -- | -- |
| vLLM | vllm-nvfp4 | chat-short | -- | 8 | -- | -- | -- | -- | -- | -- | -- |
| vLLM | vllm-nvfp4 | long-context | -- | 1 | -- | -- | -- | -- | -- | -- | -- |
| SGLang | sglang-nvfp4 | chat-short | -- | 8 | -- | -- | -- | -- | -- | -- | -- |
| TensorRT-LLM | trtllm-fp8 | chat-short | -- | 8 | -- | -- | -- | -- | -- | -- | -- |
| llama.cpp | llamacpp-q4km | chat-short | -- | 1 | -- | -- | -- | -- | -- | -- | -- |
| vLLM (baseline) | vllm-bf16-baseline | chat-short | -- | 1 | -- | -- | -- | -- | -- | -- | -- |

Full data, per-config launch commands and raw JSON: [`docs/RESULTS.md`](docs/RESULTS.md)

---

## Why this exists

Public LLM benchmarks are almost always either:

1. **Vendor-run** -- a single engine, a single model, a single carefully-chosen config, no baseline.
2. **Uniform** -- every quantization gets the same "fine" verdict with no perplexity or accuracy
   evidence.
3. **Datacenter-only** -- A100/H100 numbers that tell a person with one GPU nothing about what they
   can actually ship.

`inferlab` is the benchmark I wanted when deciding how to serve a 27B model locally: one command,
every engine, identical prompts, honest error bars, and a quality guardrail so a config that is fast
*and* broken never wins.

## What it measures

| Metric | Definition |
|---|---|
| **TTFT** | Time to first *generated* token -- what the user feels as "responsiveness" |
| **Decode tok/s** | `completion_tokens / (total_latency - TTFT)` -- the number that matters for streaming UX |
| **E2E throughput** | `completion_tokens / total_latency` -- aggregate serving capacity |
| **Peak VRAM** | Sampled from `nvidia-smi` at 100 ms during the run -- catches OOM-adjacent configs |
| **PPL delta** | Perplexity delta vs. the BF16 reference on a fixed corpus -- quantization damage |
| **Task accuracy** | Accuracy on a fixed multiple-choice set -- the metric that catches "fast but wrong" |

Every run records the full config: engine version, model revision, quantization scheme, KV-cache
dtype, max context, batch size, sampling params, and the git commit of the harness.

---

## Install

```bash
git clone https://github.com/oren40/inferlab.git
cd inferlab
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[bench]"
```

Requires: Python 3.11+, `nvidia-smi` on PATH, and at least one OpenAI-compatible server running.

## Usage

**1. Start a backend** (see [`backends/`](backends/README.md) for the exact launch command per
engine):

```bash
# vLLM, NVFP4 weights, FP8 KV-cache, chunked prefill
python -m vllm.entrypoints.openai.api_server \
    --model <nvfp4-repo> \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-model-len 24576 \
    --port 8000
```

**2. Benchmark it:**

```bash
inferlab bench \
    --backend vllm-nvfp4 \
    --url http://localhost:8000 \
    --workload mixed \
    --concurrency 1 8 32 \
    --max-tokens 512 \
    --runs 5 --warmup 2 \
    --vram --json results/vllm-nvfp4.json
```

**3. Measure what the quantization cost you:**

```bash
inferlab quality \
    --url http://localhost:8000 \
    --model <nvfp4-repo> \
    --tasks reasoning mcq \
    --perplexity-corpus data/perplexity/sample.txt \
    --baseline-json results/vllm-bf16-baseline.json
```

**4. Generate the tables in `docs/RESULTS.md`:**

```bash
inferlab report results/*.json --out docs/RESULTS.md
```

Everything is driven by CLI flags or [`bench/configs/default.yaml`](bench/configs/default.yaml) --
no notebook state, no manual transcription, no results that cannot be reproduced from the repo.

## Concurrency is not optional

A single-request tok/s number describes a demo, not a server. `inferlab` sweeps concurrency
(1 -> 8 -> 32 by default) because that is where chunked prefill, continuous batching and KV-cache
pressure either work or fall apart.

## Roadmap

- [x] Streaming benchmark harness (TTFT / decode / E2E / VRAM)
- [x] Concurrency sweep
- [x] Perplexity delta vs. BF16 reference
- [x] Task-accuracy guardrail
- [x] Report generator (`docs/RESULTS.md`)
- [ ] Speculative decoding / MTP comparison
- [ ] Multi-GPU (TP=2) scaling records
- [ ] Long-context (128K) KV-cache pressure curve
- [ ] Interactive results dashboard

See [docs/RESULTS.md](docs/RESULTS.md) for measured data as it lands.

## Methodology

Read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) before quoting any number from this repo. It
documents warmup policy, how tokens are counted, why TTFT excludes queueing, and the known sources
of error. Benchmarks without a methodology are marketing.

## Contributing

Runs on hardware other than an RTX 5090 are the most useful contribution available -- open a PR that
drops a JSON file into `results/` and adds a row to `RESULTS.md`. A name for the GPU in the config
is enough.

## License

Apache-2.0. See [LICENSE](LICENSE).
