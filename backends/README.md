# Backends

One launch command per engine, matching the recipes in [`configs/default.yaml`](../configs/default.yaml).
Every command below exposes an **OpenAI-compatible** endpoint on port 8000 so `inferlab bench`
works identically against all of them.

Before running any of these: the model weights must already be downloaded, and the GPU must be
idle. A benchmark taken while something else uses the card is not a benchmark.

---

## vLLM

```bash
# NVFP4 weights, FP8 KV-cache, chunked prefill, 24k context
python -m vllm.entrypoints.openai.api_server \
    --model <hf-repo-with-nvfp4-weights> \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.92 \
    --port 8000
```

Notes on the flags, because they are the whole point:

| Flag | Why |
|---|---|
| `--kv-cache-dtype fp8` | Halves KV-cache memory. On 32 GB this is the difference between 8k and 24k context. |
| `--enable-chunked-prefill` | Bounds the latency a short request waits behind a long prompt. Required for any sane TTFT p95 on long-context workloads. |
| `--gpu-memory-utilization 0.92` | Leaves headroom so the sampler can distinguish "fit" from "barely fit". |
| `--max-model-len 24576` | Pins the context so every row in RESULTS.md is directly comparable. |

BF16 quality reference (expect it to be tight on memory at 24k):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <hf-repo-bf16> \
    --kv-cache-dtype auto \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.92 \
    --port 8000
```

## SGLang

```bash
python -m sglang.launch_server \
    --model-path <hf-repo-with-nvfp4-weights> \
    --kv-cache-dtype fp8 \
    --chunked-prefill-size 8192 \
    --context-length 24576 \
    --mem-fraction-static 0.88 \
    --port 8000
```

SGLang's radix attention is what makes repeated-prefix workloads (agents, batch summarization,
system prompts) cheaper than they look. The single-request decode row will not reveal that --
watch the concurrency sweep instead.

## TensorRT-LLM

TensorRT-LLM needs a per-model, per-GPU engine build, so the launch command is the short part:

```bash
trtllm-serve serve \
    <engine-build-dir> \
    --port 8000
```

Record the `tensorrt_llm` and TensorRT versions in the result JSON (or the `extra_meta` of the
recipe). An engine built against a different TensorRT minor version is a different engine, and a
row that cannot be rebuilt cannot be quoted.

## llama.cpp

```bash
llama-server \
    --model <model-q4_k_m.gguf> \
    --ctx-size 24576 \
    --n-gpu-layers 999 \
    --port 8000
```

Included because it is the floor everyone compares against, and because it keeps working on
hardware where nothing else will. Expect far lower VRAM and far lower throughput; that trade is
the finding, not a defect of the measurement.

---

## Docker

[`docker-compose.yml`](../docker-compose.yml) brings up a vLLM server with the NVFP4/FP8
recipe and a healthcheck. Use it when you want the environment to be part of the artifact rather
than part of the folklore:

```bash
docker compose up vllm
# then, from the host:
inferlab bench --config vllm-nvfp4 --url http://localhost:8000 --vram --json results/vllm-nvfp4.json
```

The NVIDIA Container Toolkit is required.

## Verifying the endpoint before you benchmark

`inferlab bench` fails loudly if it cannot reach a server, but it is worth checking by hand:

```bash
curl http://localhost:8000/v1/models
```

If this returns no models, the weights did not load -- fix that before spending twenty minutes
collecting numbers about a server that is not serving.

## Recording a new backend

Add a recipe to [`configs/default.yaml`](../configs/default.yaml) with its launch command in the
`notes` field, then run:

```bash
inferlab bench --config <name> --url http://localhost:8000 --vram --json results/<name>.json
inferlab report results/*.json --out docs/RESULTS.md
```

Nothing enters RESULTS.md except through `inferlab report`.
