"""Benchmark workloads.

A workload is a prompt plus the decoding budget it should be given. Workloads are
deterministic (seeded) so that two runs on the same config produce the same tokens and
therefore comparable numbers.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Rough planning constant: ~4 characters per token for English/code text. Only used to size
# synthetic long-context prompts; measured token counts always come from the server.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Workload:
    """A single benchmark prompt."""

    name: str
    prompt: str
    max_tokens: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def approx_tokens(self) -> int:
        return len(self.prompt) // CHARS_PER_TOKEN


_SHORT_CHAT = "Explain the difference between prefill and decode in LLM inference, in three sentences."
_CODE_GEN = (
    "Write a Python function `running_median(nums)` that returns a list of the running median "
    "after each element is added. Use two heaps. Include type hints and a docstring."
)
_REASONING = (
    "A server has 32 GB of VRAM. Model weights in FP8 are 14 GB. You want a 24k-token context "
    "with an FP8 KV-cache. Estimate how many concurrent sequences fit and state the assumptions "
    "you had to make. Show your arithmetic."
)
_SUMMARIZE = (
    "Summarize the following text in exactly five bullet points, no preamble:\n\n"
    "Continuous batching keeps the GPU busy by admitting new requests into the batch as soon as "
    "a slot frees, instead of waiting for the whole batch to finish. Chunked prefill splits a "
    "long prompt into chunks that are scheduled together with decode steps, which bounds the "
    "latency a short request must wait behind a long one. KV-cache quantization trades a small "
    "amount of numerical error for a large reduction in memory per cached token, which directly "
    "increases the number of resident sequences. Speculative decoding runs a cheap draft model "
    "and verifies its tokens with the target model in one forward pass, which can raise decode "
    "throughput when the draft model agrees often enough."
)

_FILLER_WORDS = ["system", "latency", "throughput", "cache", "buffer", "schedule", "request", "sequence", "token", "batch", "prefill", "decode", "engine", "kernel", "memory", "bandwidth", "checkpoint", "replica", "cluster", "node", "metric", "trace", "window", "latency", "pipeline", "queue", "worker", "thread", "socket", "packet", "header", "payload", "checksum", "index", "shard", "partition", "snapshot", "journal", "replay", "backup", "replica", "failover", "quorum", "lease", "election", "consensus", "commit", "log"]


def _filler_paragraph(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(_FILLER_WORDS) for _ in range(words))


def long_context_prompt(target_tokens: int = 24_576, needle: str | None = None) -> str:
    """Build a deterministic long-context prompt with a verifiable needle in the middle.

    Sizing the prompt at ~target_tokens of filler text and hiding a unique fact in the middle
    means a model that actually attended to the whole context can answer, while one that
    truncated silently cannot. That turns context length into a measurable claim.
    """
    rng = random.Random(20260101)
    needle = needle or (
        "The calibration constant for the Haifa cluster is 7.31 petahash-seconds per terabyte."
    )

    body_chars = target_tokens * CHARS_PER_TOKEN
    half = body_chars // 2

    left: list[str] = []
    written = 0
    while written < half:
        para = _filler_paragraph(rng, 90)
        left.append(para)
        written += len(para) + 2

    filler = " ".join(left)
    return (
        "Below is a long technical log. Read all of it.\n\n"
        f"{filler}\n\n"
        f"NOTE: {needle}\n\n"
        f"{filler}\n\n"
        "Question: what is the calibration constant for the Haifa cluster, and what are its units? "
        "Answer with the number and units only."
    )


_BUILTINS: dict[str, Workload] = {
    "chat-short": Workload(
        name="chat-short",
        prompt=_SHORT_CHAT,
        max_tokens=256,
        tags=("chat",),
        description="Short conversational turn; isolates decode speed at low prompt cost.",
    ),
    "chat-long": Workload(
        name="chat-long",
        prompt=_SUMMARIZE + _SUMMARIZE,
        max_tokens=512,
        tags=("chat", "long-input"),
        description="~600 prompt tokens, longer generation; mixed prefill/decode pressure.",
    ),
    "code-gen": Workload(
        name="code-gen",
        prompt=_CODE_GEN,
        max_tokens=600,
        tags=("code",),
        description="Long generation dominated by decode; exposes per-token cost.",
    ),
    "reasoning": Workload(
        name="reasoning",
        prompt=_REASONING,
        max_tokens=600,
        tags=("reasoning",),
        description="Arithmetic-heavy answer; useful for accuracy checks, not just speed.",
    ),
    "long-context": Workload(
        name="long-context",
        prompt=long_context_prompt(),
        max_tokens=128,
        tags=("long-context", "needle"),
        description="~24k-token prompt with a needle in the middle; stresses prefill and KV-cache.",
    ),
}

# Order matters for the "mixed" preset: cheapest first so a failure is obvious and quick.
MIXED = ("chat-short", "code-gen", "reasoning", "chat-long", "long-context")


def builtin_names() -> tuple[str, ...]:
    return tuple(_BUILTINS)


def resolve(names: Sequence[str]) -> list[Workload]:
    """Resolve workload names, expanding presets. Unknown names raise KeyError with a hint."""
    resolved: list[Workload] = []
    for raw in names:
        for name in raw.split(","):
            name = name.strip()
            if not name:
                continue
            if name in ("mixed", "all"):
                resolved.extend(_BUILTINS[n] for n in MIXED)
                continue
            if name not in _BUILTINS:
                raise KeyError(
                    f"unknown workload {name!r}; available: {', '.join(builtin_names())} "
                    f"(or the preset 'mixed')"
                )
            resolved.append(_BUILTINS[name])
    if not resolved:
        raise ValueError("no workloads resolved")
    return resolved


def from_file(path: str | Path) -> Workload:
    """Load a workload from a .txt (prompt) or .json ({\"prompt\": ..., \"max_tokens\": ...}) file."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
        return Workload(
            name=data.get("name", p.stem),
            prompt=data["prompt"],
            max_tokens=int(data.get("max_tokens", 512)),
            tags=tuple(data.get("tags", ())),
            description=data.get("description", f"loaded from {p}"),
        )
    return Workload(
        name=p.stem,
        prompt=p.read_text(encoding="utf-8").strip(),
        max_tokens=512,
        tags=("custom",),
        description=f"loaded from {p}",
    )
