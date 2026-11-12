"""A synthetic OpenAI-compatible backend, used to validate the pipeline without a GPU.

Why this exists
---------------
A benchmarking tool that has never been run end-to-end is a liability, not an asset.
CI needs to prove that `inferlab bench`, `inferlab quality` and `inferlab report` work --
that streaming is parsed, that usage is read, that perplexity is computed, that
RESULTS.md regenerates -- without waiting for a 32 GB GPU to be free.

So the harness is exercised against a deterministic fake server that speaks the same
protocol. It produces numbers, but produces *no evidence*: the numbers describe this
mock, not any hardware.

The separation is enforced, not advisory:

- every synthetic result carries ``synthetic: true``
- ``report.render`` puts synthetic results in a clearly-labeled "Pipeline verification"
  section, never in the measured-results table
- ``run_pipeline_check`` asserts that no synthetic row can reach the measured set

Treat this as a smoke test for the pipeline. It says "the tool runs". It never says
"the tool measures well".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

DEFAULT_MODEL = "synthetic-reference-model"

# A fixed corpus so the mock generates identically on every run. Changing these words
# changes the CI numbers -- which is fine, because they do not mean anything.
_WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "the", "engine", "samples", "tokens", "and", "the", "cache", "warms", "up", "before", "the", "benchmark", "records", "the", "first", "measurement"]


@dataclass
class SyntheticConfig:
    """Everything the mock needs to behave consistently."""

    model_id: str = DEFAULT_MODEL
    max_tokens: int = 64
    prompt_tokens: int = 40
    seed: int = 20261113
    # False exercises the harness's chunk-counting fallback.
    report_usage: bool = True
    # False exercises the missing-letter path in accuracy scoring.
    report_top_logprobs: bool = True
    # True forces finish_reason="length", exercising the truncation guard.
    truncate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "prompt_tokens": self.prompt_tokens,
            "seed": self.seed,
            "report_usage": self.report_usage,
            "report_top_logprobs": self.report_top_logprobs,
            "truncate": self.truncate,
        }


class SyntheticBackend:
    """A minimal, deterministic OpenAI-compatible server for pipeline validation."""

    def __init__(self, cfg: SyntheticConfig | None = None) -> None:
        self.cfg = cfg or SyntheticConfig()
        self.requests_seen: list[dict[str, Any]] = []

    # -- HTTP handlers ----------------------------------------------------------------

    async def models(self, _: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": self.cfg.model_id, "object": "model"}]}
        )

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests_seen.append(body)

        max_tokens = int(body.get("max_tokens", self.cfg.max_tokens))
        completion_tokens = max_tokens if self.cfg.truncate else min(max_tokens, self.cfg.max_tokens)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        if not body.get("stream"):
            return web.json_response(self._chat_payload(completion_tokens, include_usage))

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def emit(payload: dict[str, Any]) -> None:
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())

        # A role-only delta first: real engines emit this before any token, and the
        # harness is supposed to measure past it.
        await emit(self._chunk({"role": "assistant"}, None))
        for i in range(completion_tokens):
            await emit(self._chunk({"content": f"{_WORDS[i % len(_WORDS)]} "}, None))
        await emit(self._chunk({}, "length" if self.cfg.truncate else "stop"))

        if include_usage and self.cfg.report_usage:
            await emit({**self._chunk({}, None), "choices": [], "usage": self._usage(completion_tokens)})

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": "synthetic",
            "object": "chat.completion.chunk",
            "model": self.cfg.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def _usage(self, completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": self.cfg.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": self.cfg.prompt_tokens + completion_tokens,
        }

    def _chat_payload(self, completion_tokens: int, include_usage: bool) -> dict[str, Any]:
        return {
            "id": "synthetic",
            "object": "chat.completion",
            "model": self.cfg.model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if self.cfg.truncate else "stop",
                    "message": {"role": "assistant", "content": " ".join(_WORDS[:2])},
                }
            ],
            "usage": self._usage(completion_tokens) if include_usage else None,
        }

    async def completions(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests_seen.append(body)

        prompt = str(body.get("prompt", ""))
        echo = bool(body.get("echo"))
        logprobs_n = int(body.get("logprobs", 0))
        rng = random.Random(self.cfg.seed)

        if echo:
            n_prompt_tokens = max(1, len(prompt) // 4)
            token_logprobs: list[float | None] = [None] + [
                round(rng.uniform(-2.5, -0.5), 4) for _ in range(n_prompt_tokens - 1)
            ]
            tokens = ["tok"] * n_prompt_tokens
        else:
            token_logprobs = [None]
            tokens = ["tok"]

        letters = ["A", "B", "C", "D", "E"]
        top_logprobs: dict[str, float] = {}
        if self.cfg.report_top_logprobs:
            top_logprobs = dict(
                zip(letters, [round(rng.uniform(-3.0, -1.0), 4) for _ in letters])
            )

        logprobs_block: dict[str, Any] = {"tokens": tokens, "token_logprobs": token_logprobs}
        # top_logprobs is per generated token, so a list with one entry when
        # max_tokens == 1 -- which is what accuracy scoring requests.
        if logprobs_n > 0:
            logprobs_block["top_logprobs"] = [top_logprobs]

        return web.json_response(
            {
                "id": "synthetic",
                "object": "text_completion",
                "model": self.cfg.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": " ".join(_WORDS[:2]),
                        "finish_reason": "stop",
                        "logprobs": logprobs_block,
                    }
                ],
            }
        )

    # -- lifecycle --------------------------------------------------------------------

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/completions", self.completions)
        return app

    async def serve(self) -> tuple[web.AppRunner, str]:
        runner = web.AppRunner(self.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[attr-defined]

    async def __aenter__(self) -> tuple[SyntheticBackend, str]:
        self._runner, self._url = await self.serve()
        return self, self._url

    async def __aexit__(self, *exc: object) -> None:
        await self._runner.cleanup()


@dataclass
class SyntheticRunResult:
    """What a pipeline-validation run produced, with synthetic provenance attached."""

    backend_url: str
    bench_path: str | None = None
    quality_path: str | None = None
    report_path: str | None = None
    n_bench_samples: int = 0
    notes: list[str] = field(default_factory=list)


async def run_pipeline_check(
    outdir: str = "results-synthetic",
    *,
    report_out: str | None = None,
) -> SyntheticRunResult:
    """Exercise bench -> quality -> report against the synthetic backend.

    This is the CI smoke test. It proves the pipeline runs; it produces no evidence
    about any hardware.
    """
    from . import harness as _harness
    from . import quality as _quality
    from . import report as _report
    from .workloads import resolve

    notes: list[str] = []

    async with SyntheticBackend() as (_backend, url):
        cfg = _harness.BenchConfig(
            url=url,
            model="auto",
            backend_label="synthetic-pipeline-check",
            engine="unknown",
            workloads=resolve(["mixed"]),
            concurrencies=[1, 2],
            runs=2,
            warmup=1,
            sample_vram=False,
        )
        cfg.extra_meta["synthetic"] = True
        bench = await _harness.run_benchmark(cfg)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    bench_path = str(out / "synthetic-bench.json")
    Path(bench_path).write_text(json.dumps(bench.to_dict(), indent=2), encoding="utf-8")

    quality_payload: dict[str, Any] = {"schema": "inferlab.quality/1"}
    async with SyntheticBackend() as (_, url), aiohttp.ClientSession() as session:
        try:
            ppl = await _quality.perplexity(
                session,
                url=url,
                model=DEFAULT_MODEL,
                api_key="EMPTY",
                text=" ".join(_WORDS * 20),
            )
            quality_payload.update(
                {
                    "backend": "synthetic-pipeline-check",
                    "model": DEFAULT_MODEL,
                    "synthetic": True,
                    "perplexity": ppl.to_dict(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - a synthetic failure is a CI signal
            notes.append(f"perplexity path failed: {type(exc).__name__}: {exc}")

        try:
            acc = await _quality.accuracy(
                session,
                url=url,
                model=DEFAULT_MODEL,
                api_key="EMPTY",
                tasks=_quality.load_tasks(["data/quality/tasks.json"]),
            )
            quality_payload.update({"accuracy": acc.to_dict()})
        except Exception as exc:  # noqa: BLE001
            notes.append(f"accuracy path failed: {type(exc).__name__}: {exc}")

    quality_path = str(out / "synthetic-quality.json")
    Path(quality_path).write_text(json.dumps(quality_payload, indent=2), encoding="utf-8")

    if report_out:
        bench_results = _report.load_results([bench_path])
        quality_results = _report.load_results([quality_path])
        leaked = [
            r for r in bench_results if not r.get("config", {}).get("synthetic")
        ]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} synthetic result(s) are missing the synthetic flag and "
                "would appear in the measured-results table. Fix the flag, do not "
                "suppress this."
            )
        text = _report.render(bench_results)
        text += _report.render_quality(quality_results)
        text += _report.render_synthetic_note(_report.partition(bench_results)[1])
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text(text, encoding="utf-8")

    return SyntheticRunResult(
        backend_url=url,
        bench_path=bench_path,
        quality_path=quality_path,
        report_path=report_out,
        n_bench_samples=len(bench.samples),
        notes=notes,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inferlab self-test",
        description=(
            "Validate the bench/quality/report pipeline against a synthetic "
            "OpenAI-compatible server. Requires no GPU, no model and no network."
        ),
    )
    p.add_argument(
        "--outdir",
        default="results-synthetic",
        help="where to write the synthetic result JSON",
    )
    p.add_argument(
        "--report-out",
        default=None,
        help="also render a report here, so you can see the synthetic section render",
    )
    return p


def self_test_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Running inferlab pipeline validation against a synthetic server...")
    print("(These numbers describe a mock. They are not measurements of anything.)")
    print()

    result = asyncio.run(
        run_pipeline_check(outdir=args.outdir, report_out=args.report_out)
    )

    print(f"backend     : {result.backend_url}")
    print(f"bench json  : {result.bench_path}  ({result.n_bench_samples} samples)")
    print(f"quality json: {result.quality_path}")
    if result.report_path:
        print(f"report      : {result.report_path}")

    if result.notes:
        print()
        print("Warnings:")
        for note in result.notes:
            print(f"  - {note}")
        return 1

    print()
    print("Pipeline OK: streaming parse, token counting, perplexity, accuracy scoring")
    print("and report generation all ran end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test_main())