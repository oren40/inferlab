"""Streaming benchmark harness for OpenAI-compatible LLM servers.

Design notes
------------
* Token counts come from the server's ``usage`` field when it reports one
  (``stream_options.include_usage``), and fall back to counting SSE chunks otherwise.
  Chunk counting is an approximation -- one chunk is not one token on every engine -- so the
  source is recorded per sample and surfaced in the report.
* TTFT is measured to the first non-empty *content* delta, not to the first SSE chunk. Engines
  emit a role-only delta before any token, which would otherwise deflate TTFT.
* Requests inside one concurrency level start together. TTFT therefore includes queueing
  behind the requests issued in the same wave, which is exactly what a user under load feels.
  Run concurrency=1 to get the isolated TTFT.
* Every sample is kept, not just aggregates, so percentiles and outliers can be recomputed
  later without re-running the benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .vram import VramSampler
from .workloads import Workload, from_file, resolve

# --------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------


@dataclass
class Sample:
    """One completed (or failed) streaming request."""

    workload: str
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    token_source: str  # "usage" | "chunk-count"
    ttft_ms: float
    total_ms: float
    decode_tps: float
    e2e_tps: float
    error: str | None = None
    # ""length"" means the server hit --max-tokens and stopped early. Such a sample is
    # NOT a valid completion: the workload did not finish, so its timing is measuring a
    # truncated answer. See truncation_notes().
    finish_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.completion_tokens > 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class BenchConfig:
    url: str
    model: str
    api_key: str = "EMPTY"
    backend_label: str = "unnamed"
    engine: str = "unknown"
    workloads: list[Workload] = field(default_factory=list)
    concurrencies: list[int] = field(default_factory=lambda: [1])
    max_tokens_override: int | None = None
    temperature: float = 0.0
    runs: int = 3
    warmup: int = 1
    timeout_s: float = 600.0
    sample_vram: bool = False
    gpu_index: int = 0
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "model": self.model,
            "backend_label": self.backend_label,
            "engine": self.engine,
            "workloads": [w.name for w in self.workloads],
            "concurrencies": self.concurrencies,
            "max_tokens_override": self.max_tokens_override,
            "temperature": self.temperature,
            "runs": self.runs,
            "warmup": self.warmup,
            "sample_vram": self.sample_vram,
            **self.extra_meta,
        }


@dataclass
class BenchResult:
    config: BenchConfig
    samples: list[Sample]
    vram: dict[str, Any]
    wall_time_s: float
    harness_commit: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "inferlab.bench/1",
            "timestamp": self.timestamp,
            "harness_commit": self.harness_commit,
            "config": self.config.to_dict(),
            "vram": self.vram,
            "wall_time_s": self.wall_time_s,
            "samples": [asdict(s) for s in self.samples],
        }

    def aggregates(self) -> list[dict[str, Any]]:
        """One row per (workload, concurrency) pair, with percentiles over successful samples."""
        groups: dict[tuple[str, int], list[Sample]] = {}
        for s in self.samples:
            if s.ok:
                groups.setdefault((s.workload, s.concurrency), []).append(s)

        rows: list[dict[str, Any]] = []
        for (name, conc), items in sorted(groups.items()):
            ttft = [s.ttft_ms for s in items]
            decode = [s.decode_tps for s in items]
            e2e = [s.e2e_tps for s in items]
            rows.append(
                {
                    "workload": name,
                    "concurrency": conc,
                    "n": len(items),
                    "failed": sum(1 for s in self.samples if s.workload == name and not s.ok),
                    "ttft_ms": _stats(ttft),
                    "decode_tps": _stats(decode),
                    "e2e_tps": _stats(e2e),
                    "total_tokens": sum(s.completion_tokens for s in items),
                }
            )
        return rows


# --------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    n = len(values)
    out = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
    }
    if n >= 2:
        # quantiles(n=20)[-1] is the 95th percentile; n=100 variant is finer but noisier on n<20
        out["p95"] = statistics.quantiles(values, n=20, method="inclusive")[18]
    else:
        out["p95"] = values[0]
    return out


def _git_commit() -> str | None:
    """Best-effort harness commit. Provenance must never be the reason a benchmark fails."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


# --------------------------------------------------------------------------------------
# the actual request
# --------------------------------------------------------------------------------------


def _payload(cfg: BenchConfig, workload: Workload) -> dict[str, Any]:
    max_tokens = cfg.max_tokens_override or workload.max_tokens
    return {
        "model": cfg.model,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _stream_one(
    session: aiohttp.ClientSession,
    cfg: BenchConfig,
    workload: Workload,
    concurrency: int,
) -> Sample:
    url = cfg.url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    started = time.perf_counter()
    ttft_ms: float | None = None
    chunks = 0
    completion_tokens = 0
    prompt_tokens = 0
    token_source = "chunk-count"
    finish_reason: str | None = None
    error: str | None = None

    try:
        async with session.post(
            url,
            json=_payload(cfg, workload),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=cfg.timeout_s),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                return _failed_sample(workload, concurrency, f"HTTP {resp.status}: {body}")

            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # A usage-only chunk has an empty choices list.
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    token_source = "usage"
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                # finish_reason arrives on the last chunk. "length" means the server
                # stopped at the cap, not that the model finished -- treat those samples
                # as invalid, not as fast completions.
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    chunks += 1
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
    except TimeoutError:
        error = f"timeout after {cfg.timeout_s:.0f}s"
    except aiohttp.ClientError as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - one bad request must not abort the sweep
        error = f"{type(exc).__name__}: {exc}"

    total_ms = (time.perf_counter() - started) * 1000.0

    if error:
        return Sample(
            workload=workload.name,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            token_source=token_source,
            ttft_ms=total_ms,
            total_ms=total_ms,
            decode_tps=0.0,
            e2e_tps=0.0,
            error=error,
        )

    if token_source == "chunk-count":
        completion_tokens = chunks
    if prompt_tokens == 0:
        prompt_tokens = workload.approx_tokens

    if completion_tokens <= 0:
        return _failed_sample(workload, concurrency, "empty completion (stop at 0 tokens)")

    decode_ms = max(total_ms - (ttft_ms or total_ms), 0.001)
    return Sample(
        workload=workload.name,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_source=token_source,
        ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
        total_ms=total_ms,
        decode_tps=completion_tokens / (decode_ms / 1000.0),
        e2e_tps=completion_tokens / (total_ms / 1000.0),
        finish_reason=finish_reason,
    )


def _failed_sample(workload: Workload, concurrency: int, reason: str) -> Sample:
    return Sample(
        workload=workload.name,
        concurrency=concurrency,
        prompt_tokens=0,
        completion_tokens=0,
        token_source="none",
        ttft_ms=0.0,
        total_ms=0.0,
        decode_tps=0.0,
        e2e_tps=0.0,
        error=reason,
    )


async def _wave(
    session: aiohttp.ClientSession,
    cfg: BenchConfig,
    workload: Workload,
    concurrency: int,
) -> list[Sample]:
    tasks = [_stream_one(session, cfg, workload, concurrency) for _ in range(concurrency)]
    return list(await asyncio.gather(*tasks))


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


async def _ping(session: aiohttp.ClientSession, cfg: BenchConfig) -> str:
    """Return the first served model id. Fails loudly if the server is not reachable."""
    url = cfg.url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                raise RuntimeError(f"GET {url} returned HTTP {r.status}")
            data = await r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if not ids:
                raise RuntimeError("server returned no models")
            return ids[0]
    except aiohttp.ClientError as exc:
        raise RuntimeError(
            f"cannot reach an OpenAI-compatible server at {cfg.url} ({type(exc).__name__}). "
            "Start a backend first -- see backends/README.md"
        ) from exc


async def run_benchmark(cfg: BenchConfig) -> BenchResult:
    started = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=0)  # we control concurrency ourselves
    async with aiohttp.ClientSession(connector=connector) as session:
        served = await _ping(session, cfg)
        if cfg.model in ("auto", ""):
            cfg.model = served

        sampler = VramSampler(cfg.gpu_index) if cfg.sample_vram else None
        if sampler:
            sampler.start()

        samples: list[Sample] = []

        # Warmup: exercise the exact code path once per workload so CUDA graph capture,
        # kernel autotuning and page-table warmup are not charged to the measured runs.
        if cfg.warmup > 0:
            for workload in cfg.workloads:
                await _wave(session, cfg, workload, 1)

        for workload in cfg.workloads:
            for concurrency in cfg.concurrencies:
                for _ in range(cfg.runs):
                    samples.extend(await _wave(session, cfg, workload, concurrency))

        if sampler:
            sampler.stop()
        vram = sampler.summary() if sampler else {"enabled": False}

    return BenchResult(
        config=cfg,
        samples=samples,
        vram=vram,
        wall_time_s=time.perf_counter() - started,
        harness_commit=_git_commit(),
    )


# --------------------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------------------


def markdown_table(result: BenchResult) -> str:
    rows = result.aggregates()
    header = (
        "| workload | conc | n | failed | TTFT p50 (ms) | TTFT p95 (ms) | "
        "decode p50 (tok/s) | decode p95 (tok/s) | E2E p50 (tok/s) | tokens |"
    )
    sep = "|" + "---|" * 10
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| {workload} | {conc} | {n} | {failed} | {ttft50:.0f} | {ttft95:.0f} | "
            "{dec50:.1f} | {dec95:.1f} | {e2e50:.1f} | {tok} |".format(
                workload=r["workload"],
                conc=r["concurrency"],
                n=r["n"],
                failed=r["failed"],
                ttft50=r["ttft_ms"]["median"],
                ttft95=r["ttft_ms"]["p95"],
                dec50=r["decode_tps"]["median"],
                dec95=r["decode_tps"]["p95"],
                e2e50=r["e2e_tps"]["median"],
                tok=r["total_tokens"],
            )
        )
    return "\n".join(lines)


def truncation_notes(samples: Sequence[Sample]) -> list[str]:
    """Warn about samples the server stopped early.

    A truncated sample measures a partial answer. Its decode tok/s is still real
    arithmetic, but the workload did not finish -- so comparing it against a config that
    completed the same prompt is comparing two different problems.

    This exists because of a real-world observation: on GSM8K, raising the generation cap
    from 320 to 1024 tokens moved accuracy from 76.6% to 95.3%, almost entirely because
    answers were cut off mid-reasoning and then scored as wrong. Timing rows hide that.
    """
    counts: dict[tuple[str, int], int] = {}
    for s in samples:
        if s.truncated:
            key = (s.workload, s.concurrency)
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        return []

    lines = [
        "",
        f"TRUNCATION: {sum(counts.values())} sample(s) hit the token cap and did not finish.",
        "Per (workload, concurrency):",
    ]
    for (name, conc), n in sorted(counts.items()):
        lines.append(f"  {name} @ concurrency={conc}: {n} truncated")
    lines += [
        "",
        "  These rows are NOT comparable with rows that completed. Raise --max-tokens",
        "  (per-workload budgets live in inferlab/workloads.py) before quoting them.",
    ]
    return lines


def summary_json(result: BenchResult) -> dict[str, Any]:
    return {
        "backend": result.config.backend_label,
        "model": result.config.model,
        "url": result.config.url,
        "vram": result.vram,
        "aggregates": result.aggregates(),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inferlab bench",
        description="Benchmark an OpenAI-compatible LLM server.",
    )
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument("--model", default="auto", help="model id; 'auto' uses the first served model")
    p.add_argument("--api-key", default="EMPTY", help="Bearer token (vLLM accepts any non-empty)")
    p.add_argument("--backend", default="unnamed", help="label recorded with the results")
    p.add_argument(
        "--engine",
        default="unknown",
        choices=["vllm", "sglang", "tensorrt-llm", "llama.cpp", "unknown"],
        help="inference engine under test",
    )
    p.add_argument(
        "--workload",
        action="append",
        default=None,
        help="workload name, comma-separated list, or 'mixed'. Repeatable. "
        "Also accepts a path to a .txt/.json prompt file.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="named recipe from configs/default.yaml (e.g. vllm-nvfp4); "
        "CLI flags override recipe values",
    )
    p.add_argument(
        "--recipes",
        default=None,
        help="path to a recipe file (default: configs/default.yaml)",
    )
    p.add_argument(
        "--list-configs",
        action="store_true",
        help="print the available recipe names and exit",
    )
    p.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1],
        help="concurrency levels to sweep, e.g. --concurrency 1 8 32",
    )
    p.add_argument("--max-tokens", type=int, default=None, help="override per-workload budget")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3, help="measured runs per (workload, concurrency)")
    p.add_argument("--warmup", type=int, default=1, help="discarded warmup requests per workload")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    p.add_argument("--vram", action="store_true", help="sample peak VRAM with nvidia-smi")
    p.add_argument("--gpu", type=int, default=0, help="GPU index to sample when --vram is set")
    p.add_argument("--json", dest="json_out", default=None, help="write full results to this path")
    p.add_argument("--quiet", action="store_true", help="suppress the markdown table on stdout")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_configs:
        from . import recipes

        try:
            for name in recipes.names(args.recipes):
                print(recipes.describe(name, args.recipes))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    recipe: dict[str, Any] = {}
    if args.config:
        from . import recipes

        try:
            recipe = recipes.get(args.config, args.recipes)
        except KeyError as exc:
            print(str(exc).strip("'"), file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.workload:
        workloads: list[Workload] = []
        for item in args.workload:
            if Path(item).exists():
                workloads.append(from_file(item))
            else:
                workloads.extend(resolve(item.split(",")))
    else:
        workloads = resolve([str(recipe.get("workloads", "mixed"))])

    # Explicit CLI flags win; the recipe fills anything the user did not specify.
    cfg = BenchConfig(
        url=args.url,
        model=args.model,
        api_key=args.api_key,
        backend_label=args.backend if args.backend != "unnamed" else str(recipe.get("backend", "unnamed")),
        engine=args.engine if args.engine != "unknown" else str(recipe.get("engine", "unknown")),
        workloads=workloads,
        concurrencies=sorted(set(args.concurrency) | set(recipe.get("concurrency", [1]))),
        max_tokens_override=args.max_tokens,
        temperature=args.temperature if args.temperature != 0.0 else float(recipe.get("temperature", 0.0)),
        runs=max(1, args.runs if args.runs != 3 else int(recipe.get("runs", 3))),
        warmup=max(0, args.warmup if args.warmup != 1 else int(recipe.get("warmup", 1))),
        timeout_s=args.timeout if args.timeout != 600.0 else float(recipe.get("timeout_s", 600.0)),
        sample_vram=args.vram or bool(recipe.get("sample_vram", False)),
        gpu_index=args.gpu if args.gpu != 0 else int(recipe.get("gpu_index", 0)),
        extra_meta={k: v for k, v in recipe.items() if k in ("config", "description", "notes")},
    )

    result = asyncio.run(run_benchmark(cfg))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)

    if not args.quiet:
        print(f"\n# {result.config.backend_label} -- model={result.config.model}\n")
        print(markdown_table(result))
        vram = result.vram
        if vram.get("peak_mb") is not None:
            print(f"\nPeak VRAM: {vram['peak_mb']:.0f} MiB "
                  f"({vram['samples']} samples @ {vram['interval_s'] * 1000:.0f} ms)")
        elif vram.get("error"):
            print(f"\nVRAM sampling unavailable: {vram['error']}")
        failures = [s for s in result.samples if not s.ok]
        if failures:
            print(f"\n{len(failures)} request(s) failed; first error: {failures[0].error}")

        for line in truncation_notes(result.samples):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
