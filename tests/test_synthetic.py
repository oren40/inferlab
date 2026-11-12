"""Tests for the synthetic backend and the synthetic/measured separation.

The point of the mock is to prove the pipeline runs without a GPU. The point of these
tests is to prove the mock can never be mistaken for a measurement.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from inferlab import harness as _harness
from inferlab import report as _report
from inferlab import testing as _testing
from inferlab.workloads import resolve


def test_synthetic_results_are_partitioned_from_measured():
    fake_bench = {
        "schema": "inferlab.bench/1",
        "config": {"backend_label": "synthetic-x", "synthetic": True, "engine": "unknown"},
        "samples": [],
        "vram": {"enabled": False},
    }
    real_bench = {
        "schema": "inferlab.bench/1",
        "config": {"backend_label": "vllm-nvfp4", "engine": "vllm"},
        "samples": [],
        "vram": {"enabled": False},
    }
    measured, synthetic = _report.partition([fake_bench, real_bench])
    assert [r["config"]["backend_label"] for r in measured] == ["vllm-nvfp4"]
    assert [r["config"]["backend_label"] for r in synthetic] == ["synthetic-x"]


def test_quality_results_are_recognised_as_synthetic():
    payload = {"schema": "inferlab.quality/1", "synthetic": True, "backend": "synthetic-x"}
    measured, synthetic = _report.partition([payload])
    assert measured == []
    assert len(synthetic) == 1


def test_synthetic_rows_never_reach_the_measured_table():
    """The single most important test in this file."""
    fake = {
        "schema": "inferlab.bench/1",
        "harness_commit": "deadbee",
        "config": {
            "engine": "unknown",
            "backend_label": "synthetic-pipeline-check",
            "synthetic": True,
            "url": "http://127.0.0.1:0",
            "runs": 2,
            "warmup": 1,
        },
        "vram": {"peak_mb": 0.0, "samples": 0},
        "samples": [
            {
                "workload": "chat-short",
                "concurrency": 1,
                "prompt_tokens": 10,
                "completion_tokens": 64,
                "token_source": "usage",
                "ttft_ms": 1.0,
                "total_ms": 100.0,
                "decode_tps": 640.0,
                "e2e_tps": 640.0,
                "error": None,
            }
        ],
    }
    md = _report.render([fake])
    assert "synthetic-pipeline-check" not in md, (
        "a synthetic row leaked into the measured-results table"
    )
    assert "_no results yet_" in md

    qmd = _report.render_quality(
        [{"schema": "inferlab.quality/1", "synthetic": True, "backend": "synthetic-x",
          "model": "m", "perplexity": {"perplexity": 1.0, "mean_nll": 0.0}}]
    )
    assert "synthetic-x" not in qmd


def test_synthetic_note_is_loud_and_mentions_self_test():
    fake_bench = {
        "schema": "inferlab.bench/1",
        "config": {"backend_label": "synthetic-x", "synthetic": True},
        "samples": [],
        "vram": {"enabled": False},
    }
    note = _report.render_synthetic_note([fake_bench])
    assert "NOT a benchmark result" in note
    assert "inferlab self-test" in note
    assert "not measurements" in note

    assert _report.render_synthetic_note([]) == "", "clean runs must get no separator"


def test_synthetic_backend_serves_models():
    async def go():
        async with _testing.SyntheticBackend() as (_, url):
            import aiohttp

            async with aiohttp.ClientSession() as s, s.get(f"{url}/v1/models") as r:
                data = await r.json()
            return data

    data = asyncio.run(go())
    assert data["data"][0]["id"] == _testing.DEFAULT_MODEL


def test_synthetic_backend_streams_usage_and_role_delta():
    """The harness measures past a role-only delta and prefers server-reported usage."""
    async def go():
        async with _testing.SyntheticBackend() as (_, url):
            cfg = _harness.BenchConfig(
                url=url,
                model="auto",
                workloads=resolve(["chat-short"]),
                concurrencies=[1],
                runs=1,
                warmup=0,
            )
            return await _harness.run_benchmark(cfg)

    result = asyncio.run(go())
    assert result.samples, "synthetic backend produced no samples"
    s = result.samples[0]
    assert s.ok, f"synthetic request failed: {s.error}"
    # usage-based counting, not chunk counting
    assert s.token_source == "usage", s.token_source
    assert s.completion_tokens > 0
    assert s.finish_reason == "stop"
    assert not s.truncated


def test_synthetic_backend_can_force_truncation():
    """The mock must be able to exercise the truncation guard, or CI never covers it."""
    async def go():
        cfg = _testing.SyntheticConfig(truncate=True)
        async with _testing.SyntheticBackend(cfg) as (_, url):
            bcfg = _harness.BenchConfig(
                url=url,
                model="auto",
                workloads=resolve(["chat-short"]),
                concurrencies=[1],
                runs=1,
                warmup=0,
                max_tokens_override=8,
            )
            return await _harness.run_benchmark(bcfg)

    result = asyncio.run(go())
    assert result.samples
    assert all(s.truncated for s in result.samples), (
        "expected every sample to be truncated when the mock is told to cap"
    )
    notes = _harness.truncation_notes(result.samples)
    assert notes, "truncated samples produced no warning"


def test_synthetic_backend_scores_quality():
    async def go():
        import aiohttp

        async with _testing.SyntheticBackend() as (_, url), aiohttp.ClientSession() as s:
            from inferlab import quality

            p = await quality.perplexity(s, url=url, model="m", api_key="EMPTY", text="hello world " * 50)
            a = await quality.accuracy(
                s, url=url, model="m", api_key="EMPTY",
                tasks=quality.load_tasks(["data/quality/tasks.json"]),
            )
            return p, a

    ppl, acc = asyncio.run(go())
    assert ppl.tokens > 0, "perplexity path scored no tokens"
    assert ppl.perplexity > 0
    assert acc.total == 26, f"expected all 26 tasks scored, got {acc.total}"


def test_pipeline_check_runs_end_to_end(tmp_path: Path):
    outdir = str(tmp_path / "syn")
    report_out = str(tmp_path / "SELF-TEST.md")
    result = asyncio.run(
        _testing.run_pipeline_check(outdir=outdir, report_out=report_out)
    )
    assert result.n_bench_samples > 0
    assert Path(result.bench_path or "").exists()
    assert Path(result.quality_path or "").exists()
    assert Path(report_out).exists()

    bench = json.loads(Path(result.bench_path or "").read_text(encoding="utf-8"))
    assert bench["config"]["synthetic"] is True

    text = Path(report_out).read_text(encoding="utf-8")
    assert "Pipeline verification" in text
    assert "NOT a benchmark result" in text
    # The mock must not appear as a measured row
    assert text.count("| _no results yet_ |") >= 1
