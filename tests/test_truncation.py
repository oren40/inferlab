"""Tests for the truncation guard.

Driven by a real observation from the launch thread: raising a generation cap from
320 to 1024 tokens moved GSM8K accuracy from 76.6% to 95.3%, almost entirely because
answers were cut off mid-reasoning and then scored as wrong.

A harness that ignores `finish_reason` cannot see that failure mode. These tests make
sure the guard stays visible.
"""
from __future__ import annotations

import json

from inferlab import harness, workloads


def _sample(ttft: float = 100.0, decode: float = 50.0, conc: int = 1) -> harness.Sample:
    return harness.Sample(
        workload="chat-short",
        concurrency=conc,
        prompt_tokens=10,
        completion_tokens=100,
        token_source="usage",
        ttft_ms=ttft,
        total_ms=ttft + 1000.0,
        decode_tps=decode,
        e2e_tps=decode / 2,
    )


def _result(samples: list[harness.Sample]) -> harness.BenchResult:
    cfg = harness.BenchConfig(
        url="http://x",
        model="m",
        workloads=[workloads.resolve(["chat-short"])[0]],
    )
    return harness.BenchResult(
        config=cfg, samples=samples, vram={"enabled": False}, wall_time_s=1.0
    )


def test_truncated_flag_reflects_finish_reason():
    complete = _sample()
    complete.finish_reason = "stop"
    truncated = _sample()
    truncated.finish_reason = "length"
    unknown = _sample()
    unknown.finish_reason = None

    assert complete.truncated is False
    assert truncated.truncated is True
    assert unknown.truncated is False


def test_truncation_notes_are_emitted_only_when_needed():
    complete = _sample()
    complete.finish_reason = "stop"
    unknown = _sample()

    assert harness.truncation_notes([complete, unknown]) == [], (
        "clean runs must not emit truncation warnings -- a guard that always fires trains "
        "people to ignore it"
    )

    truncated = _sample()
    truncated.finish_reason = "length"
    notes = harness.truncation_notes([complete, truncated, truncated])
    assert notes, "truncated samples produced no warning at all"

    joined = "\n".join(notes)
    assert "2" in joined, f"expected the truncated count in the warning, got:\n{joined}"
    assert "chat-short" in joined
    assert "NOT comparable" in joined
    assert "--max-tokens" in joined, "the warning must say how to fix it"


def test_truncation_notes_break_down_by_workload_and_concurrency():
    a = _sample(conc=1)
    a.finish_reason = "length"
    b = _sample(conc=32)
    b.finish_reason = "length"
    c = _sample(conc=8)
    c.finish_reason = "length"
    c.workload = "long-context"

    joined = "\n".join(harness.truncation_notes([a, b, c]))
    assert "concurrency=1" in joined
    assert "concurrency=32" in joined
    assert "long-context @ concurrency=8" in joined
    assert "TRUNCATION: 3" in joined


def test_finish_reason_survives_json_roundtrip():
    """finish_reason must reach the results file, or the warning only exists in memory."""
    truncated = _sample()
    truncated.finish_reason = "length"
    payload = json.loads(json.dumps(_result([truncated]).to_dict()))
    assert payload["samples"][0]["finish_reason"] == "length"


def test_aggregates_still_exclude_failed_samples():
    """Truncation is not the same as failure: a truncated sample still counts as ok."""
    truncated = _sample()
    truncated.finish_reason = "length"
    failed = harness._failed_sample(workloads.resolve(["chat-short"])[0], 1, "boom")

    assert truncated.ok is True, "a truncated sample is not a failed request"
    assert failed.ok is False

    rows = _result([truncated, failed]).aggregates()
    assert len(rows) == 1
    assert rows[0]["n"] == 1
    assert rows[0]["failed"] == 1


def test_workload_budgets_are_documented_per_workload():
    """The truncation warning points at per-workload budgets; those must be explicit."""
    for w in workloads.resolve(["mixed"]):
        assert w.max_tokens > 0, f"{w.name} has no max_tokens budget to raise"
        # A budget under 128 cannot answer a question of any length
        assert w.max_tokens >= 128, (
            f"{w.name} has a max_tokens budget of {w.max_tokens}, which is small enough "
            "that answers will truncate and be scored as failures"
        )
