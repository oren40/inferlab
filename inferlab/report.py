"""Render benchmark JSON into the RESULTS.md table.

Results are generated, never transcribed. Hand-edited tables are how benchmarks quietly become
marketing, so `inferlab report` is the only supported path into docs/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HEADER = (
    "| Engine | Backend label | Workload | Model | Conc | n | TTFT p50 (ms) | TTFT p95 (ms) | "
    "Decode p50 (tok/s) | Decode p95 (tok/s) | E2E p50 (tok/s) | Peak VRAM (GB) |"
)
SEP = "|" + "---|" * 12

# Column order matches HEADER. Unknown engines render as-is.
ENGINE_ORDER = ["vllm", "sglang", "tensorrt-llm", "llama.cpp", "unknown"]


def is_synthetic(result: dict[str, Any]) -> bool:
    """True for pipeline-validation results, which must never enter the measured table."""
    cfg = result.get("config") or {}
    if cfg.get("synthetic"):
        return True
    # A quality result marks itself at the top level.
    return bool(result.get("synthetic"))


def partition(
    results: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (measured, synthetic). Synthetic rows name a mock server, not hardware."""
    measured = [r for r in results if not is_synthetic(r)]
    synthetic = [r for r in results if is_synthetic(r)]
    return measured, synthetic


def render_synthetic_note(synthetic: Sequence[dict[str, Any]]) -> str:
    """A hard, loud separator between real measurements and pipeline validation."""
    if not synthetic:
        return ""
    return "\n---\n\n## Pipeline verification (synthetic -- NOT a benchmark result)\n\n> **The rows below were produced against a mock OpenAI-compatible server in CI, on a CPU, with no model loaded. They exist to prove the harness, the quality scorer and this report generator actually run end to end. They are not measurements of any model, engine or GPU, and must not be quoted as such.**\n\nRun it yourself, no GPU required:\n\n```bash\ninferlab self-test\n```\n\nReal hardware results live in the table at the top of this file. When the top table is empty, nothing has been measured yet -- the section below is not a substitute for it.\n"

def load_results(paths: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(load_results([str(f) for f in sorted(path.glob("*.json"))]))
            continue
        if not path.exists():
            print(f"warning: {p} not found, skipping", file=sys.stderr)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"warning: {p} is not valid JSON ({exc}), skipping", file=sys.stderr)
            continue
        schema = data.get("schema")
        if schema in ("inferlab.bench/1", "inferlab.quality/1"):
            out.append(data)
        else:
            print(
                f"warning: {p} has schema {schema!r}, skipping "
                "(expected inferlab.bench/1 or inferlab.quality/1)",
                file=sys.stderr,
            )
    return out


def _rows_for(result: dict[str, Any]) -> list[list[str]]:
    cfg = result.get("config", {})
    engine = cfg.get("engine", "unknown")
    label = cfg.get("backend_label", "?")
    model = cfg.get("model", "?")
    vram = result.get("vram") or {}
    peak_gb = vram.get("peak_mb")
    peak_str = f"{peak_gb / 1024:.1f}" if peak_gb else "-"

    rows: list[list[str]] = []
    for agg in result.get("aggregates") or _aggregate(result):
        rows.append(
            [
                engine,
                label,
                agg["workload"],
                model,
                str(agg["concurrency"]),
                str(agg["n"]),
                f"{agg['ttft_ms']['median']:.0f}",
                f"{agg['ttft_ms']['p95']:.0f}",
                f"{agg['decode_tps']['median']:.1f}",
                f"{agg['decode_tps']['p95']:.1f}",
                f"{agg['e2e_tps']['median']:.1f}",
                peak_str,
            ]
        )
    return rows


def _aggregate(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute aggregates for older result files that were written without them."""
    import statistics

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in result.get("samples", []):
        if s.get("error") or not s.get("completion_tokens"):
            continue
        groups.setdefault((s["workload"], s["concurrency"]), []).append(s)

    def stats(key: str) -> dict[str, float]:
        vals = [s[key] for s in items]
        if len(vals) >= 2:
            p95 = statistics.quantiles(vals, n=20, method="inclusive")[18]
        else:
            p95 = vals[0] if vals else 0.0
        return {
            "mean": statistics.fmean(vals) if vals else 0.0,
            "median": statistics.median(vals) if vals else 0.0,
            "p95": p95,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    out = []
    for (name, conc), items in sorted(groups.items()):
        out.append(
            {
                "workload": name,
                "concurrency": conc,
                "n": len(items),
                "failed": 0,
                "ttft_ms": stats("ttft_ms"),
                "decode_tps": stats("decode_tps"),
                "e2e_tps": stats("e2e_tps"),
                "total_tokens": sum(s["completion_tokens"] for s in items),
            }
        )
    return out


def render(results: Sequence[dict[str, Any]]) -> str:
    """Render the measured-results table. Synthetic rows are dropped here by design."""
    measured, _synthetic = partition(results)

    all_rows: list[list[str]] = []
    for r in measured:
        all_rows.extend(_rows_for(r))

    def sort_key(row: list[str]) -> tuple[int, str, str, int]:
        try:
            engine_rank = ENGINE_ORDER.index(row[0])
        except ValueError:
            engine_rank = len(ENGINE_ORDER)
        return (engine_rank, row[1], row[2], int(row[4]))

    lines = [
        "# Results",
        "",
        "Every number below is generated by `inferlab report results/*.json` from the raw JSON in",
        "`results/`. Nothing here is hand-edited. See [METHODOLOGY.md](METHODOLOGY.md) for how each",
        "column is measured and what it does and does not tell you.",
        "",
        HEADER,
        SEP,
    ]
    if not all_rows:
        lines += [
            "| _no results yet_ | - | - | - | - | - | - | - | - | - | - | - |",
            "",
            "Nothing has been benchmarked on real hardware yet. Start a backend (see",
            "`backends/README.md`), then:",
            "",
            "```bash",
            "inferlab bench --backend <label> --engine vllm --json results/<label>.json",
            "inferlab report results/*.json --out docs/RESULTS.md",
            "```",
            "",
            "### The pipeline itself is verified",
            "",
            "The table above is empty because no GPU benchmark has been published -- not",
            "because the harness is untested. CI runs the whole pipeline (streaming parse,",
            "token counting, perplexity, accuracy scoring, report generation) against a",
            "deterministic synthetic server, and asserts that no synthetic row can ever",
            "reach this table. Reproduce it locally, no GPU required:",
            "",
            "```bash",
            "inferlab self-test",
            "```",
            "",
            "That produces numbers. It produces no evidence about any model, engine or GPU,",
            "and it is deliberately kept out of the table above.",
        ]
    else:
        lines += ["| " + " | ".join(row) + " |" for row in sorted(all_rows, key=sort_key)]

    lines += ["", "## Config provenance", "", "| Backend | URL | Runs | Warmup | Harness commit |", "|---|---|---|---|---|"]
    for r in measured:
        cfg = r.get("config", {})
        lines.append(
            "| {label} | {url} | {runs} | {warmup} | {commit} |".format(
                label=cfg.get("backend_label", "?"),
                url=cfg.get("url", "?"),
                runs=cfg.get("runs", "?"),
                warmup=cfg.get("warmup", "?"),
                commit=(r.get("harness_commit") or "-")[:8],
            )
        )

    lines += [
        "",
        "## Notes on reading this table",
        "",
        "- **Workload matters more than the headline.** A 300 tok/s number on `chat-short` and a",
        "  120 tok/s number on `long-context` are not in conflict -- they are different jobs. Only",
        "  compare rows with the same workload.",
        "- **TTFT p50 vs p95.** The median is the typical wait; p95 is the one that shows chunked",
        "  prefill and continuous batching doing their job. A p95 far above p50 under load means",
        "  head-of-line blocking.",
        "- **Decode tok/s.** `completion_tokens / (total_latency - TTFT)`, counted from the server's",
        "  `usage` field when available.",
        "- **Peak VRAM.** Sampled from `nvidia-smi` every 100 ms during the run. `-` means `--vram`",
        "  was not passed, not that VRAM was low.",
        "- **Concurrency.** A single-request number describes a demo. Only the higher-concurrency",
        "  rows describe a server.",
        "- **Quality.** Speed rows here say nothing about whether the config is still correct.",
        "  Cross-check every entry against the quality table below.",
        "",
    ]
    return "\n".join(lines) + "\n"


QUALITY_HEADER = (
    "| Backend | Model | Perplexity | Mean NLL (nats/tok) | Accuracy | n |"
)


def render_quality(results: Sequence[dict[str, Any]]) -> str:
    """Render the quality table. Synthetic rows are excluded by the same rule as bench."""
    _measured, _synthetic = partition(results)
    results = _measured
    lines = [
        "## Quality guardrails",
        "",
        "Perplexity is measured on a fixed corpus with `echo=True, max_tokens=0`. Accuracy is",
        "multiple-choice accuracy scored by option log-probability, so it is deterministic and",
        "comparable across engines. A config whose quality row is missing has not been validated.",
        "",
        QUALITY_HEADER,
        "|" + "---|" * 6,
    ]
    rows = 0
    for r in results:
        ppl = r.get("perplexity")
        acc = r.get("accuracy")
        if not (ppl or acc):
            continue
        rows += 1
        lines.append(
            "| {label} | {model} | {ppl} | {nll} | {acc} | {n} |".format(
                label=r.get("backend", "?"),
                model=r.get("model", "?"),
                ppl=f"{ppl['perplexity']:.4f}" if ppl else "-",
                nll=f"{ppl['mean_nll']:.4f}" if ppl else "-",
                acc=f"{acc['accuracy']:.3f}" if acc else "-",
                n=str(acc["total"]) if acc else "-",
            )
        )
    if not rows:
        lines.append("| _no quality runs yet_ | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inferlab report",
        description="Render benchmark results into docs/RESULTS.md.",
    )
    p.add_argument("paths", nargs="*", default=["results"], help="result JSON files or directories")
    p.add_argument("--out", default="docs/RESULTS.md", help="output markdown path")
    p.add_argument("--quality", nargs="*", default=None, help="quality result JSON files")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    bench = load_results(args.paths or ["results"])
    quality = load_results(args.quality) if args.quality else []

    _measured, synthetic = partition(bench)
    text = render(bench)
    if quality:
        text += render_quality(quality)

    # Synthetic rows are appended behind a hard separator and counted, never mixed in.
    note = render_synthetic_note(synthetic)
    if note:
        text += note
        print(
            f"note: {len(synthetic)} synthetic result(s) excluded from the measured table "
            "and listed under 'Pipeline verification'",
            file=sys.stderr,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(bench)} bench result(s), {len(quality)} quality result(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
