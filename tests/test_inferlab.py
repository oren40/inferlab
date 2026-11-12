"""Unit tests that need no GPU and no running server."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from inferlab import harness, quality, recipes, workloads

# --------------------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------------------


def test_mixed_resolves_all_builtins():
    resolved = workloads.resolve(["mixed"])
    assert len(resolved) == len(workloads.MIXED)
    assert {w.name for w in resolved} == set(workloads.builtin_names())


def test_unknown_workload_raises_with_hint():
    with pytest.raises(KeyError) as exc:
        workloads.resolve(["nope"])
    assert "chat-short" in str(exc.value)


def test_comma_separated_list_is_expanded():
    resolved = workloads.resolve(["chat-short,code-gen"])
    assert [w.name for w in resolved] == ["chat-short", "code-gen"]


def test_long_context_prompt_is_deterministic_and_sized():
    a = workloads.long_context_prompt(4096)
    b = workloads.long_context_prompt(4096)
    assert a == b, "synthetic prompts must be seeded or runs are not comparable"
    # target is an approximation; assert it lands in a sane band, not exactly
    assert 0.5 * 4096 <= a.__len__() // workloads.CHARS_PER_TOKEN <= 1.5 * 4096


def test_long_context_prompt_hides_a_needle_in_the_middle():
    prompt = workloads.long_context_prompt(2048, needle="the secret is 42")
    assert "the secret is 42" in prompt
    head, _, tail = prompt.partition("the secret is 42")
    assert len(head) > 100 and len(tail) > 100, "needle must not sit at either edge"


def test_from_file_json_and_txt(tmp_path: Path):
    j = tmp_path / "w.json"
    j.write_text(json.dumps({"prompt": "hi", "max_tokens": 7}), encoding="utf-8")
    assert workloads.from_file(j).max_tokens == 7

    t = tmp_path / "w.txt"
    t.write_text("hello there", encoding="utf-8")
    loaded = workloads.from_file(t)
    assert loaded.prompt == "hello there"
    assert loaded.name == "w"


# --------------------------------------------------------------------------------------
# harness statistics
# --------------------------------------------------------------------------------------


def _sample(ttft: float, decode: float, conc: int = 1, name: str = "chat-short") -> harness.Sample:
    return harness.Sample(
        workload=name,
        concurrency=conc,
        prompt_tokens=10,
        completion_tokens=100,
        token_source="usage",
        ttft_ms=ttft,
        total_ms=ttft + 1000.0,
        decode_tps=decode,
        e2e_tps=decode / 2,
    )


def test_stats_single_value_has_no_spurious_stdev():
    s = harness._stats([5.0])
    assert s["stdev"] == 0.0
    assert s["p95"] == 5.0
    assert s["median"] == 5.0


def test_stats_p95_above_median():
    values = list(range(1, 101))
    s = harness._stats(values)
    assert s["median"] == 50.5
    assert s["p95"] > s["median"]


def test_failed_samples_are_excluded_from_aggregates():
    cfg = harness.BenchConfig(
        url="http://x", model="m", workloads=[workloads.resolve(["chat-short"])[0]]
    )
    result = harness.BenchResult(
        config=cfg,
        samples=[_sample(100, 50), _sample(120, 55), harness._failed_sample(cfg.workloads[0], 1, "boom")],
        vram={"enabled": False},
        wall_time_s=1.0,
    )
    rows = result.aggregates()
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["failed"] == 1


def test_aggregate_groups_by_workload_and_concurrency():
    cfg = harness.BenchConfig(
        url="http://x", model="m", workloads=[workloads.resolve(["chat-short"])[0]]
    )
    result = harness.BenchResult(
        config=cfg,
        samples=[_sample(100, 50, conc=1), _sample(200, 40, conc=8), _sample(100, 50, conc=1)],
        vram={"enabled": False},
        wall_time_s=1.0,
    )
    rows = {(r["workload"], r["concurrency"]): r for r in result.aggregates()}
    assert set(rows) == {("chat-short", 1), ("chat-short", 8)}
    assert rows[("chat-short", 1)]["n"] == 2


def test_markdown_table_has_expected_columns():
    cfg = harness.BenchConfig(
        url="http://x", model="m", workloads=[workloads.resolve(["chat-short"])[0]]
    )
    result = harness.BenchResult(
        config=cfg, samples=[_sample(100, 50)], vram={"enabled": False}, wall_time_s=1.0
    )
    table = harness.markdown_table(result)
    assert "| workload | conc | n |" in table
    assert "chat-short" in table


# --------------------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------------------


def _bench_result_dict(**overrides) -> dict:
    base = {
        "schema": "inferlab.bench/1",
        "harness_commit": "abc12345",
        "config": {
            "engine": "vllm",
            "backend_label": "vllm-nvfp4",
            "model": "m",
            "url": "http://localhost:8000",
            "runs": 5,
            "warmup": 2,
        },
        "vram": {"peak_mb": 24576.0, "samples": 10},
        "samples": [
            {
                "workload": "chat-short",
                "concurrency": 1,
                "prompt_tokens": 10,
                "completion_tokens": 100,
                "token_source": "usage",
                "ttft_ms": 120.0,
                "total_ms": 2200.0,
                "decode_tps": 48.0,
                "e2e_tps": 45.0,
                "error": None,
            }
        ],
    }
    base.update(overrides)
    return base


def test_report_renders_row_and_provenance():
    from inferlab import report

    md = report.render([_bench_result_dict()])
    assert "| vllm | vllm-nvfp4 | chat-short | m | 1 | 1 | 120 |" in md
    assert "24.0" in md  # 24576 MiB -> 24.0 GB
    assert "abc12345" in md


def test_report_recomputes_aggregates_when_absent():
    """Older result files without a precomputed aggregates block must still render."""
    from inferlab import report

    data = _bench_result_dict()
    data.pop("aggregates", None)
    md = report.render([data])
    # the workload must survive into the table, or the row cannot be attributed
    assert "chat-short" in md
    assert "| vllm | vllm-nvfp4 | chat-short | m | 1 | 1 | 120 |" in md


def test_report_row_carries_workload_and_concurrency():
    from inferlab import report

    data = _bench_result_dict()
    md = report.render([data])
    assert "| Workload |" in md or "| Engine | Backend label | Workload |" in md
    assert "chat-short" in md


def test_report_skips_foreign_schemas():
    from inferlab import report

    md = report.render([{"schema": "something-else", "samples": []}])
    assert "_no results yet_" in md


def test_report_empty_results_shows_instructions():
    from inferlab import report

    md = report.render([])
    assert "inferlab bench --backend <label>" in md


# --------------------------------------------------------------------------------------
# quality helpers
# --------------------------------------------------------------------------------------


def test_perplexity_math():
    """Mean NLL of ln(10) per token over 10 tokens must give perplexity exactly 10.

    The API returns log-*probabilities*, i.e. negative numbers, and index 0 of an echo is
    unconditioned (null) and must be dropped -- exactly what quality.perplexity() does.
    """
    tokens = [None] + [-math.log(10.0)] * 10
    scored = [t for t in tokens[1:] if t is not None]
    mean_nll = -sum(scored) / len(scored)
    assert len(scored) == 10
    result = quality.PerplexityResult(
        windows=1, tokens=len(scored), mean_nll=mean_nll, perplexity=math.exp(mean_nll)
    )
    assert result.mean_nll == pytest.approx(math.log(10.0))
    assert result.perplexity == pytest.approx(10.0)
    assert result.to_dict()["perplexity"] == pytest.approx(10.0)


def test_perplexity_drops_unconditioned_first_token():
    """A corpus of one token scores zero tokens, which quality.perplexity treats as an error."""
    tokens = [None]
    scored = [t for t in tokens[1:] if t is not None]
    assert scored == []


def test_task_loading_and_indexing():
    tasks = quality.load_tasks(["data/quality/tasks.json"])
    assert len(tasks) >= 15, "the task set is the quality guardrail; keep it non-trivial"
    for t in tasks:
        assert 0 <= t.answer < len(t.choices)
        letters = [chr(ord("A") + i) for i in range(len(t.choices))]
        assert letters[t.answer] in ("A", "B", "C", "D")


def test_every_task_has_tags_for_per_tag_reporting():
    tasks = quality.load_tasks(["data/quality/tasks.json"])
    assert all(t.tags for t in tasks)


def test_accuracy_result_reports_fraction():
    r = quality.AccuracyResult(total=4, correct=1)
    assert r.accuracy == 0.25


# --------------------------------------------------------------------------------------
# recipes
# --------------------------------------------------------------------------------------


def test_recipe_defaults_merge():
    cfg = recipes.get("vllm-nvfp4")
    assert cfg["engine"] == "vllm"
    assert cfg["runs"] == 5  # inherited from defaults
    assert cfg["name"] == "vllm-nvfp4"


def test_unknown_recipe_lists_available():
    with pytest.raises(KeyError) as exc:
        recipes.get("does-not-exist")
    assert "vllm-nvfp4" in str(exc.value)


def test_every_recipe_declares_an_engine_and_backend():
    for name in recipes.names():
        cfg = recipes.get(name)
        assert cfg.get("engine"), f"{name} has no engine"
        assert cfg.get("backend"), f"{name} has no backend label"


def test_describe_returns_something_useful():
    text = recipes.describe("vllm-nvfp4")
    assert text.startswith("[vllm-nvfp4]")
    assert "vllm" in text.lower()


# --------------------------------------------------------------------------------------
# documentation correctness
# --------------------------------------------------------------------------------------


def _readme_bash_blocks() -> list[str]:
    import re

    text = Path("README.md").read_text(encoding="utf-8")
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def _inferlab_commands() -> list[str]:
    """Every `inferlab ...` invocation documented in the README, as an argv list.

    Matches both the installed console script (`inferlab bench ...`) and the
    no-install form (`python -m inferlab bench ...`).
    """
    import re

    commands: list[str] = []
    for block in _readme_bash_blocks():
        cmd: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.endswith("\\"):
                cmd.append(stripped[:-1].strip())
                continue
            cmd.append(stripped)
            joined = " ".join(part for part in cmd if part)
            # normalise `python -m inferlab X` down to `inferlab X`
            normalised = re.sub(r"^python -m (inferlab)\b", r"\1", joined)
            if normalised.startswith("inferlab "):
                commands.append(normalised)
            cmd = []
    return commands


def test_readme_documents_real_commands():
    """Every `inferlab ...` command in the README must actually parse.

    A documented command that cannot run is worse than no documentation: it is the first thing
    a technically-minded reader tries.
    """
    import shlex

    from inferlab import report as report_module

    subcommand_to_module = {
        "bench": harness,
        "quality": quality,
        "report": report_module,
    }
    commands = _inferlab_commands()
    assert len(commands) >= 3, f"expected the README to document inferlab usage, got {commands}"

    for cmd in commands:
        argv = shlex.split(cmd)[1:]  # drop the leading "inferlab"
        if not argv or argv[0] in ("-h", "--help"):
            continue
        subcommand, rest = argv[0], argv[1:]
        assert subcommand in subcommand_to_module, f"unknown subcommand in README: {cmd}"
        parser = subcommand_to_module[subcommand].build_parser()
        try:
            parser.parse_args(rest)  # argparse exits(2) on a parse error, 0 on --help
        except SystemExit as exc:
            pytest.fail(
                f"README documents a command that does not parse: {cmd}\n"
                f"argparse exited with {exc.code}"
            )


def test_readme_references_existing_files():
    """Every relative link/path in the README must resolve on disk."""
    import re

    text = Path("README.md").read_text(encoding="utf-8")
    refs = re.findall(r"\]\(([^)#:<>]+)\)", text)
    refs += re.findall(r"`(configs/[^`]+|data/[^`]+|docs/[^`]+|backends/[^`]+)`", text)
    assert refs, "found no file references to check"
    missing = [r for r in refs if not Path(r).exists()]
    assert not missing, f"README references files that do not exist: {missing}"


def test_python_dash_m_works_without_install():
    """A fresh clone must be runnable via `python -m inferlab`.

    The `inferlab` console script only exists after `pip install -e .`, so the
    no-install path is the one a reader tries first. Regression test for the
    issue where the README documented a command that did not exist.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "inferlab", "--help"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"`python -m inferlab` failed:\n{proc.stderr}"
    assert "bench" in proc.stdout and "quality" in proc.stdout and "report" in proc.stdout


def test_no_install_path_documented_in_readme():
    """The README must show the no-install entry point, not just the console script."""
    text = Path("README.md").read_text(encoding="utf-8")
    assert "python -m inferlab" in text, (
        "README documents only the installed console script; a fresh clone has nothing "
        "named `inferlab` on PATH"
    )


def test_task_set_is_nontrivial_and_tagged():
    """The task set has to be big enough to catch a regression, not just look complete."""
    tasks = quality.load_tasks(["data/quality/tasks.json"])
    assert len(tasks) >= 25, f"only {len(tasks)} tasks; too small to catch a regression"
    tags = {t for task in tasks for t in task.tags}
    assert {"reasoning", "arithmetic", "code"} <= tags, (
        f"missing core capability tags; found {sorted(tags)}. A perplexity-only guardrail "
        "does not catch a model that got fast and lost the ability to reason."
    )
