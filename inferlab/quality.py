"""Quality guardrails for quantized inference.

Speed is the easy half. A config that serves 300 tok/s and answers arithmetic wrong is a
regression, not an optimization, so every optimization in this lab has to clear the same two
checks before its speed number is quoted:

* **perplexity delta** -- mean per-token NLL on a fixed corpus, compared against the BF16
  reference. A delta of 0.00 nats/token is free; 0.05+ on a 27B model is where quality work
  starts noticing.
* **task accuracy** -- multiple-choice accuracy scored by *log-probability of each option*, not
  by asking the model to emit a letter and parsing it. Log-prob scoring is deterministic and
  comparable across engines, quantization schemes and temperatures; free-form parsing is not.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

# --------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------


@dataclass
class PerplexityResult:
    windows: int
    tokens: int
    mean_nll: float
    perplexity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "tokens": self.tokens,
            "mean_nll": self.mean_nll,
            "perplexity": self.perplexity,
        }


@dataclass
class TaskItem:
    id: str
    question: str
    choices: list[str]
    answer: int
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskItem:
        return cls(
            id=str(d["id"]),
            question=str(d["question"]),
            choices=[str(c) for c in d["choices"]],
            answer=int(d["answer"]),
            tags=tuple(d.get("tags", ())),
        )


@dataclass
class AccuracyResult:
    total: int
    correct: int
    per_tag: dict[str, dict[str, int]] = field(default_factory=dict)
    misses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "per_tag": self.per_tag,
            "misses": self.misses,
        }


# --------------------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------------------


async def perplexity(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    api_key: str,
    text: str,
    window_chars: int = 1800,
    timeout_s: float = 300.0,
) -> PerplexityResult:
    """Mean per-token negative log-likelihood over a corpus.

    The corpus is evaluated in overlapping-free character windows because servers cap the
    prompt at max_model_len; window boundaries mean the first token of each window is
    conditioned on less context than it deserves, so it is dropped (echo returns a null
    log-prob for it anyway).
    """
    url = url.rstrip("/") + "/v1/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    windows = _window(text, window_chars)
    total_nll = 0.0
    total_tokens = 0

    for chunk in windows:
        payload = {
            "model": model,
            "prompt": chunk,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise RuntimeError(f"perplexity request failed: HTTP {resp.status}: {body}")
            data = await resp.json()

        logprobs = (data.get("choices") or [{}])[0].get("logprobs") or {}
        token_logprobs = logprobs.get("token_logprobs") or []
        for lp in token_logprobs[1:]:  # index 0 is unconditioned / null
            if lp is not None:
                total_nll += -float(lp)
                total_tokens += 1

    if total_tokens == 0:
        raise RuntimeError("no tokens scored; check that the corpus is non-empty and echo is supported")

    mean_nll = total_nll / total_tokens
    return PerplexityResult(
        windows=len(windows),
        tokens=total_tokens,
        mean_nll=mean_nll,
        perplexity=math.exp(min(mean_nll, 700.0)),
    )


def _window(text: str, size: int) -> list[str]:
    """Split into sentence-aligned chunks so window edges land on punctuation."""
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sentence in _sentences(text):
        if buf_len + len(sentence) > size and buf:
            out.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sentence)
        buf_len += len(sentence) + 1
    if buf:
        out.append(" ".join(buf))
    return [c for c in out if c.strip()]


def _sentences(text: str) -> list[str]:
    parts = text.replace("\n", " ").split(". ")
    return [p if p.endswith(".") else p + "." for p in parts if p.strip()]


# --------------------------------------------------------------------------------------
# multiple-choice accuracy via log-probability scoring
# --------------------------------------------------------------------------------------


async def _choice_logprobs(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    api_key: str,
    prompt: str,
    letters: Sequence[str],
    timeout_s: float = 120.0,
) -> dict[str, float]:
    """Return P(letter) for each candidate letter at the position right after `prompt`."""
    endpoint = url.rstrip("/") + "/v1/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
        "echo": False,
    }
    async with session.post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as resp:
        if resp.status != 200:
            body = (await resp.text())[:300]
            raise RuntimeError(f"scoring request failed: HTTP {resp.status}: {body}")
        data = await resp.json()

    logprobs = (data.get("choices") or [{}])[0].get("logprobs") or {}
    top = (logprobs.get("top_logprobs") or [{}])[-1]
    scores: dict[str, float] = {}
    for letter in letters:
        raw = None
        for tok, lp in top.items():
            if tok.strip().strip(".").upper() == letter.upper():
                raw = lp
                break
        scores[letter] = -1e9 if raw is None else float(raw)
    return scores


async def accuracy(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    api_key: str,
    tasks: Sequence[TaskItem],
    timeout_s: float = 120.0,
) -> AccuracyResult:
    letters = [chr(ord("A") + i) for i in range(26)]

    total = 0
    correct = 0
    per_tag: dict[str, dict[str, int]] = {}
    misses: list[dict[str, Any]] = []

    for task in tasks:
        prompt = f"{task.question}\n\nOptions:\n" + "\n".join(
            f"{letters[i]}. {text}" for i, text in enumerate(task.choices)
        )
        prompt += "\n\nAnswer:"
        try:
            scores = await _choice_logprobs(
                session,
                url=url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                letters=letters[: len(task.choices)],
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - a bad task should not abort the eval
            misses.append({"id": task.id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        predicted = max(scores, key=lambda k: scores[k])
        expected = letters[task.answer]
        total += 1
        ok = predicted == expected
        if ok:
            correct += 1
        else:
            misses.append(
                {
                    "id": task.id,
                    "expected": expected,
                    "predicted": predicted,
                    "tags": list(task.tags),
                }
            )

        for tag in task.tags:
            bucket = per_tag.setdefault(tag, {"total": 0, "correct": 0})
            bucket["total"] += 1
            bucket["correct"] += 1 if ok else 0

    return AccuracyResult(total=total, correct=correct, per_tag=per_tag, misses=misses)


def load_tasks(paths: Sequence[str]) -> list[TaskItem]:
    items: list[TaskItem] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("tasks", [])
        items.extend(TaskItem.from_dict(d) for d in data)
    if not items:
        raise ValueError("no tasks loaded")
    return items


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inferlab quality",
        description="Measure the quality cost of a quantization / engine config.",
    )
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="auto")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--backend", default="unnamed", help="label recorded with the results")
    p.add_argument("--perplexity-corpus", default=None, help="path to a .txt corpus")
    p.add_argument("--window-chars", type=int, default=1800)
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="paths to task JSON files, or a preset name",
    )
    p.add_argument(
        "--baseline-json",
        default=None,
        help="a previous quality JSON to compute deltas against",
    )
    p.add_argument("--json", dest="json_out", default=None, help="write results to this path")
    p.add_argument("--timeout", type=float, default=300.0)
    return p


DEFAULT_TASKS = "data/quality/tasks.json"


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    task_paths = [t for t in (args.tasks or []) if Path(t).exists()]
    presets = [t for t in (args.tasks or []) if not Path(t).exists()]
    if (
        not task_paths
        and (not presets or "default" in presets)
        and Path(DEFAULT_TASKS).exists()
    ):
        task_paths = [DEFAULT_TASKS]

    async with aiohttp.ClientSession() as session:
        model = args.model
        if model in ("auto", ""):
            async with session.get(
                args.url.rstrip("/") + "/v1/models",
                headers={"Authorization": f"Bearer {args.api_key}"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                data = await r.json()
                model = next(m["id"] for m in data.get("data", []) if m.get("id"))

        result: dict[str, Any] = {
            "schema": "inferlab.quality/1",
            "backend": args.backend,
            "model": model,
            "url": args.url,
        }

        if args.perplexity_corpus:
            text = Path(args.perplexity_corpus).read_text(encoding="utf-8")
            ppl = await perplexity(
                session,
                url=args.url,
                model=model,
                api_key=args.api_key,
                text=text,
                window_chars=args.window_chars,
                timeout_s=args.timeout,
            )
            result["perplexity"] = ppl.to_dict()

        if task_paths:
            acc = await accuracy(
                session,
                url=args.url,
                model=model,
                api_key=args.api_key,
                tasks=load_tasks(task_paths),
                timeout_s=args.timeout,
            )
            result["accuracy"] = acc.to_dict()

        if not (args.perplexity_corpus or task_paths):
            raise SystemExit(
                "nothing to do: pass --perplexity-corpus and/or --tasks (expected a task JSON "
                f"such as {DEFAULT_TASKS})"
            )

    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        result["baseline"] = {
            "backend": baseline.get("backend"),
            "model": baseline.get("model"),
        }
        if "perplexity" in baseline and "perplexity" in result:
            result["delta"] = {
                "mean_nll": result["perplexity"]["mean_nll"] - baseline["perplexity"]["mean_nll"],
                "perplexity": result["perplexity"]["perplexity"]
                - baseline["perplexity"]["perplexity"],
            }
        if "accuracy" in baseline and "accuracy" in result:
            delta = dict(result.get("delta", {}))
            delta["accuracy"] = result["accuracy"]["accuracy"] - baseline["accuracy"]["accuracy"]
            result["delta"] = delta

    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(_run(args))

    print(f"\n# quality -- {result['backend']} (model={result['model']})\n")
    if "perplexity" in result:
        p = result["perplexity"]
        print(f"perplexity : {p['perplexity']:.4f}  (mean NLL {p['mean_nll']:.4f} nats/token, "
              f"{p['tokens']} tokens over {p['windows']} windows)")
    if "accuracy" in result:
        a = result["accuracy"]
        print(f"accuracy   : {a['accuracy']:.3f}  ({a['correct']}/{a['total']})")
        for tag, bucket in sorted(a["per_tag"].items()):
            print(f"  {tag:<12} {bucket['correct']}/{bucket['total']}")
        if a["misses"]:
            print(f"  misses: {len(a['misses'])} (see JSON for detail)")
    if "delta" in result:
        print("\ndeltas vs baseline:")
        for k, v in result["delta"].items():
            print(f"  {k:<12} {v:+.4f}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
