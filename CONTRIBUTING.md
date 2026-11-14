# Contributing

The most useful contribution is a benchmark run on hardware this repo has not seen.
The second most useful is closing one of the two open gaps below.

**[The number one rule](#results-are-generated-never-transcribed) is at the top for a
reason.** Everything else in this file is negotiable; that one is not.

---

## Before you open a PR

```bash
pip install -e ".[dev]"
ruff check .        # must be clean
pytest -q           # 45 tests, no GPU required
inferlab self-test  # full pipeline against a synthetic server
```

`pytest` and `self-test` need no GPU, so a contributor on a laptop can validate the
whole change before touching hardware.

---

## Results are generated, never transcribed

`docs/RESULTS.md` is generated from `results/*.json` by `inferlab report`. It is never
hand-edited, and CI diffs the committed file against a regenerated copy on every push.

To add a run:

```bash
# 1. start the backend -- see backends/README.md for the launch command per engine
inferlab bench --config <recipe> --url http://localhost:8000 --vram --json results/<label>.json

# 2. regenerate the tables
inferlab report results/*.json --out docs/RESULTS.md
```

(Note the tables live in `docs/`, not at the repo root.)

If your row is missing, the most common causes are:

| Symptom | Cause |
|---|---|
| `TRUNCATION` warning in the output | `finish_reason == "length"`. The workload did not finish. Raise `--max-tokens`, re-run. Do not table a truncated row. |
| Row absent entirely | Zero successful samples. Check the printed error, not the table. |
| `Peak VRAM` column shows `-` | `--vram` was not passed, or `nvidia-smi` is not on PATH. `-` means *unknown*, not *low*. |

**A run with a truncation warning is not a run.** Rerun it. This rule came from a real
measurement: raising a generation cap from 320 to 1024 tokens moved GSM8K accuracy from
76.6% to 95.3%, almost entirely from mid-reasoning truncation. A timing table cannot see
that failure, which is exactly why the guard exists.

---

## Adding a config

Add a recipe to `configs/default.yaml` with its launch command in `notes`, and put the
command in `backends/README.md` too. The recipe is the only place a measured
configuration is declared, so:

```bash
inferlab bench --config <your-recipe> --vram --json results/<label>.json
```

must reproduce the row that carries the same label. State the engine version in the
`notes` field. An engine built against a different minor version is a different engine,
and a row that cannot be rebuilt cannot be quoted.

---

## Reporting a run on your hardware

The most valuable thing you can send. Requirements are small:

1. A JSON file dropped into `results/`, named after the backend label.
2. `--vram` passed, with `nvidia-smi` available (or the column will read `-`).
3. No truncation warnings in the output.
4. A row regenerated into `docs/RESULTS.md` by `inferlab report` -- not typed by hand.
5. The GPU named in the recipe's `notes` field.

A run showing a config is *worse* than expected is more useful than one that confirms
the maintainer's numbers. Send those too.

---

## Open gaps, with acceptance criteria

These are the two places where `inferlab` currently cannot do something rather than
doing it badly. Both are stated in the README's Limitations section. Both have a written
bar so the discussion is about the design, not about whether it is good enough.

### 1. A vendor-agnostic VRAM probe

**The gap.** `inferlab/vram.py` shells out to `nvidia-smi`. Every row that is not
NVIDIA therefore gets no VRAM column at all, and the sampler reports "unavailable".

**Why it matters.** VRAM is the metric that separates "it works" from "it works until
concurrency goes up two steps". Without it, a non-NVIDIA run cannot be compared with an
NVIDIA run at all.

**Acceptance criteria.**

- `VramSampler` gains a backend selection (`nvidia-smi`, `rocm-smi`, `vulkaninfo`, or a
  readable interface that works on all three), with the existing behaviour preserved as
  the default.
- The chosen backend is recorded in the result JSON, next to `peak_mb`, so the provenance
  of the number is never ambiguous.
- When no probe is available, the run still completes and the column reads `-`. This
  already works; do not regress it.
- A test covers each new path using a stubbed command. Do not add a GPU to CI.
- The README's Limitations section drops this bullet.

### 2. Execution-based evaluation

**The gap.** Task accuracy is multiple-choice, scored by option log-probability. It
catches gross capability loss. It does **not** test whether generated code runs.

**Acceptance criteria.**

- Generates a completion, runs it, and compares behaviour against a reference, with the
  generated code executed in a sandbox that cannot touch the host. *This is the hard
  part and the reason it is on the roadmap rather than in the tree.* A design for the
  sandbox is expected in the issue before code.
- `finish_reason` is recorded per task, and a length stop is reported as truncation
  rather than as a wrong answer. `inferlab bench` already does this for timing; the eval
  path must match.
- Results carry the execution timeout and sandbox identifier, or they are not
  reproducible.
- The task set ships with a verifier for each item, not just a prompt and an expected
  answer string.
- The README's Limitations section drops the bullet that calls the task set a regression
  guard rather than a capability measure.

**Please open an issue for either of these before writing code.** The bar is written down
specifically so that a PR does not have to be judged on whether it looks impressive.

---

## Tests

`tests/` needs no GPU. The two classes of test that matter most here:

- **Tests that enforce the honesty invariants.** See
  `tests/test_truncation.py` (a truncated row must be flagged) and
  `tests/test_synthetic.py` (a synthetic row must never reach the measured table). These
  are not decoration; each one exists because a real failure mode was found in review.
- **Tests that guard the documentation.** `test_readme_documents_real_commands` parses
  every `inferlab` command in the README; `test_readme_references_existing_files` checks
  every relative link resolves. A README that documents a command which cannot run is
  worse than no README.

A guard that fires on every run teaches people to ignore it, so tests assert the
absence of warnings on clean input as well as their presence on bad input.

---

## What will not be merged

- A number typed into `docs/RESULTS.md` without matching JSON in `results/`.
- A benchmark row with a truncation warning in its output.
- A new engine backend without a launch command in `backends/README.md`.
- A benchmark that changes the workload set without a note in `docs/METHODOLOGY.md` saying so.
  Comparability across commits is the point of the repo.

## License

Apache-2.0. By contributing you agree your work is licensed under the same terms.