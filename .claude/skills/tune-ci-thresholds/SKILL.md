---
name: tune-ci-thresholds
description: Calibrate ASR, TTS, and Qwen3-Omni CI thresholds with complete repeated observations, strict sample-scope validation, GPU-group-isolated cleanup, environment provenance, metric statistics, and operational reliability reporting.
---

# Tune CI thresholds

Use this skill to observe CI correctness and performance in a CI-comparable H100
environment. The default policy is five complete observations per selected stage
and a strict worst-of-five baseline. It does not commit or push changes.

Read these files before running a calibration:

- `CONTRACT.md`: data integrity, completeness, provenance, and apply invariants.
- `AGENT-PRECHECK.md`: mandatory checks before a run.
- `OPERATIONS.md`: GPU-group layouts, cleanup ownership, supervision, and recovery.

## Supported suites

| Model | Scope |
|---|---|
| `asr` | MOSS-Transcribe-Diarize and Qwen3-ASR CI |
| `tts` | Every configured Higgs and MOSS preset; CI may select one preset, calibration observes both |
| `omni` | Numeric threshold stages in Qwen3-Omni CI |

`stages.yaml` is generated from the current test files and `config.yaml`. It is
not a hand-maintained source of truth.

## Standard workflow

1. Resolve a host profile and choose an explicit GPU layout (see below).
2. Create a fresh UTC-timestamped run directory on current `HEAD` (one per
   independent calibration process).
3. Run `precheck` for every selected model.
4. Start one **IDE-visible** progress Tab A and one dynamic server-log Tab B
   per GPU group (`nohup` to `/tmp` alone does not count — see `OPERATIONS.md`).
5. Run all selected stages for five repeats.
6. Poll `status`, `strict-audit`, the active pytest log, and GPU state at least
   every 120 seconds.
7. Generate `report.md` only after the shared readiness gate passes.
8. Show the report before asking whether thresholds should be applied. For
   speed metrics, skim per-run spread first (see Threshold application).
9. Apply only with explicit user confirmation. Run a post-apply validation.

```bash
export TUNE_HOST=sglang-h100-ci
export TUNE_REPO_ROOT=/path/to/current/worktree
export TUNE_VENV_PYTHON=/path/to/omni/bin/python
export TUNE_GPU_INCLUDE=0,1
export TUNE_GPU_EXCLUDE=6,7

RUN=".tune-runs/$(date -u +%Y%m%dT%H%M%SZ)_omni_r5"
python .claude/skills/tune-ci-thresholds/tune.py \
  --model omni precheck --output-dir "$RUN"
python .claude/skills/tune-ci-thresholds/tune.py \
  --model omni run --stages ALL --repeats 5 --output-dir "$RUN"
python .claude/skills/tune-ci-thresholds/tune.py strict-audit --run-dir "$RUN"
python .claude/skills/tune-ci-thresholds/tune.py report --run-dir "$RUN"
```

Use `--resume` only to continue the same run directory on the same commit. A new
user request always gets a new run directory.

## GPU execution layouts

Layouts are modes, not a fixed two-group recipe. Any number of concurrent GPU
groups is valid when include sets are disjoint and each process has its own run
directory, cache root, and Tab A/B pair. See `OPERATIONS.md` for isolation
rules that make concurrency safe.

### Mode A — one group, one calibration (simplest)

`TUNE_GPU_INCLUDE=0,1` runs every selected `stage × 5` sequentially. Each
pytest invocation is cleaned up before the next invocation.

### Mode C — N groups, N independent full calibrations (default for multi-GPU)

**Preferred** when several two-GPU groups are available (for example three
groups on `0,1` / `2,3` / `4,5`). Each group runs a complete `ALL × 5` into its
own run directory:

```bash
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages ALL --repeats 5 \
  --output-dir "$RUN_G01"
TUNE_GPU_INCLUDE=2,3 python tune.py --model omni run --stages ALL --repeats 5 \
  --output-dir "$RUN_G23"
TUNE_GPU_INCLUDE=4,5 python tune.py --model omni run --stages ALL --repeats 5 \
  --output-dir "$RUN_G45"
```

These are independent replications. Do **not** `merge-runs` them into one
worst-of-five report; that silently changes N. Compare distributions, or
explicitly analyze them as more than five observations when the user asks.

### Mode B — N groups share one calibration scope (optional speedup)

Split stages across groups with **disjoint** stage ownership, then merge:

```bash
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run \
  --stages <partition-A> --repeats 5 --output-dir "$RUN_A"
TUNE_GPU_INCLUDE=2,3 python tune.py --model omni run \
  --stages <partition-B> --repeats 5 --output-dir "$RUN_B"
```

Do not merge by copying JSON files. After every partition is strict-ready:

```bash
python tune.py merge-runs --run-dir "$RUN_A" --run-dir "$RUN_B" \
  --output-dir "$RUN_COMBINED"
```

`merge-runs` validates commit, model, repeat count, stage schema, environment
identity, and disjoint stage ownership. Use Mode B only when the user wants one
combined worst-of-five faster; it is not the default multi-GPU layout.

Every concurrent process must set `TUNE_GPU_INCLUDE`. Cleanup is scoped to the
physical GPU indices owned by that process. Global `pkill`, user-wide kills, and
host-wide cleanup are forbidden.

## Stage schema lifecycle

After CI test or threshold changes:

```bash
python .claude/skills/tune-ci-thresholds/tune.py --model asr discover
python .claude/skills/tune-ci-thresholds/tune.py --model tts discover
python .claude/skills/tune-ci-thresholds/tune.py --model omni discover
```

Review the diff. `CONCURRENCY` is never a sample count. Full-dataset tests must
declare `expected_samples` in model config when the test has no literal sample
cap. Current explicit scopes include MMMU=50 and MMSU=2000.

Run must not proceed on a test/threshold SHA mismatch. Regenerate stages first.

## Reports

The report has two distinct views.

### Metric calibration

For every metric it contains all per-run values plus:

- strict worst-of-N;
- median, min, max, range, standard deviation, and coefficient of variation;
- IQR-based outlier flags, without automatically deleting an observation;
- aggregate success count and 95% Wilson interval for accuracy where sample
  counts are available;
- seed policy recorded for every run.

Accuracy/WER and performance retain separate threshold semantics. Display
rounding never changes the raw worst value used by `apply-plan`.

Pytest may exit non-zero because an **old** CI threshold assertion failed while
metrics and full sample scope were still produced. That is a threshold failure,
not a missing observation. Completeness is decided by `strict-audit` /
`status` (`missing=[]`, N/N strict), not by pytest pass/fail. See `CONTRACT.md`.

### Operational reliability

For every stage it contains:

- logical observations and total infrastructure attempts;
- retried observations and failed attempts;
- partial-sample observations;
- attempt reason, duration, pytest exit code, and physical GPU indices in the
  underlying `run{k}.json`.

Infrastructure failures are not silently treated as metric observations.

## Environment comparability

`precheck` writes `environment-fingerprint.json` containing:

- image name/digest when supplied by the runtime;
- host/platform, Python executable, driver, GPU UUID/SKU/memory and topology;
- torch/sglang versions and a full dependency-freeze hash;
- relevant environment variables and selected GPU group;
- required model and dataset IDs and cache state.

Set `OMNI_CI_IMAGE_DIGEST` or `CONTAINER_IMAGE_DIGEST` when the runtime exposes
the immutable image identity. Without it the report says image identity is
unverified; matching editable source and core pins alone does not prove complete
CI equivalence.

In the usual maintained calibration environment, update the checkout and run
`uv pip install -e .`, then let precheck verify pins and assets. A meaningful
mismatch is reported as non-comparable and must not drive threshold changes.

## Threshold application

`report` and `apply-plan` call the same `validate_run_ready()` gate. Both refuse
partial observations, wrong sample scope, missing metrics, or mixed commit SHA.

`apply-plan` is read-only JSON: for each metric it emits `worst_raw`,
`write_value`, `current_raw`, and `direction` (`tightens` / `loosens` /
`equal` / `fixed`). The agent (or operator) performs the file edits.
Application writes pre-slack reference values only. CI assertion slack remains
in the tests. Never write constants derived by `apply_slack`,
`apply_wer_slack`, `apply_mos_slack`, `THRESHOLD_SLACK_HIGHER`, or
`THRESHOLD_SLACK_LOWER`.

**Fixed thresholds (never apply):** symbols in
`_FIXED_THRESHOLD_SYMBOLS` are excluded from discover and must not be rewritten
during calibration. Today that includes
`MOSS_TD_STREAM_N_ABOVE_50_CER_MAX` (keep at `31`; streaming headcount is too
unstable for worst-of-N). If `apply-plan` reports `direction=fixed`, leave the
literal unchanged.

Supported decisions after the report:

- `report`: do not edit thresholds.
- `smart`: apply correctness/quality references; automatically tighten speed;
  ask before loosening speed. Skip `direction=fixed`.
- `full`: apply every non-equal worst-of-N `write_value`. Skip
  `direction=fixed`.

Before applying speed changes, skim each speed stage’s five raw values. If the
relative range is large (rough guide: ≳ 20–30% of the median for throughput or
latency/RTF), flag the stage and ask before writing large loosens. Rejected or
contaminated sessions need a fresh run directory — see Contaminated-run
recovery in `OPERATIONS.md`.

After edits:

1. Regenerate stages (`discover`) and confirm source symbols still match.
2. Re-run `apply-plan`; every metric should report `direction=equal`.
3. Run focused unit/static tests when practical.
4. Run at least one validation observation using the applied references and
   derived slack when the user wants post-apply confirmation.
5. Confirm serialization and rounding did not tighten past the raw worst value.

Do not edit threshold files before the final apply decision. Do not commit or
push without explicit authorization.

## Files

```text
tune-ci-thresholds/
  SKILL.md
  CONTRACT.md
  OPERATIONS.md
  AGENT-PRECHECK.md
  tune.py
  tail_calibration_pytest.sh
  watch_calibration_group.sh
  watch_calibration_servers.sh
  hosts/*.yaml
  models/{asr,tts,omni}/{config.yaml,stages.yaml}
```
