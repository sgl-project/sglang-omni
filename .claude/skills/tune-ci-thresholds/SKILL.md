---
name: tune-ci-thresholds
description: Calibrate ASR, TTS, and Qwen3-Omni CI thresholds with complete repeated observations, rejection and replenishment of destructive rounds, strict sample-scope validation, GPU-group-isolated cleanup, environment provenance, metric statistics, and operational reliability reporting.
---

# Tune CI thresholds

Use this skill to observe CI correctness and performance in a CI-comparable H100
environment. The policy is at least `--repeats` (default 5) **clean**
observations per selected stage and a strict worst-of-N baseline over those
observations. Rounds whose numbers came from a broken execution are rejected
and replaced, so a stage may end up with more rounds than that. It does not
commit or push changes.

Read these files before running a calibration:

- `CONTRACT.md`: data integrity, completeness, provenance, and apply invariants.
- `AGENT-PRECHECK.md`: mandatory checks before a run.
- `OPERATIONS.md`: GPU-group layouts, cleanup ownership, supervision, and recovery.

## Supported suites

| Model | Scope |
|---|---|
| `asr` | MOSS-Transcribe-Diarize and Qwen3-ASR CI |
| `tts` | Every configured Higgs and MOSS preset; CI may select one preset, calibration observes both. Also includes the Higgs mixed-arrival DP2 serving stages (`tts_serving_mixed_*`) and the Stage 5 MPS placement stages (`tts_mps_dp2_*`). The MPS references live in `tests/test_ci/tts_mps_ci_config.py` rather than the test file itself |
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
5. Run all selected stages for `--repeats` rounds. `run` then rejects
   destructive rounds and replenishes automatically until every stage has
   `--repeats` clean observations (see Destructive rounds).
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

Use `--resume` only to continue the same run directory. A new user request
always gets a new run directory.

`HEAD` moving does not invalidate the observations already in that directory.
`--resume` proves whether the commits are **measurement-equivalent** — nothing
outside `.claude/skills/` changed except numeric constants — and continues when
they are, blocking only on a change that could alter a measured number. See
`CONTRACT.md`. Never conclude that existing data is worthless because a gate
refused; check what actually changed first.

## GPU execution layouts

Layouts are modes, not a fixed two-group recipe. Any number of concurrent GPU
groups is valid when include sets are disjoint and each process has its own run
directory, cache root, and Tab A/B pair. See `OPERATIONS.md` for isolation
rules that make concurrency safe.

### Mode A — one group, one calibration (simplest)

`TUNE_GPU_INCLUDE=0,1` runs every selected stage sequentially until each has
`--repeats` clean observations. Each pytest invocation is cleaned up before the
next invocation.

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
report; that silently changes N. Compare distributions, or explicitly analyze
them as a larger observation set when the user asks.

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
identity, and disjoint stage ownership. Each partition must reach `--repeats`
clean observations on its own before merging. Use Mode B only when the user
wants one combined worst-of-N faster; it is not the default multi-GPU layout.

Every concurrent process must set `TUNE_GPU_INCLUDE`. Cleanup is scoped to the
physical GPU indices owned by that process. Global `pkill`, user-wide kills, and
host-wide cleanup are forbidden.

## Scoping a run to fewer GPUs

`precheck` sizes its GPU requirement from the largest `gpus_per_test` in the
model. `run --stages <subset>` computes the requirement from the selected
stages and runs its own precheck, so use it when a subset needs fewer GPUs than
the full model suite.

Presets own disjoint constant namespaces through
`calibration_presets[*].constant_filter`. Until this was fixed, the filter was
applied only to tests that also declare `variants`, so every preset of a
variant-less test claimed the first preset's symbols and one preset's
worst-of-N could be written over another's. Always read the `discover` output
and confirm each stage points at its own symbols.

Run `discover` on a Linux checkout. It records a sha256 of the working-tree
file, so regenerating on a CRLF checkout rewrites every hash and the schema
then mismatches on the calibration host.

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

## Destructive rounds

A round can finish with full sample scope and non-null metrics and still be
worthless: host contention, a cold autotune cache, or a thrashing server
produce numbers that describe the machine, not the model. Feeding one such
round into strict worst-of-N sets the CI reference from the accident.

A value is destructive only when it is both far from the robust centre
(MAD z > 3.5) **and** separated from its nearest neighbour by a ≥20% gap. A
round is destructive if **any** one of its metrics is — a round whose speed
collapsed cannot be trusted for accuracy either — and it is then discarded for
every stage in that pytest invocation.

`run` handles this automatically: reject, run `2n` replacement rounds with new
indices, re-detect (replacements can be destructive too), and stop once
`--repeats` clean observations exist. Rejected `run{k}.json` files stay on disk
as evidence. `strict-audit` renders them as `D` (`✓✓✓✓D✓✓`, effective N=6).

If a stage's values split into two comparable populations, no round can be
identified as the broken one — the block is discarded and re-run once. A stage
still split after that is bistable by nature: every observation is kept, worst-of-N
stays conservative, and the log says `STILL unstable after a restart`. Review
those stages by hand before applying their thresholds.

Budget roughly +35–40% wall clock on a contended host.
`--no-destructive-rejection` turns the mechanism off, and `--repeats < 5`
disables it automatically with a warning (too few points to identify an
outlier). `CONTRACT.md` has the caps and the full policy.

`test_destructive_rejection.py` covers this machinery end to end without GPUs
— run it after touching any of it.

## Reports

The report has three views; the middle one appears only when a round was
rejected.

### Metric calibration

For every metric it contains all per-run values plus:

- strict worst-of-N over the clean observations;
- median, min, max, range, standard deviation, and coefficient of variation;
- IQR-based outlier flags — informational only; an IQR flag never removes an
  observation, and ordinary outliers stay in the aggregation;
- aggregate success count and 95% Wilson interval for accuracy where sample
  counts are available;
- seed policy recorded for every run.

Accuracy/WER and performance retain separate threshold semantics. Display
rounding never changes the raw worst value used by `apply-plan`.

Pytest may exit non-zero because an **old** CI threshold assertion failed while
metrics and full sample scope were still produced. That is a threshold failure,
not a missing observation. Completeness is decided by `strict-audit` /
`status` (`missing=[]`, every stage at `--repeats` clean observations), not by
pytest pass/fail. See `CONTRACT.md`.

### Rejected destructive rounds

Every rejected round with its raw values, the rest-median it was compared
against, the gap, and the z that condemned it. Rejected rounds whose
**correctness** metrics moved also appear under **Suspected real defects**:
contention explains a slow round, not a wrong one, so those are leads to
investigate rather than noise to discard.

### Operational reliability

For every stage it contains:

- logical observations and total infrastructure attempts;
- retried observations and failed attempts;
- partial-sample observations;
- attempt reason, duration, pytest exit code, and physical GPU indices in the
  underlying `run{k}.json`.

Infrastructure failures are not silently treated as metric observations.

## Environment comparability

CI pins every perf-gate pytest session to a NUMA-local cpuset: the runner lane
exports `OMNI_CI_CPUSET` and `tests/test_model/conftest.py` applies it
(sched_setaffinity, inherited by spawned servers). A PR job does not need to
know which lane it drew — the runner injects the env. Calibration **does**,
because it chooses the GPUs: `tune.py` looks up `TUNE_GPU_INCLUDE` in the host
profile's `gpu_group_cpusets` table and exports the matching `OMNI_CI_CPUSET`.

| GPU pair | CPU cores (32 logical) |
|---|---|
| `0,1` | `0-15,64-79` |
| `2,3` | `16-31,80-95` |
| `4,5` | `48-63,112-127` |
| `6,7` | `32-47,96-111` |

Keep that table in sync with the production runner `.env` partition. Numbers
measured without the pin do not describe the environment CI gates run in. An
explicit `OMNI_CI_CPUSET` in the operator's environment overrides the table.
Missing both the table entry and the env var is a hard error — calibration
never runs unpinned.

### CPU occupation policy

- **Precheck (may stop):** if the lane cpuset is already busy with foreign
  load (`>` 20%), refuse the session and write the busy fraction into
  `precheck.json` / the fingerprint. Fix the host (stop CI / other jobs) and
  re-run precheck.
- **During `run` (must not stop the calibration):** a live foreign-load
  monitor watches the reserved cores. If intrusion exceeds two foreign cores,
  `tune.py` aborts **only that stage attempt**, reports the peak, wipes its
  basetemp artifacts so contaminated metrics never enter `run{k}.json` or the
  report, waits until the cpuset is idle **and** the owned GPUs are free, then
  retries the same stage. Contention retries are unbounded; only ordinary
  infra failures (OOM/crash/…) consume `_MAX_RUN_ATTEMPTS`.
- **Tab C** (`watch_calibration_cpuset.sh <gpu-group>`) is the
  operator-visible supervisor for **that process's** lane — start one Tab C
  per concurrent `TUNE_GPU_INCLUDE` (e.g. `2,3` watches `16-31,80-95`).
  Enforcement lives in `tune.py` from the picked GPUs, not from a hardcoded
  `0,1` cpuset.

`precheck` writes `environment-fingerprint.json` containing:

- image name/digest when supplied by the runtime;
- host/platform, Python executable, driver, GPU UUID/SKU/memory and topology;
- torch/sglang versions and a full dependency-freeze hash;
- relevant environment variables and selected GPU group;
- resolved lane `cpuset` and its precheck busy fraction;
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
partial observations, wrong sample scope, missing metrics, artifacts from a
commit that is neither the calibration commit nor measurement-equivalent to it,
or a stage left with fewer than `--repeats` clean observations after
destructive rejection.

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

Destructive rejection removes single broken rounds; it does not certify a
stage. Before applying speed changes, skim each speed stage’s clean raw values.
If the relative range is still large (rough guide: ≳ 20–30% of the median for
throughput or latency/RTF), the stage is genuinely noisy — flag it and ask
before writing large loosens. A session the user rejects as contaminated needs
a fresh run directory — see Contaminated-run recovery in `OPERATIONS.md`.

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
  test_cpuset_gates.py
  test_destructive_rejection.py
  tail_calibration_pytest.sh
  watch_calibration_group.sh
  watch_calibration_servers.sh
  watch_calibration_cpuset.sh
  hosts/*.yaml
  models/{asr,tts,omni}/{config.yaml,stages.yaml}
```
