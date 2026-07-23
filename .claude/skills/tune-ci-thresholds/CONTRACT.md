# Calibration contract

These invariants define a valid calibration. CLI commands must enforce them;
agent instructions are not a substitute for code gates.

## Session identity

- A new request uses a new UTC-timestamped directory on current `HEAD`.
- Resume is interruption recovery for the same directory and commit.
- Every run artifact records the calibration commit.
- A calibration never mixes artifacts from commits, schemas, or environments.

## Observation validity

A stage repeat is strict-valid only when:

- its result JSON exists and is readable;
- every tracked metric is non-null;
- sample `ok == total`;
- `total == expected_samples` when configured;
- the recorded commit matches the plan.

Threshold assertion failures may still yield a valid observation when all
metrics and samples were produced. Typical artifact shape:

- `status=ok`
- `reason=threshold_assertion (exit 1)` (or equivalent)
- non-null metrics and full sample scope

That case is a **threshold failure** against the previous CI constants, not a
**missing** observation. Agents must not treat pytest `failed: exit 1` as
incomplete calibration when `strict-audit` marks the cell ✓ and `status`
reports `missing=[]`.

Infrastructure crashes, OOMs, timeouts, missing output, SIGKILL from foreign
cleanup, and partial samples are not valid metric observations.

## Worst-of-N

- Default N is 5.
- Every selected stage needs exactly N strict-valid observations.
- Lower-bound metrics use the minimum; upper-bound metrics use the maximum.
- No partial or infrastructure-failed observation participates in aggregation.
- Outliers are flagged and retained unless a separately documented invalidation
  proves the run was not a valid observation.

## Schema

- `models/<model>/config.yaml` declares non-inferable metric paths and sample
  scopes.
- `stages.yaml` is generated deterministically from config and current tests.
- `CONCURRENCY` is execution fan-out, never sample count.
- A test or threshold-file hash mismatch blocks calibration.
- Report and apply consume the schema bound to the run.

## GPU ownership and concurrent isolation

- A pytest invocation owns only its selected physical GPU indices.
- Cleanup may target only those indices (`CUDA_VISIBLE_DEVICES` = physical ids;
  unset CVD before `nvidia-smi --id`).
- Concurrent groups require disjoint `TUNE_GPU_INCLUDE` values, separate run
  directories, and separate cache roots.
- Cleanup must not kill processes whose CVD is disjoint from its scope, nor
  ephemeral version-probe cmdlines.
- Global process-pattern and user-wide kills are forbidden.
- Non-CI `wait_for_gpu_memory_release` requires an explicit CVD scope.

## Final consumers

`report` and `apply-plan` use the same readiness validator. Neither may consume
an incomplete run. Apply writes raw pre-slack references and must never write a
derived assertion threshold.

`apply-plan` is read-only planning output. File edits happen only after an
explicit apply decision (`report` / `smart` / `full`). After a `full` apply,
re-running `apply-plan` should show `direction=equal` for every calibratable
metric (`direction=fixed` symbols are never rewritten).

Fixed threshold symbols (`_FIXED_THRESHOLD_SYMBOLS` in `tune.py`, currently
`MOSS_TD_STREAM_N_ABOVE_50_CER_MAX`) are not discover targets and must remain
hand-pinned across calibration cycles.

## Speed health before apply

Strict readiness is necessary but not sufficient for speed thresholds. Before
applying large speed `loosens` / `tightens`, inspect per-run values for each
speed stage:

- Flag stages whose five observations have a large relative range (rough guide:
  max−min over |median| ≳ 0.20–0.30 for throughput or latency/RTF).
- Present the spread and ask before writing those references, especially when
  the session shared the host with other heavy GPU work.
- Contaminated or rejected sessions must not drive apply; recover with a fresh
  run directory per `OPERATIONS.md`.

## Required provenance

The final artifact records commit, dirty state, venv, dependency hash, core
versions, container identity when available, driver/GPU/topology, selected GPU
group, relevant environment, required model/dataset IDs, attempt history, and
seed policy.
