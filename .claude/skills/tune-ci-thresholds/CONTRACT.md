# Calibration contract

These invariants define a valid calibration. CLI commands must enforce them;
agent instructions are not a substitute for code gates.

## Session identity

- A new request uses a new UTC-timestamped directory on current `HEAD`.
- Resume is interruption recovery for the same directory, on the calibration
  commit or a commit proven **measurement-equivalent** to it (below).
- Every run artifact records the commit it was produced on.
- A calibration never mixes artifacts from commits that could measure
  differently, from different schemas, or from different environments.

### Measurement-equivalent commits

`HEAD` moving is not by itself a reason to discard observations. What matters
is whether anything that can change a measured number moved. Two commits are
measurement-equivalent when, for every file that differs:

- it lives under `.claude/skills/` (calibration tooling, not measured code); or
- it is a `.py` file whose AST is identical once every numeric literal is
  blanked — i.e. only constants such as CI thresholds changed.

Anything else — changed logic, a non-Python file, an added or removed file —
is a real change and blocks reuse.

Threshold constants decide whether an assertion passes; they do not affect what
the benchmark measures. Refusing to reuse observations across such a commit
throws away hours of valid GPU time and buys no integrity.

`run --resume` performs this proof itself, prints the verdict, and records the
accepted commit under `equivalent_commits` in `plan.json`. Provenance accepts a
run artifact whose `git_sha` is the calibration commit or any commit recorded
there. `merge-runs` is unaffected: its inputs must still share one calibration
commit.

## Observation validity

A stage repeat is strict-valid only when:

- its result JSON exists and is readable;
- every tracked metric is non-null;
- sample `ok == total`;
- `total == expected_samples` when configured;
- the recorded commit is the calibration commit or one recorded as
  measurement-equivalent to it;
- the round is not marked destructive (see below).

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

## Lane cpuset binding

Every calibration process must pin to the runner-lane cpuset that owns its
`TUNE_GPU_INCLUDE` GPUs (or an explicit `OMNI_CI_CPUSET`). Unpinned runs are
refused.

- **Precheck:** a cpuset already busy with foreign load is a hard error; the
  session must not start.
- **Mid-run intrusion:** aborting a stage attempt on foreign-load is an
  infrastructure discard, not a metric observation and not a destructive
  round. Contaminated basetemp artifacts are wiped so they never appear in
  `run{k}.json` or the report. `run` waits for CPU and GPU recovery and
  retries the same stage without stopping the calibration. Contention
  retries do not consume the ordinary infra attempt budget.
- **What counts as intrusion:** only the driver's live monitor decides, and
  only on load that persists — above `_CONTENTION_FAIL_CORES` for
  `_CONTENTION_FAIL_WINDOWS` consecutive sample windows. A discard costs the
  attempt's whole runtime, so the evidence must outlast one window. The
  discarded attempt records the peak, mean and window count actually
  measured; a record that reports the threshold back as the observation is
  not an audit trail.
- **The session's own contention line is not evidence.** The
  `[cpuset-contention]` summary a pytest session prints at teardown roots its
  tree at the pytest pid, so stages whose servers double-fork and reparent to
  init have their own CPU charged as foreign. It is recorded as
  `session_summary_peak_cores` for triage and must never gate a round.

## Destructive observations

A third category, distinct from infrastructure failure (not an observation) and
threshold-assertion failure (a valid observation): a round that completed with
full sample scope and non-null metrics but whose numbers describe the machine
rather than the model — host contention, a cold autotune cache, a thrashing
server.

A metric value is **destructive** only when both hold:

- robust MAD z above `_DESTRUCTIVE_Z` (3.5) — far from the centre; and
- separated from its nearest neighbour by at least `_DESTRUCTIVE_GAP` (20%) of
  the remaining values' median, and by more than the metric's display
  resolution.

The gap test is mandatory, not an optimisation. At N=5 the MAD is a coarse
estimator and z alone flags ordinary tail points. In a metric-rich unit such a
flag lands in *every* round, which would condemn all of them and make
reject-and-replace non-terminating.

A non-finite metric (NaN / inf) is condemned outright — it is a broken
measurement, and left in the sample it poisons the median and makes every
comparison silently false.

Rejection is **per round, not per metric**. A round whose speed collapsed
cannot be trusted for accuracy either, so a single destructive metric discards
the whole round for every stage in that pytest invocation.

Detection needs at least `_DESTRUCTIVE_MIN_OBS` (5) observations. Below that
the MAD cannot separate a broken round from ordinary spread, so `run` with
`--repeats < 5` prints a warning and disables rejection rather than pretending
to protect the calibration.

Rejected rounds are marked `destructive: true` with evidence and **kept on
disk** — never deleted. They are replaced by additional rounds, not re-run in
place, so the audit trail shows what was rejected and why.

Already-rejected rounds are removed from the sample before judging the rest.
Left in, a condemned value sits next to the next broken round and cancels its
gap, so two similar broken rounds mask each other and both survive.

### Degenerate samples

Past roughly 40% contamination the median and MAD move *inside* the bad
cluster and identification inverts — the good rounds get flagged instead. The
detector cannot see this from the flags alone, so a separate guard checks
whether the values split into two populations of size >= 2 separated by
`_DEGENERATE_SPLIT` (50%) of the median. That threshold is deliberately far
above `_DESTRUCTIVE_GAP`: mild bimodality is a noisy stage, surfaced by the
speed-health check, not a detector failure.

A degenerate sample is treated as `n >= _DESTRUCTIVE_FULL_RERUN_N`.

### Replenishment

- `n` destructive rounds in a unit → run `2n` additional rounds.
- Re-detect over **all** rounds each cycle; replacement rounds can themselves
  be destructive.
- Stop when at least `repeats` clean observations exist.
- `n >= 3` (`_DESTRUCTIVE_FULL_RERUN_N`), or a degenerate sample: the "the
  others agree" premise the gap test relies on has failed. Discard the whole
  block and restart, at most `_DESTRUCTIVE_MAX_RESTARTS` (1) time.
- If it is still unstable after that restart, the instability belongs to the
  test or the host, not to one round. **Keep every observation and reject
  nothing** — worst-of-N over all of them is the conservative bound, and
  blocking the report would be worse than a wide threshold. The stage is
  flagged loudly for review instead.
- Hard cap `_DESTRUCTIVE_MAX_ROUNDS` (15) rounds per unit.
- The scale estimate carries an absolute floor so a shrinking clean set cannot
  collapse the MAD and start rejecting ordinary variation — that is the entry
  point for outlier-peeling bias.
- `--no-destructive-rejection` disables the whole mechanism.

## Worst-of-N

- Baseline N is `--repeats` (default 5); **effective N is recorded per stage**
  and may exceed it when rounds were rejected and replenished. This is separate
  from `_DESTRUCTIVE_MIN_OBS`, which is fixed at 5 and governs only whether the
  detector will judge at all.
- Every selected stage needs at least `repeats` clean observations.
- Lower-bound metrics use the minimum; upper-bound metrics use the maximum,
  over the clean subset only.
- No partial, infrastructure-failed, or destructive observation participates in
  aggregation.
- Ordinary outliers — flagged by IQR but not meeting the destructive test — are
  retained, not trimmed.

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
an incomplete run — including a stage left with fewer than `repeats` clean
observations because replenishment hit a cap. Apply writes raw pre-slack
references and must never write a derived assertion threshold.

`apply-plan` is read-only planning output. File edits happen only after an
explicit apply decision (`report` / `smart` / `full`). After a `full` apply,
re-running `apply-plan` should show `direction=equal` for every calibratable
metric (`direction=fixed` symbols are never rewritten).

Fixed threshold symbols (`_FIXED_THRESHOLD_SYMBOLS` in `tune.py`, currently
`MOSS_TD_STREAM_N_ABOVE_50_CER_MAX`) are not discover targets and must remain
hand-pinned across calibration cycles.

## Speed health before apply

Destructive rejection removes single broken rounds; it does not certify a
stage. A stage can still be genuinely noisy across every round, and on a
contended host tail-latency metrics (p95, TTFA, inter-chunk) remain the least
trustworthy even after rejection.

Strict readiness is necessary but not sufficient for speed thresholds. Before
applying large speed `loosens` / `tightens`, inspect per-run values for each
speed stage:

- Flag stages whose clean observations still have a large relative range (rough
  guide: max−min over |median| ≳ 0.20–0.30 for throughput or latency/RTF).
  Surviving destructive rejection does not make a stage stable.
- Present the spread and ask before writing those references, especially when
  the session shared the host with other heavy GPU work.
- A session the user rejects as contaminated must not drive apply; recover with
  a fresh run directory per `OPERATIONS.md`.

## Required provenance

The final artifact records the calibration commit and any commit accepted as
measurement-equivalent, dirty state, venv, dependency hash, core versions,
container identity when available, driver/GPU/topology, selected GPU group,
relevant environment, required model/dataset IDs, attempt history, seed policy,
and — for every rejected round — its raw values and the evidence that condemned
it, so any exclusion can be re-derived.
