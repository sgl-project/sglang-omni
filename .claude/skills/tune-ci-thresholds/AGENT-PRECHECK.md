# Calibration precheck

Run this checklist before every fresh session and after environment recovery.
`tune.py run` must not start while a mandatory gate fails.

## Before anything: stop CI on the host

Calibration and CI must not run concurrently (team policy, 2026-08-08).
Pinning fixes the CPU quota, so a quiet host does not re-introduce the
idle-host bias the pre-pinning #1260 calibration suffered from; what stopping
CI removes is lane-lease races, reaper kills, and shared-bandwidth noise.

On the runner host, before the first `run`:

```bash
# Back up, then set max_runners: 0 in
# /home/sglang-omni/actions-runner-h100/omni-autoscaler.yaml (hot-reloaded).
grep ^max_runners /home/sglang-omni/actions-runner-h100/omni-autoscaler.yaml  # must be 0
pgrep -f Runner.Listener  # must print nothing once in-flight jobs drain
```

Wait for in-flight jobs to finish (or kill their job containers only with the
user's explicit approval). Restore `max_runners` to its backed-up value as
soon as the last calibration run completes; do not leave CI stopped while
writing reports.

## 0. Environment bootstrap

- Export `TUNE_HOST`, `TUNE_REPO_ROOT`, `TUNE_VENV_PYTHON`, `TUNE_GPU_INCLUDE`,
  and `TUNE_GPU_EXCLUDE` explicitly.
- Every two-GPU calibration group must be pinned to its NUMA-local lane
  bundle, matching the runner-side .env for that lane (`0,1` → `2-15,66-79`,
  `2,3` → `16-31,80-95`, `4,5` → `48-63,112-127`, `6,7` → `32-47,96-111`).
  `tune.py` resolves this from `hosts/*/gpu_group_cpusets` when
  `TUNE_GPU_INCLUDE` is set. Lane `0,1` deliberately excludes cores
  `0,1,64,65`: those are the system cores where host daemons (categraf,
  dockerd) are confined; including them measures daemon noise, not the model.
- Calibrations that write thresholds must anchor on the `0,1` lane: its
  28-core bundle is the conservative baseline for every lane. A threshold
  calibrated on a 32-core lane can be too aggressive when a CI job lands on
  `0,1`; use the 32-core lanes for measurement-only runs.
- Inside a two-GPU container the picked ids are always `0,1`, so the host
  table cannot resolve the physical lane; export `OMNI_CI_CPUSET` with the
  borrowed lane's cores explicitly.
- Missing both the table entry and `OMNI_CI_CPUSET` is a hard precheck /
  launch error — calibration never runs unpinned.
- If precheck finds the lane cpuset already busy (`>` 20% foreign), it
  **refuses** the session and reports the busy fraction. Do not start `run`
  until those cores are free (typically: stop CI first, see above).
- Mid-run intrusion does **not** stop the calibration: the live monitor
  aborts that stage attempt, discards its artifacts, waits for CPU+GPU
  recovery, and retries automatically.
- Source `.github/scripts/ci_env.sh` for CI-comparable defaults when available.
- Do **not** source `~/.zshrc` / `~/.bashrc` for calibration launches; they
  often override `CUDA_VISIBLE_DEVICES` and break multi-group pinning.
- Provide `HF_TOKEN` via the process environment or a mode-`600` file; do not
  depend on interactive shell state.
- For concurrent groups, assign each group a distinct cache root
  (`XDG_CACHE_HOME` / `HOME` / `OMNI_CI_HOME` partition) before precheck.

## 1. Scope and provenance

- Confirm model, selected stages, repeats (baseline 5 clean observations per
  stage; destructive rounds are rejected and replenished on top), and layout mode
  (A / B / C in `SKILL.md`). Prefer Mode C for multi-group hosts unless the
  user asks for one shared worst-of-N (Mode B).
- Record `git rev-parse HEAD`.
- Use a fresh `.tune-runs/<UTC>_<label>/` per calibration process unless
  explicitly resuming.
- Resume when `HEAD` matches the run plan, or when `run --resume` proves the
  commits measurement-equivalent and says so. It prints either
  `note: HEAD moved … no measured code differs` and continues, or an error
  naming the file whose logic changed.
- **Never discard observations because a gate refused.** A refusal says the
  tool would not proceed, not that the data is wrong. Check what actually
  changed (`git diff --name-only <plan-sha>..HEAD`) before spending GPU hours
  on a re-run; hours of valid observations have been thrown away this way.
- Regenerate `stages.yaml` after relevant test/config changes.

## 2. GPU ownership

- Set `TUNE_GPU_INCLUDE` to the exact group owned by this process (normally two
  GPUs such as `0,1`).
- Set `TUNE_GPU_EXCLUDE` for host-reserved GPUs.
- Concurrent processes must use disjoint include sets, run directories, and
  cache roots.
- Verify every selected GPU is idle and below 2048 MiB before launch.
- If a requested pair shows high memory with no processes, or
  `--gpu-reset` reports *In use by another client*, pick another free pair and
  tell the user (see `OPERATIONS.md` ghost-memory section). Do not wait forever
  or clean reserved GPUs.
- Never free GPUs with global `pkill` or user-wide process kills.
- Non-CI cleanup and `wait_for_gpu_memory_release` require
  `CUDA_VISIBLE_DEVICES` set to the physical ids owned by this job.

```bash
export TUNE_GPU_INCLUDE=0,1
export TUNE_GPU_EXCLUDE=6,7
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

## 3. CUDA smoke

`nvidia-smi` is insufficient. Initialize CUDA through the calibration venv
**with this group’s CVD**:

```bash
CUDA_VISIBLE_DEVICES="$TUNE_GPU_INCLUDE" "$TUNE_VENV_PYTHON" - <<'PY'
import os, torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
print(torch.__version__, torch.cuda.device_count(),
      os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
```

On cu130 environments, ensure the venv CUDA libraries are on
`LD_LIBRARY_PATH` when required by the host image.

Do not use a GPU-touching `import sglang` version probe during parallel
calibration; sibling cleanup may SIGKILL it. `tune.py` prefers
`importlib.metadata` for pin checks.

## 4. Repo and dependencies

- Repo contains `pyproject.toml` at the selected commit.
- Calibration venv exists.
- `torch` and `sglang` match current project pins.
- Editable package points to the selected worktree.
- `CAP_SYS_PTRACE` is present for the FP8 TP=2 test.

Using the maintained calibration venv normally requires only:

```bash
cd "$TUNE_REPO_ROOT"
uv pip install -e .
```

Do not rebuild the venv or bulk-download assets before precheck identifies a
specific gap.

## 5. Caches and assets

- Required Hugging Face model and dataset snapshots are locally available.
- Speaker-similarity weights and completion marker exist for TTS stages.
- UTMOS assets are warmed before TTS calibration.
- The FlashInfer JIT cache is warm for the stages being calibrated. A cold
  `fused_moe_90` build takes about eight minutes on a pinned 16-core cpuset
  and burns startup-timeout rounds until it lands. Warm it with one
  throwaway server launch, or calibrate from a container that has run the
  model before.
- This group’s cache root and `.torchinductor` are writable.
- Concurrent groups must not share a writable FlashInfer JIT dir that another
  group may delete mid-run.

## 6. Official precheck

Run it for each selected model:

```bash
python .claude/skills/tune-ci-thresholds/tune.py \
  --model <model> precheck --output-dir "$RUN"
```

Pass criteria:

- precheck exits zero;
- core dependency pins match;
- enough GPUs exist inside `TUNE_GPU_INCLUDE`;
- lane `cpuset` resolved and idle (busy ≤ 20%); busy → refuse;
- required models/datasets and metric assets are present;
- `environment-fingerprint.json` is written (includes cpuset detail);
- any unverified image identity is explicitly visible.

## 7. Active supervision

For every GPU group, start two **IDE-visible** terminals before its first run.
`nohup` into `/tmp` alone is not sufficient — the operator must see Tab A/B in
the Terminal panel (see `OPERATIONS.md`).

```bash
# Tab A: aggregate progress for every run assigned to this group.
bash .claude/skills/tune-ci-thresholds/watch_calibration_group.sh \
  <gpu-group> <run-dir> [<run-dir> ...]

# Tab B: dynamically follows server.log (or local pytest runN.log fallback).
bash .claude/skills/tune-ci-thresholds/watch_calibration_servers.sh \
  <gpu-group> <run-dir> [<run-dir> ...]

# Tab C: operator-visible cpuset busy fraction for THIS group's lane
# (pass the same pair as TUNE_GPU_INCLUDE: 0,1 / 2,3 / 4,5 / 6,7).
bash .claude/skills/tune-ci-thresholds/watch_calibration_cpuset.sh \
  "$TUNE_GPU_INCLUDE"
```

The number of Tab A terminals and Tab B terminals must each equal the number of
GPU groups (one pair per group; no duplicates). Start one Tab C per active
cpuset. Tab B must switch away from killed servers and attach logs from each
new server launch in the same terminal. Locally, expect `runN.log` fallback
because `server_log_file()` only creates `server.log` when
`GITHUB_ACTIONS=true`. Durable filtered Tab B output is teed under
`/tmp/calibration_tabB_<group>.log` as a backup.

During a run, also poll at most every 120 seconds with `status`, `strict-audit`,
and `nvidia-smi`.

A `cpuset_contention` abort is **not** a stop condition — `run` discards that
attempt, waits for CPU+GPU recovery, and retries the stage on its own.
Intervene only for setup errors (missing cpuset mapping) or when an infra
unit hits the ordinary restart cap.

Stop on CUDA initialization failure, extraction warnings, wrong sample scope,
or cleanup affecting GPUs outside the configured group.

A `round N REJECTED as destructive` line is **not** a stop condition — `run`
replaces the round on its own. Intervene only when it reports that a unit hit
the restart or round cap, which means the host is too contended to calibrate.

## 8. Completion

Before report or apply:

- every selected stage has at least `repeats` clean strict observations
  (`strict-audit`); `D` cells are rejected destructive rounds and do not count,
  so a ready stage may show more than `repeats` cells;
- `status` reports `missing=[]` (pytest exit 1 from old threshold asserts does
  not by itself mean missing metrics — see `CONTRACT.md`);
- every counted observation has full expected sample scope and all metrics;
- git provenance passes;
- `report` succeeds through `validate_run_ready()`;
- no calibration or pytest process remains alive for that run directory;
- for speed metrics, skim per-run spread before apply (see `SKILL.md` /
  `CONTRACT.md` speed health check). If a stage’s clean runs still show large
  relative range (rough guide: throughput or latency span ≳ 20–30%), flag it
  and ask before applying large loosens;
- read the report’s **Suspected real defects** section: correctness metrics
  that moved in a rejected round are investigation leads, not noise.
