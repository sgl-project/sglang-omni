# Calibration precheck

Run this checklist before every fresh session and after environment recovery.
`tune.py run` must not start while a mandatory gate fails.

## 0. Environment bootstrap

- Export `TUNE_HOST`, `TUNE_REPO_ROOT`, `TUNE_VENV_PYTHON`, `TUNE_GPU_INCLUDE`,
  and `TUNE_GPU_EXCLUDE` explicitly.
- Source `.github/scripts/ci_env.sh` for CI-comparable defaults when available.
- Do **not** source `~/.zshrc` / `~/.bashrc` for calibration launches; they
  often override `CUDA_VISIBLE_DEVICES` and break multi-group pinning.
- Provide `HF_TOKEN` via the process environment or a mode-`600` file; do not
  depend on interactive shell state.
- For concurrent groups, assign each group a distinct cache root
  (`XDG_CACHE_HOME` / `HOME` / `OMNI_CI_HOME` partition) before precheck.

## 1. Scope and provenance

- Confirm model, selected stages, repeats (default 5), and layout mode
  (A / B / C in `SKILL.md`). Prefer Mode C for multi-group hosts unless the
  user asks for one shared worst-of-five (Mode B).
- Record `git rev-parse HEAD`.
- Use a fresh `.tune-runs/<UTC>_<label>/` per calibration process unless
  explicitly resuming.
- Resume only when `HEAD` matches the run plan.
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
- required models/datasets and metric assets are present;
- `environment-fingerprint.json` is written;
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
```

The number of Tab A terminals and Tab B terminals must each equal the number of
GPU groups (one pair per group; no duplicates). Tab B must switch away from
killed servers and attach logs from each new server launch in the same
terminal. Locally, expect `runN.log` fallback because `server_log_file()` only
creates `server.log` when `GITHUB_ACTIONS=true`. Durable filtered Tab B output
is teed under `/tmp/calibration_tabB_<group>.log` as a backup.

During a run, also poll at most every 120 seconds with `status`, `strict-audit`,
and `nvidia-smi`.

Stop on CUDA initialization failure, extraction warnings, wrong sample scope,
or cleanup affecting GPUs outside the configured group.

## 8. Completion

Before report or apply:

- every selected stage has N/N strict observations (`strict-audit`);
- `status` reports `missing=[]` (pytest exit 1 from old threshold asserts does
  not by itself mean missing metrics — see `CONTRACT.md`);
- every observation has full expected sample scope and all metrics;
- git provenance passes;
- `report` succeeds through `validate_run_ready()`;
- no calibration or pytest process remains alive for that run directory;
- for speed metrics, skim per-run spread before apply (see `SKILL.md` /
  `CONTRACT.md` speed health check). If a stage’s five runs show large
  relative range (rough guide: throughput or latency span ≳ 20–30%), flag it
  and ask before applying large loosens.
