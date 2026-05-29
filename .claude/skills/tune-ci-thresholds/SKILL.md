---
name: tune-ci-thresholds
description: Run CI tests N times per stage on the H20 CI-reproduction host, produce a per-metric strict worst-of-N observation report (every stage must have N full-sample repeats), and (on user confirmation) write the worst-of-N values back into the test files as new baselines. Use when recalibrating CI thresholds after an engine update. Currently supports qwen3-omni-v1 and s2-pro-v1; extensible via models/<name>/config.yaml.
---

# tune-ci-thresholds

## Scope
This skill is for the H20 CI-reproduction host only (the same image
CI uses, `frankleeeee/sglang-omni:dev`; the container name varies).
Numbers from environments that differ meaningfully from CI (different
GPU model, different image, different pinned sglang/torch) are not
comparable and must not drive threshold changes. If you just want to
run the tests locally, use pytest directly — this skill is not for
that.

The skill is observation-first: it runs tests N times and produces a
**strict worst-of-N** report. After the report is shown, it offers a
one-shot **apply step** that writes the worst-of-N values directly
into the test files as the new P95 baselines and accuracy / WER
thresholds — **only** if the user explicitly confirms. The skill
still does NOT re-run `apply_slack` separately, generate patch files,
or commit / push anything; if the user rejects the apply prompt, the
test files stay untouched and the user picks values manually from
the report.

## Strict worst-of-N (mandatory — non-negotiable)

**Worst-of-N is only valid when every stage has N full-sample repeats.**
This is a hard requirement for report, apply, and any threshold change —
now and in all future calibrations.

### What counts as a valid repeat

A stage-run is **strict-complete** (✓) only when **both** hold:

1. **All tracked metrics extracted** — every metric in `stages.yaml`
   for that stage is non-null in `run{k}.json`.
2. **Full sample coverage** — `sample_counts.ok == sample_counts.total`
   and both are non-null (e.g. MMSU `2000/2000`, talker `20/20`).

Anything else is **not** a valid worst-of-N input:

| Symbol | Meaning | Valid for worst-of-N? |
|--------|---------|----------------------|
| **✓** | Strict-complete (`ok == total`, all metrics present) | **Yes** |
| **△** | Partial — metrics exist but `ok < total` (OOM mid-benchmark, early abort) | **No** |
| **✗** | No usable metrics — missing JSON, OOM before results, extraction failed | **No** |
| **—** | Not yet run | **No** |

**Partial runs are never acceptable for worst-of-N**, even when
`tune.py` marks the stage-run `status: ok` with reason
`threshold_assertion (OOM)` — that only means metrics were *read*, not
that the repeat was *complete*.

### tune.py `complete: true` ≠ strict-ready

`tune.py status` counts a stage-run toward `ok/total` when metrics
were extracted (including △ partial and threshold-failure runs). That
counter is a **scheduling/progress** signal, **not** strict readiness.

Before **report**, **apply**, or telling the user calibration is
done, you **must** run the strict audit below and confirm:

```
strict-ready stages: <S> / <total stages>   (each stage has N/N ✓)
```

If any stage has fewer than N ✓ repeats, calibration is **incomplete
for threshold purposes** — `--resume` / targeted re-runs until every
gap is filled. Do **not** apply thresholds from a mix of ✓, △, and ✗.

### Strict audit command (run before report / apply / status updates to user)

From repo root, after each major progress checkpoint and always before
steps 6–9:

```
python3 << 'PY'
import json
from pathlib import Path

run_dir = Path("<run-dir>")
plan = json.loads((run_dir / "plan.json").read_text())
repeats = plan["repeats"]
stage_keys = plan["stages"]

def classify(p):
    if not p.exists():
        return "—"
    d = json.loads(p.read_text())
    sc = d.get("sample_counts") or {}
    tot, ok = sc.get("total"), sc.get("ok")
    metrics = d.get("metrics") or {}
    has_all = bool(metrics) and all(v is not None for v in metrics.values())
    if not has_all:
        return "✗"
    if tot is not None and ok is not None and ok >= tot:
        return "✓"
    if tot is not None and ok is not None and ok < tot:
        return "△"
    return "✗"

ready = 0
for sk in stage_keys:
    cells = [classify(run_dir / sk / f"run{k}.json") for k in range(1, repeats + 1)]
    if cells.count("✓") == repeats:
        ready += 1
    print(f"{sk}: {''.join(cells)} ({cells.count('✓')}/{repeats} strict)")
print(f"STRICT READY: {ready}/{len(stage_keys)} stages ({repeats} repeats each)")
PY
```

When reporting progress to the user, show **strict ✓ counts** (and △/✗
gaps), not only `tune.py status` `ok/total`.

### Re-run policy for △ and ✗

- **△ partial** (e.g. videoamme_talker `15/20`): treat as **failed for
  calibration** — re-run that pytest repeat until ✓ or exhaust retries.
- **✗ no metrics**: re-run after GPU cleanup; check OOM / missing
  `*_results.json` in `_pytest/<test>/run{k}.log`.
- Do **not** skip a bad repeat because other repeats for the same stage
  already passed — worst-of-N requires **all N** valid observations.

## Models
Each supported model has a config under `models/<name>/`:
- `config.yaml` — hf model ids, datasets, default venv, test globs,
  per-test extra env, stage-key naming, and `metric_sources` (per-test
  result-JSON paths that tune.py reads to get metric values)
- `stages.yaml` — generated by `tune.py discover --model <name>`

List what's configured:
```
python .claude/skills/tune-ci-thresholds/tune.py models-list
```
Today: `qwen3-omni-v1`, `s2-pro-v1`. To add another model,
drop in a new `models/<name>/config.yaml` and run `tune.py discover
--model <name>`. No Python code changes needed unless the new model
emits metrics with a
constant-naming convention not covered by `match_metric()` in `tune.py`
— in that case the matcher has to grow first.

## Prerequisites (I verify, I do not create)
- Running inside the CI-reproduction container (image
  `frankleeeee/sglang-omni:dev` or equivalent). The container name
  is not checked — rely on the image being correct.
- Venv ready; default path comes from the selected model's config.yaml,
  overridable via `--venv-python` or `$TUNE_VENV_PYTHON`
- Branch checked out, dependencies installed
- Model weights and datasets from the config cached locally. During
  `run`, precheck lists each selected stage's required assets as `✓` /
  `✗`; standalone `precheck` checks all configured assets. On any miss,
  it prints the exact
  `huggingface-cli download …` commands to run.
- Env vars under `auto_env` in the model's config.yaml are set
  automatically at tune.py startup. The user does NOT need to `export`
  them. Proxy env vars (`http_proxy` etc.) are left alone — the tests'
  own `disable_proxy()` helper strips them for loopback calls, matching
  real CI.
- No GPU processes holding memory at **precheck** time. If all GPUs are
  busy, precheck fails with the busy PID list and the user must free them.
  **During `tune.py run`**, the tool runs `delete_gpu_process.sh` and
  waits until each selected GPU is **≤ 2048 MiB** before every pytest
  invocation and retry — this matches CI's per-stage cleanup, but only
  inside an active calibration run. Precheck itself never kills processes.

If anything's off, `precheck` fails with an actionable message; fix it
yourself and retry.

## Invocation
- `/tune-ci-thresholds` — default model, all stages, 5 repeats
- `/tune-ci-thresholds --model qwen3-omni-v1 --stages mmsu_accuracy --repeats 3`
- `/tune-ci-thresholds --resume <run-dir>` — continue an interrupted run

Common Qwen3-Omni V1 presets:
```
# Full threshold stages, excluding docs smoke tests.
python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 run \
  --stages mmmu,mmmu_talker,mmsu,mmsu_talker,tts,videoamme,videoamme_talker,videoamme_talker_tp2,videomme,videomme_talker \
  --repeats 5 --output-dir .tune-runs/<timestamp>_qwen3-omni-v1_cuda-graph_no-docs_r5

# FP8 CI stage 11.
python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 run \
  --stages videoamme_talker_tp2 \
  --repeats 5 --output-dir .tune-runs/<timestamp>_qwen3-omni-v1_fp8_stage_11_r5
```

## Environment and networking notes
- Some CI-reproduction hosts need outbound network proxies or a
  HuggingFace mirror. Keep those values environment-specific and do not
  commit real proxy hosts, ports, usernames, tokens, or personal paths
  into this skill.
- Prefer explicit environment variables in the same shell command that
  starts `tune.py` when a long run may be backgrounded. Use placeholders
  in docs and replace them only in the local shell:
  `TUNE_VENV_PYTHON=<venv-python>`,
  `ALL_PROXY=<proxy-url>`,
  `HTTP_PROXY=<proxy-url>`,
  `HTTPS_PROXY=<proxy-url>`,
  `NO_PROXY=localhost,127.0.0.1,::1`,
  `HF_ENDPOINT=<hf-endpoint>`,
  `HF_HOME=<hf-cache-dir>`,
  `OMNI_CI_HOME=<ci-slice-dir>`,
  `UV_INDEX_URL=<pypi-mirror>`,
  `UV_CACHE_DIR=/github/home/.cache/uv`, and
  `HF_HUB_DISABLE_XET=1` when the environment needs them.
- Do not wrap pytest with `proxychains4`: it can proxy loopback health
  checks and make local server startup look broken. Use proxy env vars
  plus `NO_PROXY` for local addresses.
- If HuggingFace cache locks appear, inspect active pytest/server/download
  processes first. Only stop processes from the current calibration run.

## CI Environment Alignment and Server Startup Debugging

Calibration must reproduce the **same runtime layout as GitHub Actions omni-setup**,
not merely run the same pytest command.

### Cache layout (matches `.github/actions/omni-setup`)

| Scope | Path | Notes |
|-------|------|-------|
| **Global (shared)** | `/github/home/.cache/huggingface` | `HF_HOME`; model weights |
| | `/github/home/.cache/modelscope` | `MODELSCOPE_CACHE` |
| | `/github/home/.cache/uv` | `UV_CACHE_DIR`; PyPI wheels |
| | `/github/home/.cache/flashinfer-jit-cache/` | Host-resident flashinfer-jit-cache **wheel**; download only when pin changes |
| **Per slice (`OMNI_CI_HOME`)** | `<OMNI_CI_HOME>/omni-qwen3` or `omni-s2pro` | Python venv |
| | `<OMNI_CI_HOME>/.cache` | `XDG_CACHE_HOME`; uv/torch compile artifacts |
| | `<OMNI_CI_HOME>/.cache/flashinfer` | Runtime FlashInfer JIT dir — **safe to delete** between runs |
| | `<OMNI_CI_HOME>/.torchinductor` | `TORCHINDUCTOR_CACHE_DIR` |

- **CI Actions runners** use `OMNI_CI_HOME=/github/home/pr-<N>/qwen3` (or `/s2pro`, `/unit`).
- **Calibration host** uses fixed slices from each model's `config.yaml`:
  `omni_ci_home: /github/home/calibration/qwen3` (or `.../s2pro`).
- `tune.py` / `runner.py` apply `auto_env` from `config.yaml` and **override** shell env to match CI.

### Prepare a calibration venv (first time or after deps change)

From repo root on the H20 repro host (`frankleeeee/sglang-omni:dev` semantics):

```bash
# Qwen3-Omni example (S2-Pro: OMNI_CI_HOME=/github/home/calibration/s2pro, venv omni-s2pro)
export OMNI_CI_HOME=/github/home/calibration/qwen3
export HOME=/github/home
export HF_HOME=/github/home/.cache/huggingface
export MODELSCOPE_CACHE=/github/home/.cache/modelscope
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_CACHE_DIR=/github/home/.cache/uv
export XDG_CACHE_HOME=${OMNI_CI_HOME}/.cache
export TORCHINDUCTOR_CACHE_DIR=${OMNI_CI_HOME}/.torchinductor

bash .github/scripts/prepare_omni_venv.sh omni-qwen3
ln -sfn "${OMNI_CI_HOME}/omni-qwen3" ./omni-qwen3
bash .github/scripts/install_flashinfer_jit_cache.sh omni-qwen3
bash .github/scripts/ensure_hf_models.sh omni-qwen3 \
  Qwen/Qwen3-Omni-30B-A3B-Instruct marksverdhei/Qwen3-Omni-30B-A3B-FP8
```

Subsequent runs with unchanged `pyproject.toml`: `prepare_omni_venv.sh` reuses the
venv and only refreshes the editable install (same as CI setup on a new commit).

### Required env vars (auto-set from `config.yaml`)

- `HOME=/github/home`
- `OMNI_CI_HOME`, `XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR` — per-slice paths above
- `HF_HOME`, `MODELSCOPE_CACHE`, `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`
- `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`, `UV_CACHE_DIR=/github/home/.cache/uv`
- `FLASHINFER_DISABLE_VERSION_CHECK=1`

If HF cache lives elsewhere, symlink into `/github/home/.cache/huggingface` rather
than redownloading checkpoints.

### Verify runtime before calibration

```
hostname
python -V
python - <<'PY'
import os, torch, flashinfer
print("OMNI_CI_HOME", os.environ.get("OMNI_CI_HOME"))
print("HOME", os.environ.get("HOME"))
print("XDG_CACHE_HOME", os.environ.get("XDG_CACHE_HOME"))
print("TORCHINDUCTOR_CACHE_DIR", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
print("HF_HOME", os.environ.get("HF_HOME"))
print("UV_CACHE_DIR", os.environ.get("UV_CACHE_DIR"))
print("FLASHINFER_DISABLE_VERSION_CHECK", os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer", flashinfer.__version__, flashinfer.__file__)
PY
```

### CI-like smoke test (before large calibration)

```bash
source /github/home/calibration/qwen3/omni-qwen3/bin/activate
export GITHUB_ACTIONS=true RUNNER_TEMP=/tmp PYTHONPATH=$PWD
export HOME=/github/home OMNI_CI_HOME=/github/home/calibration/qwen3
export HF_HOME=/github/home/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=${OMNI_CI_HOME}/.cache
export TORCHINDUCTOR_CACHE_DIR=${OMNI_CI_HOME}/.torchinductor
export UV_CACHE_DIR=/github/home/.cache/uv FLASHINFER_DISABLE_VERSION_CHECK=1
export NO_PROXY=localhost,127.0.0.1,::1
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x
```

### Known failure signatures

- **Slow safetensors load / disk sleep**: IO pressure — check competing processes, not thresholds.
- **`gen_cutlass_fused_moe_sm90_module` + router timeout**: missing flashinfer-jit-cache in venv.
  Run `.github/scripts/install_flashinfer_jit_cache.sh` (uses host wheel cache).
- **nvcc/ninja referencing wrong Python home**: stale `${OMNI_CI_HOME}/.cache/flashinfer` or
  wrong `HOME`. Delete `${OMNI_CI_HOME}/.cache/flashinfer` only — **never** delete
  `/github/home/.cache/flashinfer-jit-cache/`.
- **GPU cleanup / `[Not Found]` PIDs**: kill container-visible pytest/server processes:
  `pgrep -af "multiprocessing.spawn|sglang_omni_router|sgl-omni serve|pytest|nvcc|ninja"`

After alignment fixes, rerun `tune.py precheck` and the smoke test before resuming calibration.

## Performance optimization checks
- When recalibrating after performance work, first identify what changed
  since the last comparable calibration. Use the previous report's
  provenance commit, the current `precheck.json` commit, or a
  user-provided baseline, then inspect the commit range before judging
  the numbers:
  ```
  git log --oneline <previous-calibration-commit>..<current-calibration-commit>
  git diff --stat <previous-calibration-commit>..<current-calibration-commit>
  ```
- From that range, list the performance-sensitive changes and their
  expected enablement signals. Examples: CUDA Graph replay, torch.compile,
  fused kernels, batching/concurrency changes, cache changes, scheduler
  changes, or preprocessing/audio/video pipeline changes.
- Do not infer that an optimization is active from config alone. For
  every relevant optimization, look for runtime evidence in logs, metrics,
  or profiler output that proves the optimized path actually ran. For
  example, CUDA Graph may require `cuda graph: True` decode logs; a future
  torch.compile change may require compile/cache-hit logs or other
  project-specific evidence.
- If performance is unexpectedly flat or worse, inspect both configuration
  and propagation through server args, runners, schedulers, and stage
  factories before applying thresholds. An optimization being configured
  and the optimized path actually being used are different things.
- In the final report, separate accuracy, WER, and speed conclusions.
  Explain which stages match the expected optimization gains and which
  remain dominated by other work such as preprocessing, long prefill,
  audio synthesis, ASR, or video decoding.

## Monitoring, failures, and completeness (mandatory)

### Agent polling — never blind-wait
- **Maximum idle poll interval: 120 seconds (2 minutes).** Never use
  `block_until_ms` ≥ 50 minutes or any equivalent long sleep while a
  calibration run is active. Long blind waits hide server crashes and
  waste hours.
- While `tune.py run` is in progress, **every 120s at most**:
  1. Run `python tune.py status --run-dir <run-dir>` and read JSON.
  2. Tail `<run-dir>/run.log` and the active
     `<run-dir>/_pytest/<test>/run{k}.log` (last ~30 lines).
  3. Report **strict** progress (✓/△/✗ per stage, see strict audit
     above) **and** `tune.py` `ok/total` / GPU memory. Never cite
     `ok/total` alone as "calibration progress" — it includes △ partial.
- If `status` shows `pytest_active: false` but completeness is not
  `complete: true` and the last log lines show **crash / OOM / server
  startup failure**, do **not** keep waiting — immediately resume:
  ```
  python tune.py --model <M> run --output-dir <run-dir> --resume
  ```
- If GPU memory is **> 2048 MiB** on any GPU needed for the next run,
  do not start another pytest — wait for `tune.py` cleanup or run
  `status` until memory drops.

### tune.py built-in safeguards (v0.4+)
- **GPU hard gate (< 2 GiB):** no pytest restart unless **every selected
  GPU** has `memory.used <= 2048 MiB` and no compute apps. Enforced at:
  1. `_ensure_gpus_free()` — kill stale processes, poll up to 10 min
  2. `_pick_gpus_for_launch()` — select GPUs only after cleanup
  3. `_launch_gpu_gate()` — recheck 3s before `pytest` Popen; if memory
     rose, abort launch and cleanup again
  4. After every run / before every retry — `_ensure_gpus_free()` again
  **Never** launch on 17 GiB stale contexts. If gate fails, the run
  aborts that attempt and retries only after memory drops.
- **Pytest watchdog:** polls every 30s; kills pytest early when the
  log shows server crash signatures (OOM, segfault, router/worker death).
- **Auto-retry passes:** after the first pass, `run` automatically
  re-executes any stage-run whose metrics are incomplete (up to
  `--max-passes`, default 10), with GPU cleanup between passes.
  This retries ✗ (missing metrics) — it does **not** automatically
  reject or re-run △ partial repeats. After each pass, run the **strict
  audit**; manually `--resume` until every stage is N/N ✓.
- **Per-run retries:** up to 4 attempts for OOM / crash / GPU-not-clear
  failures before marking a stage-run incomplete.
- **`status` subcommand:** machine-readable snapshot for agent polling.
- **`report` gate:** refuses to write `report.md` unless **every**
  stage × repeat has complete metrics (`125/125` for full qwen3 ALL×5,
  etc.) — this is tune.py's extraction gate, **not** strict worst-of-N.
  You must still run the **strict audit** before trusting the report
  for apply.

### Completeness is a hard prerequisite for thresholds

Two gates — **both** required before apply:

1. **tune.py gate:** `tune.py status --run-dir <run-dir>` returns
   `"complete": true` (every stage × repeat has extractable metrics).
2. **Strict gate:** strict audit shows **every stage has N/N ✓**
   (full-sample repeats only; no △, no ✗).

- **Never** show the apply prompt (step 9), run `report` for final
  artifacts, or write thresholds unless **both gates pass**.
- Partial runs (△) may exist on disk for debugging but are **never**
  valid calibration artifacts. Do not infer worst-of-N from △ or ✗ runs.
- If tune.py completeness fails after `--max-passes`, relay the
  `missing` list from `status` JSON and `--resume` — do not proceed.
- If tune.py is `complete: true` but strict audit fails, **keep
  resuming / re-running** until strict-ready — do not proceed to apply.

### Resume
- On interruptions or failed stage-runs, resume with the same
  `--output-dir --resume`; completed stage-runs are skipped, incomplete
  ones are purged and re-run automatically.
- **△ partial repeats are not auto-purged** — if strict audit shows △,
  delete the offending `run{k}.json` files for that stage (or the whole
  pytest repeat) and `--resume` so tune.py re-executes them.
- Do not rerun completed ✓ repeats from scratch unless the run directory
  is corrupt.

## Steps I follow
1. Run `python .claude/skills/tune-ci-thresholds/tune.py models-list` to
   discover available models. Then for the selected model, run
   `python tune.py --model <M> stages-list` to read the per-test-file
   bases (e.g. `mmmu`, `mmmu_talker`, `mmsu`, `mmsu_talker`, `tts`, ...) and
   group aliases such as `@accuracy`, `@speed`, and `@wer`.
2. **One-time parameter prompt.** If the invocation omits `--model`,
   `--stages`, or `--repeats`, collect missing fields from the user
   exactly once. After this, do not ask the user anything else for
   the rest of the run.

   Use two mechanisms together:

   **A. Plain text prompt for `stages`** — because the base list
   (up to 6+) does not fit in AskUserQuestion's 4-option cap. Print
   a single message listing **every** base from
   `tune.py --model <M> stages-list`, then wait for the user's reply
   on the next turn. Format:
   ```
   Which tests should I calibrate? Reply with one or more of:
     ALL                          (every stage)
     mmmu                         tests/test_model/test_qwen3_omni_mmmu_ci.py — acc + speed
     mmmu_talker                  tests/test_model/test_qwen3_omni_mmmu_talker_ci.py — acc + wer + speed
     mmsu                         tests/test_model/test_qwen3_omni_mmsu_ci.py — acc + speed
     mmsu_talker                  tests/test_model/test_qwen3_omni_mmsu_talker_ci.py — acc + wer + speed
     videomme                     tests/test_model/test_qwen3_omni_videomme_ci.py — acc + speed
     videomme_talker              tests/test_model/test_qwen3_omni_videomme_talker_ci.py — acc + wer + speed
     videoamme                    tests/test_model/test_qwen3_omni_videoamme_ci.py — acc + speed
     videoamme_talker             tests/test_model/test_qwen3_omni_videoamme_talker_ci.py — acc + wer + speed
     videoamme_talker_tp2         tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py — acc + wer + speed
     tts                          tests/test_model/test_qwen3_omni_tts_ci.py — speed + wer
   Shortcuts: @accuracy, @speed, @wer (metric-group aliases).
   Combine with commas (e.g. "mmmu,mmsu" or "mmmu,@wer").
   ```
   Parse the user's free-text reply (trim whitespace, split on commas)
   and pass verbatim to `--stages`; `tune.py` handles expansion.

   **B. AskUserQuestion for `model` and `repeats`** — both are small
   finite sets. Put both in a single AskUserQuestion call (two
   questions). Skip any field already specified by the invocation.
     - `model`: list the names from `models-list`. If only one is
       available and no `--model` given, skip asking (just use it).
     - `repeats`: options `1 (smoke)` / `2` / `3` / `5 (default)`.

   If the invocation already has `--stages`, `--model`, and
   `--repeats`, skip step 2 entirely.

   When passing `--stages` to `tune.py run`, bases (`mmmu`),
   exact stage keys (`mmmu_accuracy`), and `@group` aliases are all
   accepted and expanded automatically.
3. Run `python tune.py --model <M> precheck --output-dir <run-dir>`.
   On failure, relay the message verbatim and stop.
   **`<run-dir>` must live under `.tune-runs/<timestamp>_<label>/`** at
   the repo root (e.g. `.tune-runs/20260423T050000Z_mmsu_r3/`). That
   path is already gitignored. Do NOT point `<run-dir>` inside
   `.claude/skills/` or anywhere else under version control — run
   artifacts can be large and must not leak into commits.
4. State plan in one line:
   `Running <M>: <stages>, <N> repeats, est. <T>.`
   No further confirmation.
5. **Before** launching run, tell the user the output dir and the log
   paths, plus the **2-minute polling contract**:
   ```
   Output dir: <run-dir>
   tail -f <run-dir>/run.log                               # tune.py progress
   tail -f <run-dir>/_pytest/<test>/run1.log               # pytest subprocess
   Agent polls every ≤120s:
     python tune.py status --run-dir <run-dir>
   ```
   Then run `python tune.py --model <M> run --stages ... --repeats N
   --output-dir <run-dir>`. While the subprocess runs, poll with
   `status` every **≤120s** — never blind-wait ≥50 min. On crash or
   incomplete metrics, `--resume` immediately.
6. When `tune.py run` exits 0, verify **both** gates:
   - `python tune.py status --run-dir <run-dir>` → `"complete": true`
   - Strict audit → every stage **N/N ✓** (see "Strict worst-of-N")
   If strict audit fails, `--resume` (or targeted re-runs) until it
   passes — **do not** open `report.md` for final threshold work yet.
   When both pass, run `python tune.py report --run-dir <run-dir>` if
   needed, then open `<run-dir>/report.md`. In the report narrative,
   note any △ runs that were superseded by successful re-runs.
7. For every `{{CONTEXT:<stage_key>}}` placeholder:
   a. Load `models/<M>/stages.yaml`; find that stage's `test` path and
      `context_vars`.
   b. Read the test file; extract the literal numeric value of each
      listed constant (e.g. `MAX_SAMPLES = 2000` → `2000`).
   c. Load `precheck.json` for GPU count + model.
   d. Replace the placeholder with one line, e.g.:
      `— <N>× <gpu_model> from precheck.json, 2000 samples,
      max_tokens=32, concurrency=8, 5 runs`.
      If the stage is the docs stage (no threshold constants), write
      `— <N>× <gpu_model>, docs smoke, <N> runs`.
   e. If a context var is not found in the file, write `?`. Never
      guess or copy from another stage.
8. Tell the user the report path. Treat `<run-dir>/report.md` as the
   canonical calibration artifact: it must keep the full per-run tables,
   worst-of-N rows, provenance, context lines, and (after apply) the
   applied-changes table. Do not replace it with a lightweight summary.
9. **Apply prompt — strictly after the entire run is done AND both
   completeness gates pass.** This prompt is the LAST thing the skill
   does, and must only fire once ALL of the following have completed
   for the whole `--stages` set:
   `tune.py run` has exited with exit code 0,
   `tune.py status --run-dir <run-dir>` shows `"complete": true`,
   **strict audit shows every stage N/N ✓** (full-sample repeats —
   e.g. 25/25 stages × 5 for full qwen3 ALL),
   `report.md` has been written, every `{{CONTEXT:...}}` placeholder in
   step 7 has been resolved, and step 8 has shown the user the report
   path. Never ask between stages, between repeats, on partial failure,
   or while any pytest subprocess is still alive — the user may be
   running unattended for an hour+ and must not be interrupted mid-run.
   If the run was aborted, either completeness gate failed, any stage has
   △/✗ repeats, or any stage-run is missing metrics, skip step 9
   entirely.

   Use AskUserQuestion to ask exactly once which **apply mode** to use:
     - `report` — only the report, no test files touched
     - `smart` — auto-apply accuracy and WER worst-of-N; auto-tighten
       speed thresholds; ask only for speed metrics that would loosen
     - `full` — write worst-of-N for every metric, no further prompts
   If the user picks `report`, stop without touching any file.

   For `smart` and `full`, first run
   `python tune.py apply-plan --run-dir <run-dir>` to get a JSON with,
   per metric: `source_kind` (bare / nested), `symbol`, `subkey`,
   `concurrency`, `worst_op`, `per_run_raw`, `worst_raw`, `worst_rounded`
   (display-only), `write_value` (the literal to write), `current_raw`,
   and `direction` (`tightens` / `loosens` / `equal` / `unknown`).

   **Which value to write:**
     - **`wer`:** `write_value` = `ceil(worst_raw, 4 dp)` — never round
       down or to `display.digits` (e.g. 0.02387640 → 0.0239, not
       0.023876404494382022 or 0.0238). Write into `*_MAX` /
       `*_CORPUS_MAX`; CI tests derive the assertion threshold via
       `apply_wer_slack(reference)` (×1.25).
     - **`accuracy`:** `write_value` = `worst_raw` exactly into
       `*_MIN_ACCURACY` — no post-calibration slack multiplier.
       Report percentages use 2 decimal places for readability only.
     - **`speed`:** use `write_value` from apply-plan (rounded unless that
       would tighten beyond `worst_raw`). Never re-round or multiply by
       `scale`.

   Bounded write rules (enforced in `write_value`):
     - `worst_op == "min"`: written value must be `<= worst_raw`
     - `worst_op == "max"`: written value must be `>= worst_raw`
     If display rounding would violate either bound, `write_value` falls
     back to `worst_raw` with full precision.

   **Mode `full`**: for every metric in every non-docs stage, edit the
   test file using the rules in (b) below, no questions asked.

   **Mode `smart`**: classify each metric:
     - **auto-apply** iff `stage_group` in (`accuracy`, `wer`), OR
       (`stage_group == "speed"` AND `direction == "tightens"`).
       Edit using rules in (b).
     - **auto-skip** iff `direction == "equal"` (nothing to do).
     - **interactive** otherwise — i.e. any `speed` metric that would
       `loosen` the threshold. For each interactive metric, fire
       AskUserQuestion (one per metric) showing:
         - the per-run raw values from `per_run_raw`
         - the current literal in the test file (`current_raw`)
         - the proposed value (`write_value` — full-precision for wer/acc)
         - direction tag
       with options:
         1. `Keep current` — leave the literal as-is
         2. `Apply worst-of-N (<write_value>)` — write `write_value`
         3. `Custom value` — the user supplies a number; write it
            verbatim after validating it parses as a float
       Always include the "Other" free-text fallback (the
       AskUserQuestion harness adds it automatically). If the user gives
       a custom numeric value, validate that it parses as a float and
       write exactly that raw value (not the display-scaled value).

   (b) **Edit rules** (used by both `full` and `smart`'s auto-apply
   path, and after the user accepts in interactive prompts):
     - Write **`write_value`** from `apply-plan` — never `worst_rounded`
       directly, and never re-format with `display.digits`.
     - **Bare `source`** (no `[...]`), e.g. `MMMU_MIN_ACCURACY`:
       replace the RHS literal of `MMMU_MIN_ACCURACY = <old>` with
       `write_value`.
     - **Nested `source`**, e.g. `_MMMU_P95['throughput_qps']`:
       use the `concurrency` field from `apply-plan` output, then
       replace the entry under
       `_MMMU_P95[<C>]["throughput_qps"]` with `write_value`. If
       `concurrency` is null (no `CONCURRENCY` symbol in the test file)
       and the dict has a single key, fall back to that key; if multiple
       keys exist and `concurrency` is null, abort the apply step for
       that metric and warn the user.
     - For any metric whose `direction` came back `unknown` (couldn't
       parse current literal — usually means the test file diverged
       from `stages.yaml`), do not edit; warn and continue.

   After all edits across all stages, do two things:

   **(c) Append an "Applied changes" section to `<run-dir>/report.md`**
   so the artifact records what was actually written. Use the Edit
   tool to insert this block immediately before the existing
   `## Provenance` heading:

   ```
   ## Applied changes

   | Stage | Metric | Old | New | Direction |
   |-------|--------|-----|-----|-----------|
   | <stage_key> | <source> | <current_raw> | <new_raw> | <direction> |
   ...
   ```

   Rules:
     - Include only metrics that were actually edited. Rows for
       "Keep current" choices, mode-`report` runs, and `equal` /
       `unknown` skips are omitted.
     - `Stage` is the `stage_key` (e.g. `mmsu_accuracy`).
     - `Metric` is the literal `metric.source` from `apply-plan` —
       bare (`MMSU_MIN_ACCURACY`) or nested
       (`_MMSU_P95[8]['throughput_qps']` with the resolved
       concurrency substituted in).
     - `Old` / `New` are **raw** numeric values (matching what's in
       the test file, not display-scaled). Trim trailing zeros for
       readability.
     - `Direction` describes the effect on CI strictness — derived
       from `worst_op` and the sign of `new - old`:
         - `worst_op == "min"` (threshold is a lower bound, e.g.
           `throughput_qps`): `new > old` → `tightens`,
           `new < old` → `loosens`.
         - `worst_op == "max"` (threshold is an upper bound, e.g.
           `latency_mean_s`, `rtf_mean`, `WER_..._MAX`): `new < old`
           → `tightens`, `new > old` → `loosens`.
       Format the cell as `tightens (Δ%)` or `loosens (Δ%)` where
       `Δ%` is the signed percent change of the **raw** value
       relative to the old raw value, e.g. `tightens (+2.1%)`,
       `loosens (-7.9%)`. Use one decimal place. Direction MUST come
       from `worst_op` (not from sign-of-Δ alone) — for `max`-bounded
       metrics, a negative Δ% is a tightening.
     - If nothing was edited (all kept / all skipped), do not append
       the section at all.

   **(d) List every changed `<file>:<symbol> = <new>` tuple in one
   chat message**. If the user has explicitly authorized commit/push,
   continue to the version-control step below; otherwise stop.

10. **Optional version-control step — only with explicit user
    authorization.**
    - Keep `.tune-runs/` local and uncommitted.
    - If the calibration evidence should be committed, copy the final
      `<run-dir>/report.md` (after context replacement and any
      applied-changes section) to a stable path under `docs/calibration/`
      and commit that raw observation report. A short summary under
      `docs/` is optional, but it must not replace the raw per-run
      report.
    - Commit only threshold/test edits, skill/config changes, and
      requested calibration reports / summaries under `docs/`.
    - Run repository pre-commit hooks normally; do not bypass hooks.
    - Push only the current feature/calibration branch, never `main`.
    - Provide a PR description with: summary, calibration run directory,
      CUDA Graph evidence, worst-of-N highlights, threshold-apply policy,
      and test/pre-commit verification.

## What I do not do
- Treat `tune.py status` `ok/total` or `complete: true` as strict
  worst-of-N readiness — always run the strict audit (✓ = full samples).
- Include △ partial or ✗ failed repeats in worst-of-N calculations or
  apply decisions.
- Set up container / venv / caches during an ordinary calibration run.
  Exception: if a CI-equivalent smoke test proves that local server
  startup is not comparable to CI, pause calibration and fix environment
  alignment first (see "CI Environment Alignment and Server Startup
  Debugging").
- Check out branches or install packages unless the user explicitly asks
  or CI-alignment debugging proves a missing dependency — use
  `.github/scripts/prepare_omni_venv.sh` and
  `.github/scripts/install_flashinfer_jit_cache.sh` (not ad-hoc wheel URLs).
- Run `apply_slack` or generate patch files
- Commit or push without explicit user authorization
- Edit test files outside of the explicit apply prompt (step 9)
- Write ad-hoc apply scripts that re-round metrics — always use
  `apply-plan`'s `write_value` field when editing test files
- Round WER or accuracy thresholds to `display.digits` (report-only)
- Ask mid-run for confirmation. (I may ask once up front for missing
  model/stages/repeats — step 2 — and once at the end for the apply
  prompt — step 9. No other questions.)

## Files in this skill
```
.claude/skills/tune-ci-thresholds/
├── SKILL.md
├── tune.py                              # CLI; METRIC_SPECS + JSON extractor
│                                        # subcommands: run, report, status,
│                                        # apply-plan, precheck, discover
└── models/
    ├── qwen3-omni-v1/                   # v1 pipeline (qwen3-omni)
    │   ├── config.yaml
    │   └── stages.yaml
    └── s2-pro-v1/                       # v1 pipeline (FishAudio S2-Pro,
        ├── config.yaml                  #   uses per-test-file `variants`)
        └── stages.yaml
```

## How metric values get read
tune.py spawns pytest with `--basetemp=<fresh dir>/_pytest/<test>/basetemp_run{k}`.
Each test writes its result JSON (`mmmu_results.json`, `speed_results.json`, …)
under that dir at a deterministic path. After pytest exits, tune.py
loads those JSONs and pulls each metric by dotted key. Nothing is
parsed from stdout — the test doesn't need to print anything.

For `tmp_path`-based tests (MMMU, MMSU, VideoMME, VideoAMME and their
talker variants), `discover` **auto-infers** `json_file`, `paths`, and
`sample_counts` from the test file's AST using convention-based defaults.
MMSU's non-standard JSON layout (`speed_metrics.*` instead of `speed.*`)
is detected automatically via its benchmark module import. When a test
has no `metric_sources` entry in `config.yaml`, discover prints a
suggested config entry and uses the inferred values as fallback — so
stages.yaml is correct even without a config update.

The `metric_sources` block in `config.yaml` declares, per test file:
- `json_file` — path relative to pytest basetemp (the default file
  for every metric in this test)
- `paths` — `{metric_key: "dotted.path"}`, or `"file::dotted.path"`
  inline if the metric lives in a different JSON than the default
- `variants` — *optional*; for tests that produce parallel result
  trees (e.g. nonstream / stream voice-clone in the same pytest run).
  Each variant entry has `constant_filter` (regex matched against the
  bare constant name with any leading underscore stripped),
  `json_file`, `sample_counts`, `paths` — same shape as the
  file-level fields. Constants matching a variant's filter are
  routed only to that variant; stage keys become
  `<base>_<variant>_<group>` (e.g. `tts_nonstream_speed`,
  `tts_stream_wer`). The bare base (`tts`) still resolves to all
  variants via the alias system.

Config.yaml entries always override auto-inferred values. For TTS tests
(`tmp_path_factory`-based), auto-inference is not available — config.yaml
entries are required.

## Regenerating stages.yaml
If a test file's sha256 no longer matches `models/<M>/stages.yaml`,
`run` will warn. Regenerate with:
```
python tune.py --model <M> discover
```
This is deterministic (AST + config lookup, no LLM calls). For
`tmp_path`-based tests, discover auto-infers metric_sources from the
test file's AST and prints suggested config.yaml entries for any test
not yet in config. It also validates existing config entries against
the inferred values.

## Adding a new model
1. Create `models/<new-name>/config.yaml` mirroring `qwen3-omni-v1/config.yaml`.
   For `tmp_path`-based tests (MMMU, MMSU, VideoMME, VideoAMME + talker
   variants), `metric_sources` entries are auto-inferred by discover — you
   can omit them. For TTS tests (`tmp_path_factory`-based), add
   `metric_sources` entries manually (use existing TTS entries as template).
2. Run `python tune.py --model <new-name> discover`. Discover prints
   suggested config.yaml entries for any test not yet in config — copy
   them in for PR reviewability.
3. Any metric that shows up as `NEEDS_CONFIG` means the constant was
   recognized but neither auto-inference nor config provides a path — add
   the dotted JSON key under `metric_sources` and re-run discover.

## Adding a new metric to an existing model
If a new test file adds a threshold constant:
- Matching an existing naming pattern (`*_ACC_MIN`, `*_WER_MAX_CORPUS`,
  nested `_*_P95[*].<known_key>`) → `discover` picks it up for free.
  For `tmp_path`-based tests following standard conventions, the JSON
  path is auto-inferred. Otherwise, add its JSON dotted key under
  `metric_sources.<test_file>.paths`.
- New nested-dict key (e.g. `_*_P95[*].ttft_ms`) → add to `_NESTED` and
  `METRIC_SPECS` in `tune.py`.
- New naming pattern (e.g. `*_BLEU_MIN`) → extend `match_metric()` and
  `METRIC_SPECS` in `tune.py`.
- Metric lives in a different JSON than the test's default → use the
  `<file>::<dotted.path>` inline form in `metric_sources.<test>.paths`.

Threshold constants whose name `match_metric()` doesn't recognize are
silently ignored — extend `match_metric()` if you add a new pattern.
