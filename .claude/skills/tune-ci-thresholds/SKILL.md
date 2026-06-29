---
name: tune-ci-thresholds
description: Run CI tests N times per stage on the H100 CI-reproduction host, produce a per-metric strict worst-of-N observation report (every stage must have N full-sample repeats), and (on user confirmation) write the worst-of-N values back into the test files as new baselines. Each new user calibration request MUST use a fresh UTC-timestamp --output-dir on current HEAD; --resume only when explicitly continuing the same interrupted session on the same commit. Reports must include the full calibration commit SHA. Host-specific repo/venv/cache paths live in hosts/*.yaml (CI doc paths are reference only). Currently supports qwen3-omni-v1, qwen3-asr-v1, and tts; extensible via models/<name>/config.yaml.
---

# tune-ci-thresholds

## Scope
This skill is for the H100 CI-reproduction host (the same image
CI uses,
`crpi-n6adu6llixz83q37.cn-hangzhou.personal.cr.aliyuncs.com/hongccc/sglang-omni:dev`,
a CUDA-13 image; the container name varies). The host is 2× H100.
Numbers from environments that differ meaningfully from CI (different
GPU model, different image, different pinned sglang/torch) are not
comparable and must not drive threshold changes. If you just want to
run the tests locally, use pytest directly — this skill is not for
that.

**Host layouts are not fixed to the CI doc paths.** Paths in
`models/*/config.yaml` describe the **GitHub Actions reference**. Inside a
repro container, `hosts/<name>.yaml` gives the **in-container paths**
`tune.py` uses (see **Calibration host profiles**). User-provided paths in
chat override the YAML.

The skill is observation-first: it runs tests N times and produces a
**strict worst-of-N** report. After the report is shown, it offers a
one-shot **apply step** that writes the worst-of-N values directly
into the test files as the new P95 baselines and accuracy / WER
thresholds — **only** if the user explicitly confirms. The skill
still does NOT re-run `apply_slack` separately, generate patch files,
or commit / push anything; if the user rejects the apply prompt, the
test files stay untouched and the user picks values manually from
the report.

## Fresh calibration session (P0 — non-negotiable)

**Every new calibration request from the user starts a new session on the
current `HEAD` commit.** Resume is for **interruptions only**, never for
“we already have an old run dir”.

### Agent rules

| User says | Agent must |
|-----------|------------|
| “Calibrate …” / “Run tune-ci-thresholds …” (no `--resume`) | Create **new** `.tune-runs/<UTC-timestamp>_<label>/`, run `git rev-parse HEAD`, calibrate **that** commit |
| “Continue / resume run dir X” | `--resume --output-dir X` only; **same commit** as `plan.json` `calibration_git_sha` |
| Show results / apply thresholds | Use **only** the run dir from **this** session; never open an older `report.md` |

**Forbidden:**

- Reusing `.tune-runs/20260622T…` (or any prior dir) when the user asked for a
  **new** calibration on a **newer** commit.
- Presenting a report whose `calibration_git_sha` ≠ current `HEAD` unless the
  user explicitly asked to resume that same session.
- Saying “calibration complete” while any stage artifact lacks `git_sha` or
  records a different commit (`strict-audit` → `GIT PROVENANCE: FAIL`).
- Mixing fresh Qwen3-ASR reruns with stale Moss artifacts from an earlier
  commit in one apply.

### New run directory (mandatory naming)

```bash
RUN=".tune-runs/$(date -u +%Y%m%dT%H%M%SZ)_<short-label>"
mkdir -p "$RUN"
git rev-parse HEAD   # tell the user which commit you are calibrating
python tune.py --model <M> precheck --output-dir "$RUN"
python tune.py --model <M> run --stages <S> --repeats <N> --output-dir "$RUN"
```

Do **not** pass `--resume` on the first `run` of a new user request.
`tune.py` **refuses** `run` without `--resume` when `plan.json` already exists
in `--output-dir`.

### Resume (interruption recovery only)

Use `--resume` only when:

1. The user explicitly asked to continue an **existing** run dir, **or**
2. The same session was interrupted (pytest crash, agent timeout) and the
   commit has **not** changed.

On `--resume`, `tune.py` errors if `HEAD` ≠ `plan.json` `calibration_git_sha`.
If the user moved to a newer commit, create a **new** run dir instead.

### Report must identify the commit

Every `report.md` must show the **full** calibration commit at the top:

```markdown
**Calibration commit:** `5aa60e4bc1274e968fc11be557ca99ff9a4dff00`
```

The user must be able to verify which commit was calibrated without guessing
from the run-dir date. `strict-audit` prints `calibration_git_sha` and
`GIT PROVENANCE: ok|FAIL`; both gates must pass before report or apply.

## Zero-tolerance completeness contract (P0 — non-negotiable)

**Calibration is not done until every stage × every repeat × every tracked
metric is strict-complete (✓).** There are no exceptions, no “good enough”
partial matrices, and no moving on while gaps remain.

### Agent poll interval — never blind-wait > 2 minutes (P0 — non-negotiable)

While `tune.py run` is active, the agent **must** check calibration progress
**at least every 120 seconds (2 minutes)**. This is a hard ceiling, not a
guideline.

| Forbidden | Required instead |
|-----------|------------------|
| `Await` / `block_until_ms` ≥ 120000 on calibration | Poll with `block_until_ms` ≤ 120000; prefer 60–90s during active pytest |
| Sleeping “until the stage finishes” | `tune.py status` + `tune.py strict-audit` + tail active `_pytest` log + GPU |
| Reporting only `status ok/total` | Report **`strict-audit` N/N ✓** (includes `expected_samples` gate) |
| Trusting old run dirs without `strict-audit` | Always run `python tune.py strict-audit --run-dir <run-dir>` before report/apply |
| Reusing an old `--output-dir` for a new user calibration request | New UTC-timestamp dir + current `HEAD`; `--resume` only when user asks |
| Reporting from stale `report.md` | Regenerate on current run dir after `STRICT + GIT: READY` |

**Every poll cycle (≤120s):**

```bash
python .claude/skills/tune-ci-thresholds/tune.py status --run-dir <run-dir>
python .claude/skills/tune-ci-thresholds/tune.py strict-audit --run-dir <run-dir>
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -30 <run-dir>/_pytest/<active-test>/run{k}.log
```

### Sample scope gate (`expected_samples`)

Strict ✓ requires **both**:

1. All tracked metrics non-null; `sample_counts.ok == sample_counts.total`.
2. When `stages.yaml` lists `expected_samples`, **`total` must equal that
   value** (`tune.py discover` reads it from the test-file constants).

Always use **`tune.py strict-audit`** — not hand-rolled scripts that only
check `ok == total`.

**TTS sample scopes** (from test constants, written to `stages.yaml` by
`discover`):

| Stage group | Constant |
|-------------|----------|
| Qwen3 ASR | `SEEDTTS_ASR_CORRECTNESS_SAMPLES` |
| TTS non-stream WER / speed / UTMOS; stream WER / speed | `SEEDTTS_EN_FULLSET_SAMPLES` (or full set when `STREAMING_BENCHMARK_MAX_SAMPLES` is `None`) |
| TTS similarity | `TTS_SIMILARITY_MAX_SAMPLES` |

Use a **fresh `--output-dir`** per calibration session (see **Fresh calibration
session** above); do not mix runs from different commits or reuse an old dir
when the user requested a new calibration. Do not mix runs from
different `--stages` scopes into one apply without a full strict audit pass.

### Success criterion (all must hold)

For model `M`, `--repeats N`, and stage list `S` from `plan.json`:

```
∀ stage ∈ S, ∀ run k ∈ 1..N:
  run{k}.json exists
  ∀ metric m in stages.yaml[stage].metrics: metrics[m] is non-null
  sample_counts.ok == sample_counts.total (both non-null)
```

Equivalently: **strict audit shows N/N ✓ for every stage** and
`STRICT READY: |S|/|S|`.

### Mandatory re-run on any gap

| Gap | Agent action (immediate) |
|-----|--------------------------|
| **—** not run yet | Keep `--resume` until filled |
| **✗** missing / null metrics | Purge that `run{k}.json` (or whole pytest repeat); fix root cause; `--resume` |
| **△** partial samples (`ok < total`) | Purge; `--resume` — never use for worst-of-N |
| **Metric extraction warnings** in `run.log` (`file missing`, `ALL metrics None`) | **Stop `tune.py`**; fix `config.yaml` / test JSON output; purge affected repeats; `--resume` |
| **One stage ✓ but sibling stage ✗** from same pytest file | Purge **entire pytest repeat k** for that test file (all its stages share one invocation) |

**Forbidden:** continuing calibration, starting the next model, writing
`report.md`, or discussing apply while **any** stage has `< N` strict ✓
repeats. **Forbidden:** treating pytest PASS as success when
`run{k}.json` metrics are null or partial.

**⚠️ `discover` can mislabel `expected_samples` for full-dataset stages.** For a
stage whose only `context_var` is `CONCURRENCY` (no `MAX_SAMPLES`) — e.g.
`mmmu_accuracy` / `mmmu_speed`, `mmsu_accuracy` / `mmsu_speed` — `discover`
wrongly sets `expected_samples = CONCURRENCY` (**16**). Because `tune.py` uses
`expected_samples` for the completion gate, those stages look **perpetually
incomplete** and trigger futile `--resume` retries (mmsu is the 2000-sample slow
stage, so this wastes a lot of GPU time). **After `discover`, verify
`expected_samples` matches the real dataset size** (mmmu = **50**, mmsu =
**2000**) and fix the `stages.yaml` literals **before** `run`.

### No `-x` during calibration (P0 — non-negotiable)

`tune.py run` **must never** pass pytest `-x` / `--exitfirst`.

Calibration observes worst-of-N metrics; threshold assertions inside tests
may fail with `pytest rc=1` and that is **expected**. Early exit skips later
tests in the same file (WER, similarity, …) and produces null metrics —
the agent then wastes hours in blind retry passes that cannot succeed.

| Context | `-x` allowed? |
|---------|----------------|
| `tune.py run` (calibration) | **Never** |
| Local dev / CI gating / ad-hoc pytest | Yes (default in test `__main__` blocks) |

**Valid incomplete run:** server crash, CUDA OOM (exit 137), timeout — not
“accuracy below threshold”. When metrics are missing after a non-OOM run,
fix test order / JSON persistence / `config.yaml` paths — do not add `-x`.

### Per-test-file rule

Each CI test file (e.g. `test_qwen3_omni_mmmu_talker_ci.py`) produces one
pytest invocation per repeat `k`, feeding **all** of its stages (accuracy,
WER, speed, …). A repeat is valid only when **every** stage fed by that
invocation is ✓. If any one threshold metric is missing, **the whole
repeat is invalid** — purge all stage `run{k}.json` for that test and
re-run.

### Agent enforcement loop (every ≤120s while calibrating — P0)

**Hard rule: the agent must not go more than 120 seconds without a progress
check.** Never use long `Await`/`sleep` while calibration runs.

1. `python tune.py status --run-dir <run-dir>` (includes `strict_ready` summary)
2. `python tune.py strict-audit --run-dir <run-dir>` — **the only** progress
   metric shown to the user
3. If `STRICT READY < total stages` **or** `run.log` shows extraction
   warnings / `sample scope mismatch` → diagnose, fix, purge, `--resume`
4. If `tune.py` exits with **metric extraction HALT** → fix config before
   any further pytest
5. **Blocker fix is same-turn work** — when audit or `run.log` surfaces
   `NEEDS_CONFIG`, null metrics, `expected_samples` mismatch, or missing JSON paths,
   patch `config.yaml` / tests **immediately** in that session; purge the
   affected pytest repeat; do **not** only report the issue and keep calibrating.

`tune.py` **halts immediately** when pytest passes but metric extraction
fails (config / JSON path bug). The agent must fix and `--resume`; never
ignore HALT and start a fresh run directory without user approval.

## Calibration host profiles (mandatory — before precheck)

**Assumes the repro container is already running.** The user starts Docker;
this skill only describes **in-container paths** for `tune.py` / precheck /
calibration. Do not document `docker run`, volume maps, or host-side layout
in host profiles.

**Agent runbook:** `.claude/skills/tune-ci-thresholds/AGENT-PRECHECK.md` —
mandatory environment gate (Gates 0–8) before any `tune.py run`. Agent-facing
only; no Docker or user setup instructions.

Calibration does **not** require CI doc paths (`/sgl-workspace/...`). On a
repro machine, **`tune.py` loads `hosts/<name>.yaml`** and applies
**physical paths directly** to `auto_env` — no symlinks.

**Selection order:** `--host <name>` → `$TUNE_HOST` → autodetect by
`hostname`. List profiles: `tune.py hosts-list`.

When a host profile is active, `tune.py` automatically:

| Override | From host profile |
|----------|-------------------|
| `REPO_ROOT` / pytest `cwd` | `repo_root` |
| venv | `venv_python` → `$TUNE_VENV_PYTHON` |
| `HF_HOME` | `physical.hf_hub` |
| `SEEDTTS_SIM_CACHE_DIR` | `physical.speaker_sim` |
| `OMNI_CI_HOME` (optional) | `physical.omni_ci_home` |

User-provided paths in chat override the YAML; update the host file when
the layout stabilizes.

### What a host profile defines

| Field | Purpose |
|-------|---------|
| `repo_root` | Git checkout inside the container |
| `venv_python` | Calibration venv inside the container |
| `physical.hf_hub` | HuggingFace hub cache path **inside the container** |
| `physical.speaker_sim` | WavLM SV assets directory **inside the container** |
| `agent_policy` | e.g. report env gaps before fixing; max poll interval |

Ensure `mkdir -p /github/home/calibration` exists if FlashInfer /
torchinductor slice paths are used (once per fresh container filesystem).

### Speaker Similarity — checked in precheck

When a host profile is active (or model is `tts` / `qwen3-omni-v1`),
`precheck` verifies `physical.speaker_sim`: `.complete`,
`wavlm_large.pt`, `wavlm_large_finetune.pth`. Bootstrap once if ✗ — see
`speaker_similarity_bootstrap` in the host YAML.

### UTMOS asset — NOT covered by precheck (warm before TTS)

The TTS `tts_utmos` metric downloads `balacoon/utmos` → `utmos.jit` **on demand**
via `benchmarks.metrics.utmos.ensure_utmos_assets`, into
`/github/home/.cache/sglang-omni/utmos`. `precheck` does **not** verify it, so on
an overseas host (e.g. H100) `tts_utmos` fails **mid-run** — `hf-mirror.com` can't
serve the file. Warm it **before** TTS calibration with
`HF_ENDPOINT=https://huggingface.co` by calling `ensure_utmos_assets()` (a raw
`huggingface-cli download` won't satisfy its `.utmos_cache.json` marker):

```bash
HF_ENDPOINT=https://huggingface.co <venv>/bin/python -c \
  "from benchmarks.metrics.utmos import ensure_utmos_assets; ensure_utmos_assets()"
```

### Shipped profile: `sglang-h100-ci` (current / active)

`repo_root` `/data/sglang-omni`, `venv_python`
`/github/home/calibration/omni/bin/python`, `physical.hf_hub`
`/root/.cache/huggingface`, `physical.speaker_sim`
`/root/.cache/huggingface/speaker_sim`, `physical.omni_ci_home`
`/github/home/calibration`. Container image
`crpi-n6adu6llixz83q37.cn-hangzhou.personal.cr.aliyuncs.com/hongccc/sglang-omni:dev`,
GPUs `NVIDIA_VISIBLE_DEVICES=6,7` (2× NVIDIA H100 80GB HBM3). Pins:
**sglang 0.5.12.post1**, **torch 2.11.0** (cu130).

### Adding a new host profile

Copy `hosts/sglang-h100-ci.yaml` → `hosts/<name>.yaml`. Set `hostname`,
`repo_root`, `venv_python`, `physical.*` — **in-container paths only**.
Add venv to `default_venv_python` in `models/*/config.yaml`.

## Two-terminal supervision (mandatory — always)

Every long-running job on the repro host — calibration (`tune.py run`), WER
sweep, eval suite, ad-hoc pytest — uses **exactly two IDE terminal tabs** with
**fixed, non-interchangeable roles**. This split is permanent; never collapse
into one tab, never duplicate content across tabs, never ask the user to paste
`tail -f` themselves. **Agent spawns both tabs** (Shell tool, `block_until_ms: 0`).

```
┌─────────────────────────────────────┐   ┌────────────────────────────────────┐
│  Tab A — SUPERVISION                │   │  Tab B — JOB                       │
│  tail -f <log-path>                 │   │  wrapper script or redirected cmd  │
│                                     │   │                                    │
│  DETAILED LOG (everything verbose)  │   │  PROGRESS SUMMARY ONLY             │
│  • pytest -v -s output              │   │  • START / PASS / FAIL / ABORT     │
│  • router & worker startup          │   │  • current stage name              │
│  • CUDA graph capture progress      │   │  • log path reminder               │
│  • /health, route_completed, HTTP   │   │  • env OK / sweep section headers  │
│  • WER scores, assertions, traces   │   │                                    │
│                                     │   │  NO server lines, NO pytest spam   │
└─────────────────────────────────────┘   └────────────────────────────────────┘
              ▲                                          │
              │         log file on disk                   │
              └──────── verbose via >> log only ─────────┘
                        (never tee on Tab B)
```

### Tab A — Supervision (detailed log)

| | |
|--|--|
| **Command** | `tail -f <log-path>` — for **`tune.py run`**, `<log-path>` is **always** the newest `<run-dir>/_pytest/*/run{k}.log` (use `tail_calibration_pytest.sh`; see below) |
| **Shows** | **All** verbose output — the only place the user reads server/router/pytest details |
| **Spawn order** | **First** — before Tab B |
| **`tune.py run` forbidden** | **Never** `tail -f <run-dir>/run.log` on Tab A while Tab B runs `tune.py run`. `run.log` is tune.py’s milestone tee — **identical to Tab B stdout**. If both tabs show the same lines, Tab A is wrong. |

### Tab B — Job (progress summary)

| | |
|--|--|
| **Command** | Wrapper script, or `cmd >> <log-path> 2>&1` with no stdout tee |
| **Shows** | **Milestones only** — which stage started/finished, pass/fail exit code, where to tail |
| **Spawn order** | **Second** — after supervision tab is running |
| **Forbidden** | `tee`, `2>&1 \| tee`, pytest `-s` to job stdout, any pattern that mirrors Tab A |

Verbose subprocess output **must** go to the log file (`>> log 2>&1`). Tab B stdout
is for the operator to glance at overall progress; Tab A is for supervision.

### Agent checklist (every long run)

0. **Kill stale tabs first** — before spawning, stop any prior supervision/job
   processes (`pkill -f 'tail_calibration_pytest.sh'`, `pkill -f 'tail -f.*_pytest'`,
   sweep script, pytest) so the IDE does not accumulate duplicate terminal tabs.
1. Choose stable `<log-path>`; print it. For **`tune.py run`**, Tab A path is
   `_pytest/*/run{k}.log`, **not** `<run-dir>/run.log`.
2. Spawn **Tab A** (see job-specific command in table below).
3. Spawn **Tab B**: run command (see examples below).
4. Tell the user: **Tab A = pytest/server details, Tab B = tune.py milestones** —
   they must **not** show the same content. If they do, Tab A is tailing `run.log`
   by mistake — kill it and respawn with `tail_calibration_pytest.sh`.
5. Poll with `tail -20` on the **active `_pytest` log** / `tune.py status` every
   **≤120s**; user watches Tab A.

### Calibration (`tune.py run`) — always two tabs, no exceptions

**Never start `tune.py run` with only one terminal.** Never wrap it in
`>> <run-dir>/run.log` — `tune.py` already tees milestones to `run.log`
internally; shell redirect hides Tab B and can race/truncate the log file.

**Tab A must never tail `<run-dir>/run.log` for calibration.** That file mirrors
Tab B exactly (milestone lines only). Pytest/router/CUDA-graph output lives under
`<run-dir>/_pytest/<test>/run{k}.log`. Before `_pytest` exists, Tab A should
**wait** (helper script loops) — do **not** fall back to `run.log`.

| Tab | Role | Agent Shell (`block_until_ms: 0`) |
|-----|------|-----------------------------------|
| **A — supervision** | pytest + router/server verbose | `bash .claude/skills/tune-ci-thresholds/tail_calibration_pytest.sh <run-dir>` — resolves the **active** pytest via running process (`--basetemp`), switches immediately when tune.py starts the next test. **Forbidden:** `tail -f $(ls -t …/run*.log \| head -1)` (sticks on completed logs like failed mmmu run1). |
| **B — job** | tune.py milestone lines (stdout) | `cd /sgl-workspace/sglang-omni && python .claude/skills/tune-ci-thresholds/tune.py --model <M> run ... --output-dir <run-dir>` — **no** `tee`, **no** `>>` |

Spawn **A then B**. Tell the user which tab is pytest/server (A) vs tune
progress (B). The helper script re-points Tab A when a newer `run{k}.log`
appears; if Tab A still looks like Tab B, respawn Tab A with the helper — never
`tail -f run.log`.

### Log path conventions

| Job | Tab A `tail -f` target | Tab B command |
|-----|------------------------|---------------|
| **`tune.py run` (calibration)** | **`bash …/tail_calibration_pytest.sh <run-dir>`** (active pytest log via process) | **`python tune.py ... run`** — stdout only, no redirect |
| WER sweep (qwen3) | `/tmp/wer_ci_qwen3.log` | `bash .github/scripts/run_all_wer_ci_aligned.sh` |
| WER sweep (tts) | `/tmp/wer_ci_tts.log` | (same script; switches log at tts section) |
| Ad-hoc pytest | `/tmp/pytest_<name>.log` | `pytest ... -v -s >> /tmp/pytest_<name>.log 2>&1` |
| Eval suite | `<run-dir>/run.log` | `runner.py run ... >> <run-dir>/run.log 2>&1` |

Reference implementation: `.github/scripts/run_all_wer_ci_aligned.sh` — milestones
to stdout, pytest `>> "$LOG"`.

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

```bash
python .claude/skills/tune-ci-thresholds/tune.py strict-audit --run-dir <run-dir>
```

Example output:

```
qwen3_asr_wer: ✓✓✓✓✓ (5/5 strict, expected=<N>)
tts_moss_stream_speed: ✓✓✓✓✓ (5/5 strict, expected=<N>)
STRICT READY: 8/8 stages (5 repeats each)
```

(`<N>` comes from `expected_samples` in `stages.yaml` — run `discover` after
test constant changes.)

**Do not use hand-rolled Python that only checks `ok == total`** — use
`tune.py strict-audit`, which also enforces `expected_samples`.

When reporting progress to the user, show **strict ✓ counts** (and △/✗
gaps), not only `tune.py status` `ok/total`.

### Re-run policy for △ and ✗

- **Any gap in the N×metrics matrix** — treat as **blocking**; re-run until ✓.
- **△ partial** (e.g. videoamme_talker `15/20`): treat as **failed for
  calibration** — purge and re-run that pytest repeat until ✓ or exhaust retries.
- **✗ no metrics** (null, missing JSON, extraction failed): **stop forward
  progress** if pytest passed; purge; fix config/test output; `--resume`.
- **Partial null** (some metrics present, others null): same as ✗ — purge
  entire pytest repeat for that `k`; fix paths; `--resume`.
- Do **not** skip a bad repeat because other repeats for the same stage
  already passed — worst-of-N requires **all N** valid observations per stage.
- Do **not** advance to the next CI test file while the current file still
  has any repeat `k` with a non-✓ stage among its fed stages.

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
Today: `qwen3-omni-v1`, `qwen3-asr-v1`, `tts`. To add another model,
drop in a new `models/<name>/config.yaml` and run `tune.py discover
--model <name>`. No Python code changes needed unless the new model
emits metrics with a
constant-naming convention not covered by `match_metric()` in `tune.py`
— in that case the matcher has to grow first.

## Environment policy — check first, fix only what precheck proves missing (mandatory)

**Never download or rebuild the calibration environment proactively.**
Every calibration session starts with **read-only alignment checks**; only
run install/download commands for items that precheck (or a failed smoke
test) explicitly marks as missing or misaligned — **or** when the user
explicitly asks you to fix a named gap (e.g. speaker sim warm-cache).

### Resolve host profile first

1. Run `tune.py hosts-list`; load matching `hosts/<name>.yaml` (autodetect
   by `hostname`, or `--host`, or `$TUNE_HOST`). **No symlinks** — `tune.py`
   sets `HF_HOME`, `SEEDTTS_SIM_CACHE_DIR`, `repo_root`, and venv from the
   profile.
2. `cd <repo_root>` for all commands (or rely on autodetect).
3. Run `precheck` — it checks HF assets at `physical.hf_hub`, speaker sim
   at `physical.speaker_sim`, pins, and GPUs.
4. **Report** any remaining gap before bulk fixes unless the user directs a
   specific fix.

### Default workflow (always, before `run`)

1. **Check only** — run `tune.py precheck --output-dir <run-dir>` and read
   every line. Treat `✓` as “leave alone”. Treat `✗` / version mismatch /
   busy GPU as the **only** allowed triggers for a fix.
2. **Refresh code, not the venv** — when the venv path resolves and torch/sglang
   pins match, sync the checked-out branch with:
   `source <venv>/bin/activate && uv pip install -e .`
   Do **not** run `prepare_omni_venv.sh` for this.
3. **Fix one gap at a time** — use the exact command precheck prints (e.g.
   `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download …` for one
   missing checkpoint). Do not batch unrelated installs “just in case”.
4. **Re-run precheck** after each fix until all selected assets are `✓`,
   then start `tune.py run`.

### Forbidden unless precheck / smoke test proves need

| Action | Why forbidden by default |
|--------|--------------------------|
| `prepare_omni_venv.sh` | Only rebuilds from scratch (`rm -rf $OMNI_CI_HOME`) when `pyproject.toml`'s deps-hash changed or the venv is missing/corrupt; otherwise it reuses the slice and only runs `uv pip install --upgrade -e .`. Still don't invoke it by hand when precheck is green — let omni-setup/precheck decide. |
| `ensure_hf_models.sh` (bulk) | Download only the model id(s) precheck marks `✗`, not the whole CI model list. |
| Ad-hoc `uv pip install torch` / wheel URLs | Pins must match CI; precheck reports pin drift. |

### Stage-specific shortcuts (still check-first)

- **Qwen3-ASR (TTS CI stage 1 / `--model tts`)**: uses `omni`, **2 GPU / router DP=2**.
  Stage 1 uses the same ASR transcribe fan-out as other WER stages
  (`QWEN3_ASR_WER_CONCURRENCY` = **32** at DP=2).
  Included in `--model tts --stages ALL`; calibrate alone with
  `--stages qwen3_asr`. Venv only needs to pass precheck for torch/sglang pins
  and cached assets. Do **not** use `--skip-precheck`. Source
  `.github/scripts/ci_env.sh` before pytest/calibration.
- **TTS random-pick CI (`--model tts`)**: CI randomly selects one configured
  TTS model preset per commit, but calibration must **never** randomly select.
  `models/tts/config.yaml` expands every `calibration_preset` into its own
  stages. `--stages tts` / `ALL` therefore runs Higgs and MOSS independently
  and produces per-preset worst-of-N rows.
- **Qwen3-ASR (standalone `--model qwen3-asr-v1`)**: same runtime as above;
  use only for isolated ASR calibration — **TTS PRs should use `--model tts`**.
- **Qwen3 MoE stages**: `flashinfer-python` (cu13) JIT-compiles its MoE/cutlass
  kernels into `${OMNI_CI_HOME}/.cache/flashinfer` on each cold start. A healthy
  cu13 env compiles fast — **router/worker up in < ~60s; > 60s means the JIT path
  is broken**, not a missing wheel or a regression (see the "Server / router
  startup > 60s" signal under Known failure signatures for the env fix). All
  benchmark stages share the single **`omni`** venv — source
  `.github/scripts/ci_env.sh` before every pytest/calibration run.

### When a full venv rebuild is allowed

Only if **all** hold: user explicitly asked, **or** precheck shows venv
missing/corrupt **and** `uv pip install -e .` did not fix import/pin errors,
**and** you warned that `prepare_omni_venv.sh` may delete `$OMNI_CI_HOME`.
Prefer repairing the single reported gap over rebuilding.

## Prerequisites (I verify, I do not create)
- Running inside the CI-reproduction container (image
  `crpi-n6adu6llixz83q37.cn-hangzhou.personal.cr.aliyuncs.com/hongccc/sglang-omni:dev`
  or equivalent cu13 image). The container name is not checked — rely on the
  image being correct.
- **Host profile** loaded (`hosts/*.yaml`): `tune.py` maps `physical.*` into
  `auto_env` (no symlinks). See **AGENT-PRECHECK.md** for the full checklist.
- Venv path from the host profile or selected model's `config.yaml`
  `default_venv_python`; overridable via `--venv-python` or
  `$TUNE_VENV_PYTHON`. **Existence ≠ run `prepare_omni_venv.sh`** — run
  precheck first (see policy above).
- Branch checked out; **`uv pip install -e .` only** to sync sglang-omni onto
  the existing venv unless precheck proves the venv is missing or corrupt.
- Model weights and datasets from the config cached locally. During
  `run`, precheck lists each selected stage's required assets as `✓` /
  `✗`; standalone `precheck` checks all configured assets. On any miss,
  it prints the exact
  `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download …` commands
  to run — run **only those**.
- Env vars under `auto_env` in the model's config.yaml are set
  automatically at tune.py startup. The user does NOT need to `export`
  them. Proxy env vars (`http_proxy` etc.) are left alone — the tests'
  own `disable_proxy()` helper strips them for loopback calls, matching
  real CI.
- `HF_ENDPOINT` defaults to `https://hf-mirror.com` (the China mirror, matches CI
  omni-setup). The **H100 host is overseas**, where `hf-mirror.com` fails with
  `LocalEntryNotFoundError`; warm-cache downloads on that host (speaker-sim WavLM
  and the UTMOS asset — see below) must use `HF_ENDPOINT=https://huggingface.co`.
  Otherwise use `https://huggingface.co` only if a mirror download fails and
  precheck prints an alternate command.
- No GPU processes holding memory at **precheck** time. If all GPUs are
  busy, precheck fails with the busy PID list and the user must free them.
  **During `tune.py run`**, the tool runs `delete_gpu_process.sh` and
  waits until each selected GPU is **≤ 2048 MiB** before every pytest
  invocation and retry — this matches CI's per-stage cleanup, but only
  inside an active calibration run. Precheck itself never kills processes.

If anything's off, `precheck` fails with an actionable message — fix **only
that item**, re-run precheck, then proceed. Never “refresh the whole env”
when precheck already shows `✓` for venv pins and assets.

## Invocation
- `/tune-ci-thresholds` — default model, all stages, 5 repeats
- `/tune-ci-thresholds --model qwen3-omni-v1 --stages mmsu_accuracy --repeats 3`
- `/tune-ci-thresholds --resume <run-dir>` — continue an interrupted run

Common Qwen3-Omni V1 presets:
```
# All Qwen3-Omni threshold stages (every base from stages-list).
python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 run \
  --stages mmmu,mmmu_talker,mmsu,mmsu_talker,tts,videoamme,videoamme_talker,videoamme_talker_tp2,videomme,videomme_talker \
  --repeats 5 --output-dir .tune-runs/<timestamp>_qwen3-omni-v1_r5

# FP8 CI stage 11.
python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 run \
  --stages videoamme_talker_tp2 \
  --repeats 5 --output-dir .tune-runs/<timestamp>_qwen3-omni-v1_fp8_stage_11_r5
```

**FP8 Thinker TP=2 (stage 11) container requirements.** CI runs this stage
(`test_qwen3_omni_videoamme_talker_tp2_ci.py`) with two extra container settings
that the repro host must match, or every request 500s / the allocator
fragments:
- `--cap-add=SYS_PTRACE` — the TP=2 relay passes fds between stage processes
  via `pidfd_getfd`, which needs `CAP_SYS_PTRACE`.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` — GPU 1 co-locates thinker
  rank-1 + talker + code2wav under TP=2; expandable segments reduce allocator
  fragmentation (issue #765). tune.py sets this from `extra_env`, but the
  container must still have been **started** with `--cap-add=SYS_PTRACE`.

Common TTS preset:
```
# Full TTS CI pipeline: Qwen3-ASR + every configured TTS model preset, 5 repeats.
# As of this branch, `tts` expands to Higgs and MOSS; it is not a random pick.
python .claude/skills/tune-ci-thresholds/tune.py --model tts run \
  --stages ALL --repeats 5 --output-dir .tune-runs/<timestamp>_tts_all_r5

# Only model-dependent TTS stages, all configured presets:
python .claude/skills/tune-ci-thresholds/tune.py --model tts run \
  --stages tts --repeats 5 --output-dir .tune-runs/<timestamp>_tts_models_r5

# One preset only, for debugging or a targeted rerun after a failed repeat:
python .claude/skills/tune-ci-thresholds/tune.py --model tts run \
  --stages tts_moss --repeats 5 --output-dir .tune-runs/<timestamp>_tts_moss_r5

# Stage 1 only (Qwen3-ASR on the full SeedTTS EN set):
python .claude/skills/tune-ci-thresholds/tune.py --model tts run \
  --stages qwen3_asr --repeats 5 --output-dir .tune-runs/<timestamp>_tts_qwen3_asr_r5
```

### TTS CI stage 1 — Qwen3-ASR (mandatory in full TTS calibration)

`test-tts-ci.yaml` DAG: **`stage-1-qwen3-asr` is independent**; `stage-2-non-streaming`
and `stage-3-streaming` run in parallel; `stage-4-consistency` `needs`
[stage-2, stage-3]. So `test_qwen3_asr_ci.py` runs in parallel with the Higgs
stages. Full `--model tts --stages ALL` calibration
**must** include the Qwen3-ASR stages — never calibrate Higgs thresholds alone
while leaving Qwen3-ASR on stale literals.

| Stage key | Group | What gets written | Test constant(s) |
|-----------|-------|-------------------|------------------|
| `qwen3_asr_wer` | wer | corpus + per-sample WER ref | `SEEDTTS_ASR_CORPUS_WER_MAX`, `SEEDTTS_ASR_SAMPLE_WER_MAX` |
| `qwen3_asr_speed` | speed | throughput + latency + RTF P95 refs | `QWEN3_ASR_THROUGHPUT_MIN`, `QWEN3_ASR_LATENCY_*`, `QWEN3_ASR_RTF_*` |

Notes:
- Uses **`Qwen/Qwen3-ASR-1.7B`** via `hf_model_ids_by_test` (not the Higgs
  checkpoint). Same **`omni`** venv and 2-GPU router DP=2 as TTS stages.
  Stage 1 imports **`QWEN3_ASR_WER_CONCURRENCY`** from `tests.utils` (32),
  matching TTS WER and talker WER transcribe fan-out.
- Sample count for strict audit: **`SEEDTTS_ASR_CORRECTNESS_SAMPLES`**
  (via `expected_samples` in `stages.yaml`), JSON `summary.evaluated` /
  `summary.total_samples`.
- **CI slack:** tune.py writes P95 reference constants only; assertions use
  derived `*_THRESHOLD` values with **10% slack** (`THRESHOLD_SLACK_HIGHER=0.9`,
  `THRESHOLD_SLACK_LOWER=1.1` via `apply_wer_slack()` for WER). Do **not**
  bake slack into calibrated literals.
- Shortcuts: `qwen3_asr`, `@wer`, `@speed`.
- Standalone model **`qwen3-asr-v1`** remains for isolated ASR runs;
  **TTS pipeline calibration uses `--model tts`** so Qwen3-ASR plus every
  configured TTS model preset share one run directory and provenance.

### TTS random-pick CI vs calibration coverage

CI and calibration intentionally have different sampling policies:

- **CI:** `omni-ci.yaml` runs `pick-tts-model` and chooses one TTS preset
  (`higgs` or `moss` on this branch) for that commit. This keeps per-commit
  cost at one TTS model × two modes.
- **Calibration:** `tune.py --model tts` must run **all** TTS presets declared
  under `models/tts/config.yaml::metric_sources.test_tts_ci.py.calibration_presets`.
  It never calls CI's random picker and never infers one preset's threshold from
  another preset's numbers.
- **Worst-of-N scope:** compute worst-of-5 independently for each
  `(preset, mode, metric-group)` tuple. Do not aggregate Higgs and MOSS into a
  single worst row, and do not let a partial/failed run for one preset count as
  evidence for the other.
- **Threshold surface:** every preset in the CI random-pick set must have the
  same calibrated metric groups and assertion semantics: non-stream speed,
  non-stream WER, non-stream similarity, non-stream UTMOS, stream speed, and
  stream WER. Numeric values may differ per preset. Missing thresholds for a
  scaffold preset are allowed only as a temporary report-only state; do not
  silently reuse Higgs literals for MOSS.
- **GPU topology:** calibration stages can declare per-preset GPU needs.
  Higgs and MOSS currently use 2 GPUs total: the router launches two complete
  single-GPU workers. MOSS Local's default pipeline config is colocated, so its
  codec/vocoder run on each worker's visible `cuda:0`.

The generated stage aliases reflect this:

| Alias | Expands to |
|-------|------------|
| `tts` | all model-dependent TTS stages for every configured preset |
| `tts_higgs` | all Higgs TTS stages |
| `tts_moss` | all MOSS TTS stages |
| `tts_higgs_nonstream` / `tts_moss_stream` | one preset and one mode |
| `@speed`, `@wer`, `@similarity`, `@utmos` | metric group across presets; `@speed` and `@wer` also include Qwen3-ASR when `--model tts` is selected |

### TTS model calibration targets (stages 2–4)

**Fixed sample presets in `test_tts_ci.py` — never apply, never worst-of-N write:**
`SEEDTTS_EN_FULLSET_SAMPLES`, `TTS_SIMILARITY_MAX_SAMPLES`,
`STREAMING_BENCHMARK_MAX_SAMPLES` (when `None`, streaming uses the full
SeedTTS EN set). These define *how many* samples CI runs; tune.py reads the
numeric values via `discover` → `expected_samples`.
Generation concurrency is **16** for both non-streaming and streaming TTS stages;
the Qwen3-ASR WER transcribe phase remains **32**.

**Calibrated thresholds** (worst-of-N → `apply-plan` → `tts_ci_config.py`) use the
**same metric groups** for every TTS preset, but **different Python symbols**
per preset. Never write MOSS worst-of-N into Higgs `HIGGS_VC_*` literals or vice
versa.

| Stage key | Group | Higgs symbol(s) | MOSS symbol(s) |
|-----------|-------|-----------------|----------------|
| `tts_<preset>_nonstream_speed` | speed | `_HIGGS_VC_NON_STREAM_P95[...]` | `_MOSS_VC_NON_STREAM_P95[...]` |
| `tts_<preset>_stream_speed` | speed | `_HIGGS_VC_STREAM_P95[...]` | `_MOSS_VC_STREAM_P95[...]` |
| `tts_<preset>_nonstream_wer` | wer | `HIGGS_VC_WER_MAX_CORPUS` | `MOSS_VC_WER_MAX_CORPUS` |
| `tts_<preset>_stream_wer` | wer | `HIGGS_VC_STREAM_WER_MAX_CORPUS` | `MOSS_VC_STREAM_WER_MAX_CORPUS` |
| `tts_<preset>_nonstream_similarity` | similarity | `HIGGS_VC_SIMILARITY_MEAN_MIN` | `MOSS_VC_SIMILARITY_MEAN_MIN` |
| `tts_<preset>_nonstream_utmos` | utmos | `HIGGS_VC_UTMOS_MEAN_REFERENCE` | `MOSS_VC_UTMOS_MEAN_REFERENCE` |

`stages.yaml` `metrics.*.source` must point at the preset-specific symbol
(e.g. `tts_moss_nonstream_wer` → `MOSS_VC_WER_MAX_CORPUS`). `discover` enforces
this via `calibration_presets.<preset>.constant_filter` in
`models/tts/config.yaml` (`^HIGGS_VC_` for higgs, `^MOSS_VC_` for moss), and
reads/writes threshold literals through
`models/tts/config.yaml::metric_sources.test_tts_ci.py.threshold_file`
(`tests/test_model/tts_ci_config.py`).

Notes:
- **WER** calibrates corpus reference only (`*_WER_MAX_CORPUS`); CI asserts via
  `apply_wer_slack()`. Per-sample WER caps and generation failure budgets are
  not calibrated.
- **Similarity** calibrates `*_SIMILARITY_MEAN_MIN`, not
  `TTS_SIMILARITY_MAX_SAMPLES`.
- **UTMOS** calibrates `*_UTMOS_MEAN_REFERENCE`; CI derives the assertion
  threshold with `apply_mos_slack()`.
- Both presets have `gate_thresholds=True`. Apply worst-of-N **per preset** using
  the symbols above; a MOSS calibration run must not modify any `HIGGS_VC_*` /
  `_HIGGS_VC_*` literal, and a Higgs run must not modify any `MOSS_VC_*` /
  `_MOSS_VC_*` literal.
- When applying from a run dir, use each stage's `calibration_preset` and
  `metrics.*.source` from `apply-plan` — do not infer cross-preset symbols from
  stage titles alone.
- **Stage 4 (consistency)** is a separate CI job that runs
  `tests/test_model/test_tts_consistency_artifacts.py` (not `test_tts_ci.py`),
  comparing the stage-2/stage-3 speed artifacts with `TTS_CONSISTENCY_CONCURRENCY=16`.
  It is pass/fail only — no numeric threshold tune.py calibrates, and it is not
  one of the `test_tts_ci.py` variant stage keys. tune.py's TTS stages cover
  stage 1 (Qwen3-ASR) and the stage-2/3 voice-clone metrics; the consistency job
  is verified by re-running CI, not calibrated.

Shortcuts: `@speed`, `@wer`, `@similarity`, `@utmos`, `ALL`, or `tts` /
`tts_nonstream` / `tts_stream`.

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
| | `/github/home/.cache/uv` | `UV_CACHE_DIR`; PyPI wheels (mirror `https://mirrors.aliyun.com/pypi/simple`) |
| **Per slice (`OMNI_CI_HOME`)** | `<OMNI_CI_HOME>/omni` | Python venv (`uv venv -p 3.11`) |
| | `<OMNI_CI_HOME>/.cache` | `XDG_CACHE_HOME`; uv/torch compile artifacts |
| | `<OMNI_CI_HOME>/.cache/flashinfer` | Runtime FlashInfer JIT dir — `flashinfer-python` (cu13) compiles kernels here on each cold start. **CI wipes it before every pytest attempt** (omni-setup at job start + `run_flaky_pytest.sh` before each retry), and `tune.py` mirrors that per launch. A healthy cu13 compile is fast (startup < ~60s); a > 60s startup means the JIT path is broken, not slow. |
| | `<OMNI_CI_HOME>/.torchinductor` | `TORCHINDUCTOR_CACHE_DIR` |

- **CI Actions runners** use `OMNI_CI_HOME=/github/home/pr-<N>`.
- **Calibration host** uses `omni_ci_home: /github/home/calibration` from each model's `config.yaml`.
- `tune.py` / `runner.py` apply `auto_env` from `config.yaml` and **override** shell env to match CI.

### Prepare a calibration venv (only when precheck proves venv missing or corrupt)

**Do not run this block at the start of a normal calibration.** Use it only
after precheck reports the venv path missing, imports fail, or sglang/torch
pins cannot be fixed with `uv pip install -e .`.

From repo root on the H100 repro host (cu13 `hongccc/sglang-omni:dev` semantics):

```bash
# Qwen3-Omni example (TTS: OMNI_CI_HOME=/github/home/calibration, venv omni)
export OMNI_CI_HOME=/github/home/calibration
export HOME=/github/home
export HF_HOME=/github/home/.cache/huggingface
export MODELSCOPE_CACHE=/github/home/.cache/modelscope
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export UV_CACHE_DIR=/github/home/.cache/uv
export XDG_CACHE_HOME=${OMNI_CI_HOME}/.cache
export TORCHINDUCTOR_CACHE_DIR=${OMNI_CI_HOME}/.torchinductor

bash .github/scripts/prepare_omni_venv.sh omni
ln -sfn "${OMNI_CI_HOME}/omni" ./omni
bash .github/scripts/ensure_hf_models.sh omni \
  Qwen/Qwen3-Omni-30B-A3B-Instruct marksverdhei/Qwen3-Omni-30B-A3B-FP8
```

`prepare_omni_venv.sh` now keys on a `pyproject.toml` deps-hash: when the hash
is unchanged and the venv imports cleanly it **reuses** the slice and only runs
`uv pip install --upgrade -e .` (no wipe). It only does the fresh path
(`rm -rf $OMNI_CI_HOME`, `uv venv -p 3.11`, full reinstall) when deps changed or
the venv is missing/corrupt. Normal day-to-day calibration (venv exists, pins
ok) still needs **only** `source <venv>/bin/activate && uv pip install -e .` —
same as CI re-checkout on a new commit. Call `prepare_omni_venv.sh` only when
precheck proves the venv path is missing or corrupt.

### Required env vars (auto-set from `config.yaml`)

- `HOME=/github/home`
- `OMNI_CI_HOME`, `XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR` — per-slice paths above
- `HF_HOME`, `MODELSCOPE_CACHE`, `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`
- `UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple`, `UV_CACHE_DIR=/github/home/.cache/uv`
- `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `CUDA_VISIBLE_DEVICES` — `ci_env.sh` defaults to `0,1`; during `tune.py run` the tool overrides it per stage to the free GPUs it picks

For missing model/dataset assets only, run the precheck-printed download
command (often with `HF_ENDPOINT=https://hf-mirror.com`).

If HF cache lives on a non-default path, add a host profile with
`physical.hf_hub` — do not rely on symlinks; `tune.py` sets `HF_HOME` directly.

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
source /github/home/calibration/omni/bin/activate
export GITHUB_ACTIONS=true RUNNER_TEMP=/tmp PYTHONPATH=$PWD
export HOME=/github/home OMNI_CI_HOME=/github/home/calibration
export HF_HOME=/github/home/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export SEEDTTS_SIM_CACHE_DIR=/github/home/seedtts-wavlm-sim
export XDG_CACHE_HOME=${OMNI_CI_HOME}/.cache
export TORCHINDUCTOR_CACHE_DIR=${OMNI_CI_HOME}/.torchinductor
export UV_CACHE_DIR=/github/home/.cache/uv FLASHINFER_DISABLE_VERSION_CHECK=1
export NO_PROXY=localhost,127.0.0.1,::1
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x
```

### Known failure signatures

- **Slow safetensors load / disk sleep**: IO pressure — check competing processes, not thresholds.
- **🔑 Server / router startup > 60s ⇒ the FlashInfer JIT path is broken (not a
  threshold/code/GPU problem).** This is the single most reliable startup signal.
  A correctly-aligned cu13 env compiles FlashInfer's kernels into
  `${OMNI_CI_HOME}/.cache/flashinfer` **fast** — healthy colocated-router + worker
  startup is **under ~60s**, even though the cache starts cold every time (see
  next bullet). If `/health` / router readiness has not gone green after ~60s
  (you'll see `gen_cutlass_fused_moe_sm90_module` / long `nvcc`/`ninja` compile
  lines stuck or looping in the pytest log), the FlashInfer JIT is **failing to
  compile cleanly** — almost always because of a polluted environment: stale JIT
  artifacts from a different flashinfer/cu/GPU build, `HOME` not `/github/home`,
  `XDG_CACHE_HOME` not under `${OMNI_CI_HOME}`, or `nvcc`/`ninja` pointing at the
  wrong Python home. **Fix the env, never the thresholds:** `rm -rf
  ${OMNI_CI_HOME}/.cache/flashinfer`, re-`source .github/scripts/ci_env.sh`,
  confirm `HOME=/github/home`, then relaunch. The harness will tolerate a slow
  startup (`STARTUP_TIMEOUT=600`, router `QWEN3_OMNI_ROUTER_WAIT_TIMEOUT=180`,
  talker `timeout_s=500`) but a >60s startup is your cue to stop and fix the env.
- **Cold cache every attempt is intended — it matches CI.** CI wipes
  `${OMNI_CI_HOME}/.cache/flashinfer` **before every pytest attempt**: `omni-setup`
  wipes once at job start, and `run_flaky_pytest.sh` wipes again before each of its
  ≤3 retries. So every CI attempt starts cold and recompiles, and CI still comes up
  in time — which is exactly why a healthy compile is fast. `tune.py` mirrors this:
  `_cleanup_flashinfer_cache()` runs before **every** pytest launch (each repeat and
  each retry). **Do not "warm" or preserve the cache to speed up calibration** — that
  would diverge from CI. Calibration is "run CI 5×", so each repeat must start cold
  like a fresh CI job.
- **GPU cleanup / `[Not Found]` PIDs**: kill container-visible pytest/server processes:
  `pgrep -af "multiprocessing.spawn|sglang_omni_router|sgl-omni serve|pytest|nvcc|ninja"`
- **Between sequential pytest stages on the repro host**: `delete_gpu_process.sh`
  alone is **not enough** — orphan `multiprocessing.spawn` children can hold
  ~70–85 GiB while `nvidia-smi` shows "No running processes". Always run
  `.github/scripts/delete_gpu_process.sh --kill-orphans` (kills orphans + scans `/proc/*/fd`
  for nvidia + waits until **every** GPU `< 2048 MiB`) **before and after**
  each heavy benchmark. Do **not** start the next pytest until cleanup succeeds.
- **Starting pytest while the previous server is still tearing down** causes
  colocated-router OOM on the second worker — wait for `delete_gpu_process`, then
  `sleep 3–5` before launch.

After alignment fixes, rerun `tune.py precheck` and the smoke test before resuming calibration.

### Critical: single `omni` venv everywhere

All CI workflows, calibration models, and WER sweeps use the same venv name
(`omni`) and the same env script (`.github/scripts/ci_env.sh`). Only
`OMNI_CI_HOME` differs by host slice: `/github/home/pr-<N>` on Actions,
`/github/home/calibration` on the repro host.

| Workload | CI workflow | venv | `OMNI_CI_HOME` (calibration host) | Source env script |
|----------|-------------|------|-----------------------------------|-------------------|
| All benchmarks (unit, Qwen3, TTS, Qwen3-ASR) | `omni-ci.yaml`: `preflight → setup → pr-test (test.yaml) → tts-ci (test-tts-ci.yaml) → qwen3-omni-ci (test-qwen3-omni-ci.yaml) → cleanup` | **`omni`** | `/github/home/calibration` | `source .github/scripts/ci_env.sh` |

**Omni CI suite order (DAG):** `preflight → setup → pr-test → tts-ci →
qwen3-omni-ci → cleanup`. After the gated `preflight` and the shared `setup`
job, **unit / non-benchmark tests (`test.yaml`, "PR Test") run FIRST**, then
`test-tts-ci.yaml`, then `test-qwen3-omni-ci.yaml`. Each benchmark suite `needs`
the previous (`tts-ci needs [setup, pr-test]`, `qwen3-omni-ci needs [setup,
tts-ci]`) but is `if: always() && !cancelled() && setup == success`, so a failure
in PR Test or TTS does **not** skip the later suites. Only a failed `setup` (or
`preflight` gate) blocks the chain.

**Forbidden shortcuts (observed 2026-05-30):**

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Tab A tails `<run-dir>/run.log` during `tune.py run` | Tab A and Tab B show **identical** milestone lines; no pytest/router output | Kill Tab A; run `bash .claude/skills/tune-ci-thresholds/tail_calibration_pytest.sh <run-dir>` |
| Tab A uses `tail -f $(ls -t …/run*.log \| head -1)` | Tab A **freezes** on first attached log (e.g. mmmu FAILED) while calibration continues | **Never** manual `ls -t` tail; use `tail_calibration_pytest.sh` only |
| Ignoring metric extraction warnings and continuing calibration | Invalid `run{k}.json` on disk; strict audit ✗; worst-of-N unusable | **HALT** on extraction failure; fix config; purge; `--resume` |
| Reporting progress from `ok/total` only | User thinks calibration is done while strict audit shows ✗/△ | Always show strict audit `N/N ✓` per stage |
| Tab A helper matches wrong process or absolute `RUN_DIR` only | Tab A shows **waiting for pytest** or stale log while calibration runs | Fixed in script: `pgrep` must match `python -m pytest` (not supervisor bash); resolve log via `RUN_DIR_BASENAME` + `_pytest/<test>/runK` from relative `--basetemp` |
| Wrong or unset `OMNI_CI_HOME` | Router worker unhealthy, stale torchinductor cache, HF cache miss | `source .github/scripts/ci_env.sh` |
| `TORCHINDUCTOR_CACHE_DIR=/.torchinductor` or unset (inherits garbage) | **Every** server start re-captures CUDA graphs (~minutes); log shows long `Capturing batches` | Set via `ci_env.sh` → `${OMNI_CI_HOME}/.torchinductor` |
| `HOME=/root` or datasets under `/root/.cache/huggingface` | HF cache miss, re-download, wrong normalizer paths | `HOME=/github/home`, `HF_HOME=/github/home/.cache/huggingface` |
| Killing calibration mid-run without cleaning orphans | `nvidia-smi` shows ~70–85 GiB used but “No running processes” | `pgrep -af multiprocessing.spawn` then `kill -9`; run `delete_gpu_process.sh` |

**Before any pytest / calibration / WER sweep**, always:

```bash
cd /sgl-workspace/sglang-omni
source omni/bin/activate
source .github/scripts/ci_env.sh
python -c "import os; assert os.environ['TORCHINDUCTOR_CACHE_DIR'].startswith(os.environ['OMNI_CI_HOME'])"
python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 precheck   # or qwen3-asr-v1 / tts
```

Aligned env → Qwen3 colocated router CUDA graph capture ~5–10 s on warm
`${OMNI_CI_HOME}/.torchinductor`. Cold or wrong slice → multi-minute startup;
do **not** treat that as a threshold or code regression.

**WER CI with Qwen3-ASR router (DP=2):** still uses the **parent model’s**
venv/slice for the benchmark fixture (Qwen3 → qwen3 env; TTS → tts env; standalone ASR → tts env).
Qwen3-Omni/TTS generation concurrency is **16** where the test has a
`CONCURRENCY` knob; Qwen3-ASR WER router/transcribe fan-out is **32**.
Only the Qwen3-ASR router stage needs 2 free GPUs after `delete_gpu_process.sh`.

### Agent operational rules (mandatory)

- **Two-terminal supervision** — follow **Two-terminal supervision (mandatory —
  always)** at the top of this skill. Agent creates Tab A
  (`tail_calibration_pytest.sh <run-dir>`) then Tab B (job). Tab A = pytest log;
  Tab B = milestones only. **Never `tail -f <run-dir>/run.log` on Tab A** during
  calibration — it duplicates Tab B. **Never `tee` on Tab B.**
- **Never block on a single shell/tool wait longer than 2 minutes.** Agent
  polls with short checks (`tail -20`, `grep PASS/FAIL`, `nvidia-smi`,
  `tune.py status`) at ≤120s intervals. User supervision is **`tail -f`**, not
  agent long-wait.
- **Always `source ci_env_*.sh` in the same shell** that launches pytest or
  `tune.py run` — background wrappers must source it inside the script, not
  rely on a polluted parent shell (`TORCHINDUCTOR_CACHE_DIR=/.torchinductor`
  has been observed from stale exports).
- **One GPU consumer at a time** on the repro host: do not overlap `tune.py
  run` with full talker/WER pytest — they fight for the same 2× H100.
- **GPU idle gate before every stage** — run `.github/scripts/delete_gpu_process.sh --kill-orphans`
  (not `delete_gpu_process.sh` alone); abort if VRAM not below 2048 MiB on
  **both** GPUs before starting the next pytest.

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

### Agent polling — never blind-wait (P0)

- **Maximum idle poll interval: 120 seconds (2 minutes).** This is
  non-negotiable. Never use `block_until_ms` ≥ 120000 (or multi-minute
  `Await`) while a calibration run is active. Long blind waits hide server
  crashes and incomplete strict audits.
- While `tune.py run` is in progress, **every 120s at most** (prefer 60–90s
  during active GPU work):
  1. Run `python tune.py status --run-dir <run-dir>` and read JSON
     (`strict_ready`, `strict_complete`).
  2. Run `python tune.py strict-audit --run-dir <run-dir>` — report this
     output to the user, not `ok/total` alone.
  3. For agent polling only (not Tab A): skim `<run-dir>/run.log` milestones if
     needed; read the active `<run-dir>/_pytest/<test>/run{k}.log` (last ~30 lines)
     for pytest/server detail. **Tab A** must tail `_pytest` via
     `tail_calibration_pytest.sh`, never `run.log`.
  4. Check GPU memory ≤ 2048 MiB before next stage launches.
  5. If strict audit shows **any** stage with `< N` ✓, wrong `expected_samples`,
     or `run.log` has `sample scope mismatch` / metric extraction warnings →
     **stop treating calibration as healthy** — fix, purge, `--resume`.
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
- **Extraction HALT (v0.4.2+):** if pytest exits 0 but **any** fed stage
  has incomplete metrics (config / missing JSON), `tune.py run` **stops
  immediately** with exit code 1. Agent must fix `metric_sources` or test
  JSON output, purge the bad repeat, then `--resume`. Do not start the
  next test file until HALT is resolved.
- **Per-run infra retries:** within a single repeat, a stage-run that fails on
  OOM / crash / GPU-not-clear is retried up to 4 times to obtain one clean
  observation, before marking that repeat incomplete. A threshold-assertion
  failure is **never** retried — it is a valid worst-of-N observation. This is a
  calibration-specific mechanism for getting clean data; it is **not** CI's
  per-test failure retry (CI's logic — "rerun a failing unit test up to 3 times
  to make it pass" — is unrelated to and must not be conflated with worst-of-N
  calibration).
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

**Before step 0:** read and execute
`.claude/skills/tune-ci-thresholds/AGENT-PRECHECK.md` (full environment +
weights checklist for agents).

0. **Host profile.** `tune.py` autodetects `hosts/*.yaml` by `hostname`
   (or `--host` / `$TUNE_HOST`). It applies physical paths to `auto_env` —
   **no symlink setup**. Run `precheck` (includes speaker sim when
   applicable). Report env gaps before fixing unless user asked.
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
     tts                          tests/test_model/test_qwen3_omni_tts_ci.py — wer + utmos + speed
   Shortcuts: @accuracy, @speed, @wer, @utmos (metric-group aliases).
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
   On failure, relay the message verbatim; fix **only** the reported gap(s)
   per **Environment policy — check first** (typically `uv pip install -e .`
   and/or one `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download …`),
   re-run precheck until `✓`, then
   continue. Do **not** run `prepare_omni_venv.sh` or bulk downloads when
   precheck already passes.
   **`<run-dir>` must live under `.tune-runs/<timestamp>_<label>/`** at
   the repo root (e.g. `.tune-runs/20260423T050000Z_mmsu_r3/`). Generate
   the timestamp with **`date -u +%Y%m%dT%H%M%SZ` at session start** — never
   reuse a prior run dir unless the user explicitly asked to `--resume` it.
   That path is already gitignored. Do NOT point `<run-dir>` inside
   `.claude/skills/` or anywhere else under version control — run
   artifacts can be large and must not leak into commits.
4. State plan in one line:
   `Running <M>: <stages>, <N> repeats on commit <full-sha>, run-dir <path>.`
   No further confirmation.
5. **Before** launching run, **spawn two IDE terminal tabs** per **Two-terminal
   supervision (mandatory — always)** — Tab A helper first, Tab B job second:

   **Tab A — supervision (pytest + server log; NOT run.log):**
   ```bash
   bash .claude/skills/tune-ci-thresholds/tail_calibration_pytest.sh <run-dir>
   ```

   **Tab B — job (tune.py milestones on stdout — no redirect):**
   ```bash
   cd <repo_root from host profile> && python .claude/skills/tune-ci-thresholds/tune.py --model <M> run ... \
     --output-dir <run-dir>
   ```
   Example (`sglang-h100-ci`): `cd /data/sglang-omni && TUNE_VENV_PYTHON=/github/home/calibration/omni/bin/python python .claude/skills/tune-ci-thresholds/tune.py ...`

   Tell the user: **Tab A = pytest/server**, **Tab B = tune progress** — if both
   tabs show the same lines, Tab A is wrong (likely tailing `run.log`). Never
   `>> run.log` on Tab B (tune.py tees internally).

   Agent polls every **≤120s**: `python tune.py status --run-dir <run-dir>`.

6. When `tune.py run` exits 0, verify **all three** gates:
   - `python tune.py status --run-dir <run-dir>` → `"complete": true`
   - Strict audit → every stage **N/N ✓** (see "Strict worst-of-N")
   - Strict audit → **`GIT PROVENANCE: ok`** (every `run{k}.json` matches
     `calibration_git_sha`)
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
      max_tokens=32, concurrency=<CONCURRENCY from test>, 5 runs`.
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
     - `smart` — auto-apply accuracy, WER, similarity, and UTMOS worst-of-N;
       auto-tighten speed thresholds; ask only for speed metrics that would loosen
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
     - **`accuracy` / `similarity` / `utmos`:** `write_value` = `worst_raw`
       exactly into `*_MIN_ACCURACY`, `*_SIMILARITY_*_MIN`, or
       `*_UTMOS_*_REFERENCE`. Report percentages use 2 decimal places for
       readability only; similarity and UTMOS use raw scores (not %).
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
     - **auto-apply** iff `stage_group` in (`accuracy`, `wer`, `similarity`, `utmos`),
       OR (`stage_group == "speed"` AND `direction == "tightens"`).
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
     - **TTS preset isolation:** each stage's `calibration_preset` and
       `metrics.*.source` / `symbol` from apply-plan identify the exact
       literal to edit (`HIGGS_VC_*` for higgs, `MOSS_VC_*` for moss). Never
       substitute one preset's symbol for another, even when metric groups
       match.
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
- Proceed to the next test, model, report, or apply while **any** stage
  lacks N/N strict ✓ repeats with **every** tracked metric non-null **and**
  correct `expected_samples` scope.
- Treat `tune.py status` `ok/total` or `complete: true` as strict
  worst-of-N readiness — always run `tune.py strict-audit` (✓ = full samples
  at CI scope per `expected_samples` in `stages.yaml`).
- Blind-wait more than **120 seconds** without `status` + `strict-audit` during
  an active `tune.py run`.
- Continue calibration after `run.log` shows metric extraction warnings
  or `tune.py` extraction **HALT** — fix config, purge, `--resume` first.
- Include △ partial or ✗ failed repeats in worst-of-N calculations or
  apply decisions.
- Download, rebuild, or bulk-install the calibration environment before
  `precheck` proves a specific gap. No proactive `prepare_omni_venv.sh` or
  `ensure_hf_models.sh`.
- Run `prepare_omni_venv.sh` when precheck already shows a working venv (its
  fresh path runs `rm -rf $OMNI_CI_HOME`; it only takes that path when the
  `pyproject.toml` deps-hash changed or the venv is missing/corrupt).
- Check out branches unless the user asked or calibration requires it.
  Sync code with `uv pip install -e .` only unless precheck proves the
  venv is missing/corrupt — then follow **Environment policy — check first**.
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
├── AGENT-PRECHECK.md                  # agent runbook: env + weights before run
├── tail_calibration_pytest.sh         # Tab A helper for tune.py run (_pytest log)
├── tune.py                              # CLI; METRIC_SPECS + JSON extractor
│                                        # subcommands: run, report, status,
│                                        # strict-audit, apply-plan, precheck, discover
├── hosts/                               # per-machine repo/venv/cache layouts
│   └── sglang-h100-ci.yaml              # in-container repo/venv/cache layout for the H100 CI host
└── models/
    ├── qwen3-omni-v1/                   # v1 pipeline (qwen3-omni)
    │   ├── config.yaml
    │   └── stages.yaml
    ├── tts/                             # TTS CI pipeline (stage 1 Qwen3-ASR + Higgs/MOSS)
    │   ├── config.yaml                  #   per-preset constant_filter for discover/apply
    │   └── stages.yaml
    └── qwen3-asr-v1/                    # Isolated Qwen3-ASR only
        ├── config.yaml
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
- `sample_counts_by_group` — optional per-stage-group override for
  `sample_counts` when speed/WER/UTMOS artifacts live in different files
- `ignored_constants` — optional list of top-level constants to skip during
  discovery when a test keeps an old or disabled threshold literal around
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
with shared generation/WER artifacts, auto-inference is not available —
config.yaml entries are required.

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
- New naming pattern (e.g. `*_BLEU_MIN`, `TTS_MAX_FAILED_REQUESTS`,
  `*_SIMILARITY_*_MIN`) → extend `match_metric()` and `METRIC_SPECS` in
  `tune.py`.
- Metric lives in a different JSON than the test's default → use the
  `<file>::<dotted.path>` inline form in `metric_sources.<test>.paths`.

Threshold constants whose name `match_metric()` doesn't recognize are
silently ignored — extend `match_metric()` if you add a new pattern.
