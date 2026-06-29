# Agent checklist — environment gate before calibration

**Audience: AI agent only.** Run this checklist at the start of every calibration
session (including after a fresh container). **Do not** run `tune.py run` until
every mandatory item passes.

**Assumption:** repro container is already running. Do not document or execute
`docker run`, volume maps, or host-side setup here.

**Policy** (`hosts/*/yaml` → `agent_policy`):

- `env_check: report_missing_first` — report gaps to the user before fixing;
  fix only what they explicitly asked for, or trivial in-container creates
  (e.g. empty cache subdirs) that precheck requires.
- Do not run `prepare_omni_venv.sh` or bulk model downloads when precheck is
  green or shows a single missing repo.

Path source: `hosts/<name>.yaml` (not CI doc paths in `models/*/config.yaml`).
No symlinks — `precheck` `auto env:` lines must match `physical.*` in the
host profile.

---

## Gate 0 — Calibration scope

Read `handoff:` in the active host profile and confirm with the user if unclear.

| Scope | `--model` | Stages | Repeats | Order |
|-------|-----------|--------|---------|-------|
| TTS CI | `tts` | `ALL` | 5 | Runs Qwen3-ASR plus every configured TTS `calibration_preset`; do not use CI random pick |
| Qwen3-Omni CI | `qwen3-omni-v1` | `ALL` | 5 | — |
| Full CI | both | each `ALL` | 5 each | **TTS first**, then Qwen3-Omni |

Run precheck for **every** model you will calibrate before `tune.py run`.
For `--model tts`, the `tts` alias expands to every TTS model preset declared
in `models/tts/config.yaml` (currently Higgs and MOSS). Calibration must produce
worst-of-5 for each preset independently even though CI samples one preset per
commit.

**Threshold symbols (do not cross-apply):**

| Preset | WER (non-stream / stream) | Similarity | UTMOS | Speed P95 dict |
|--------|---------------------------|------------|-------|----------------|
| `higgs` | `HIGGS_VC_WER_MAX_CORPUS` / `HIGGS_VC_STREAM_WER_MAX_CORPUS` | `HIGGS_VC_SIMILARITY_MEAN_MIN` | `HIGGS_VC_UTMOS_MEAN_REFERENCE` | `_HIGGS_VC_NON_STREAM_P95` / `_HIGGS_VC_STREAM_P95` |
| `moss` | `MOSS_VC_WER_MAX_CORPUS` / `MOSS_VC_STREAM_WER_MAX_CORPUS` | `MOSS_VC_SIMILARITY_MEAN_MIN` | `MOSS_VC_UTMOS_MEAN_REFERENCE` | `_MOSS_VC_NON_STREAM_P95` / `_MOSS_VC_STREAM_P95` |

After changing threshold literals in `tests/test_model/tts_ci_config.py`, run
`tune.py --model tts discover` so `stages.yaml` sources stay aligned with
`calibration_presets.*.constant_filter`.

**After any `discover`, verify `expected_samples` for full-dataset stages.**
For a stage whose only `context_var` is `CONCURRENCY` (no `MAX_SAMPLES`) — e.g.
`mmmu_accuracy`/`mmmu_speed`, `mmsu_accuracy`/`mmsu_speed` — `discover` wrongly
sets `expected_samples = CONCURRENCY` (16), so the completion gate never passes
and the stage triggers futile `--resume` retries (mmsu is the 2000-sample slow
stage). Confirm `expected_samples` matches the real dataset size (mmmu = 50,
mmsu = 2000) and fix the `stages.yaml` literals **before** `run`. See SKILL.md
**Mandatory re-run on any gap**.

---

## Gate 0a — Fresh session vs resume (P0)

**Before every `tune.py run`**, decide:

| Situation | Action |
|-----------|--------|
| User asked for **new** calibration (default) | `RUN=.tune-runs/$(date -u +%Y%m%dT%H%M%SZ)_<label>` — **no** `--resume` |
| User said **continue / resume** `<run-dir>` | `--resume --output-dir <run-dir>` only |
| User moved to a newer commit since last run | **New** run dir on current `HEAD`; do not `--resume` old dir |

**Pass:**

```bash
git rev-parse HEAD
# announce: "Calibrating commit <full-sha> → run dir <path>"
```

**Fail → stop:**

- Reusing an existing `plan.json` run dir without `--resume` (`tune.py` errors).
- `--resume` when `HEAD` ≠ `plan.json` `calibration_git_sha`.
- Opening an old `report.md` instead of calibrating current `HEAD`.

See SKILL.md **Fresh calibration session**.

---

## Gate 1 — Host profile loaded

```bash
python .claude/skills/tune-ci-thresholds/tune.py hosts-list
hostname
# then cd to repo_root from host profile (sglang-h100-ci: /data/sglang-omni)
cd /data/sglang-omni
```

**Pass:**

- Active profile resolves via `$TUNE_HOST` → `--host <name>` → `hostname` match.
- First line of any `tune.py` subcommand prints
  `host: <name> (repo=<repo_root>)`.

**Fail → report:**

| Observation | Action |
|-------------|--------|
| No `host: …` line | Set `--host sglang-h100-ci` or `export TUNE_HOST=sglang-h100-ci`; if hostname wrong, report mismatch |
| `repo_root` missing / no `pyproject.toml` | Report; do not calibrate |

Reference profile `sglang-h100-ci` (`hosts/sglang-h100-ci.yaml`, current/active):

| Key | Expected path |
|-----|----------------|
| `repo_root` | `/data/sglang-omni` |
| `venv_python` | `/github/home/calibration/omni/bin/python` |
| `physical.hf_hub` | `/root/.cache/huggingface` |
| `physical.speaker_sim` | `/root/.cache/huggingface/speaker_sim` |
| `physical.omni_ci_home` | `/github/home/calibration` |

If the user gave different paths in chat, use those and report that host YAML
may be stale.

---

## Gate 2 — Repo, venv, dependency pins

Set from host profile (example uses `sglang-h100-ci`):

```bash
HOST_ROOT=/data/sglang-omni
VENV=/github/home/calibration/omni/bin/python

test -f "$HOST_ROOT/pyproject.toml" && echo PASS repo || echo FAIL repo
test -x "$VENV" && echo PASS venv || echo FAIL venv
$VENV -c "import torch, sglang, flashinfer, sglang_omni; \
  print('torch', torch.__version__); print('sglang', sglang.__version__); \
  print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

**Pass:**

- Repo and venv exist.
- Imports succeed.
- **torch 2.11.0**, **sglang 0.5.12.post1** (precheck re-validates pins).

**Fail → report:**

| Observation | Action |
|-------------|--------|
| FAIL venv | Report path; stop — do not run `prepare_omni_venv.sh` unless user asked |
| Import / pin error | Try `cd "$HOST_ROOT" && uv pip install -e .` once; re-check; if still fail, report |
| `cuda False` or GPU count ≠ 2 | Report |

---

## Gate 3 — OMNI slice directories

Required for FlashInfer / torchinductor during pytest:

```bash
OMNI=/github/home/calibration   # or physical.omni_ci_home from host profile
mkdir -p "$OMNI/.cache" "$OMNI/.torchinductor"
test -d "$OMNI/.cache" && test -d "$OMNI/.torchinductor" && echo PASS omni_slice
```

**Pass:** both subdirs exist.

**Fail → report** if creation fails (permissions / read-only root).

Optional verify (matches CI env wiring):

```bash
cd "$HOST_ROOT"
source .github/scripts/ci_env.sh
$VENV -c "import os; assert os.environ['TORCHINDUCTOR_CACHE_DIR'].startswith(os.environ['OMNI_CI_HOME'])"
```

If `ci_env.sh` sets `HF_HOME=/github/home/.cache/huggingface`, confirm that
path also contains hub snapshots **or** rely on `tune.py` host profile
(`HF_HOME=/root/.cache/huggingface`) — precheck `auto env:` is authoritative
for calibration.

---

## Gate 4 — GPUs idle

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

**Pass:** 2× H100 (or profile-equivalent), each **≤ 2048 MiB** used before
calibration runs (`tune.py` re-checks at run time).

**Fail → report** GPU busy; do not start `tune.py run`. Precheck does not kill
processes.

---

## Gate 5 — HuggingFace weights and datasets

Authoritative check: **`tune.py precheck`** (Gate 8). Quick sanity listing:

```bash
HF=/root/.cache/huggingface   # or physical.hf_hub from host profile
ls "$HF/hub" 2>/dev/null | rg -i 'qwen3|higgs|seed-tts|video|mmmu|mmsu|marksverdhei' | head -20
```

Expected repos by model (precheck validates each):

**`tts`:** models `boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999`,
`Qwen/Qwen3-ASR-1.7B`; dataset `zhaochenyang20/seed-tts-eval-arrow`.

**`qwen3-omni-v1` (adds):** models `Qwen/Qwen3-Omni-30B-A3B-Instruct`,
`marksverdhei/Qwen3-Omni-30B-A3B-FP8`; datasets `zhaochenyang20/mmsu-ci-2000`,
`zhaochenyang20/mmmu-ci-50`, `zhaochenyang20/seed-tts-eval-50-arrow`,
`zhaochenyang20/Video_MME_ci`, `zhaochenyang20/Video_AMME_ci`.

**Fail → report** missing repos. If precheck prints ✗ with
`huggingface-cli download …`, run **only** those lines (one repo at a time,
`HF_ENDPOINT=https://hf-mirror.com`). For private repos, verify `HF_TOKEN`
(`source ~/.zshrc` or env) before download; if token missing, report first.

---

## Gate 6 — Speaker similarity assets

Directory: `physical.speaker_sim` (default `/root/.cache/huggingface/speaker_sim`).

```bash
SIM=/root/.cache/huggingface/speaker_sim
VENV=/github/home/calibration/omni/bin/python

for f in wavlm_large.pt wavlm_large_finetune.pth .complete; do
  test -f "$SIM/$f" && echo PASS "$f" || echo FAIL "$f"
done

# hf-mirror.com is the China default, but the H100 host is overseas and fails it
# with LocalEntryNotFoundError — use huggingface.co for the warm-cache download.
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export SEEDTTS_SIM_CACHE_DIR="$SIM"
cd /data/sglang-omni
$VENV -m benchmarks.metrics.speaker_similarity_assets --warm-cache
# Must print: cache HIT at .../speaker_sim
```

**Pass:** three files present (each `.pt`/`.pth` ≥ 100 MB); warm-cache **HIT**.

**Fail → report**; if user asked to fix, use `speaker_similarity_bootstrap` in
host profile. Do not re-download when `.complete` exists and warm-cache HITs.

**Also warm the UTMOS asset (NOT checked by precheck).** The TTS `tts_utmos`
metric downloads `balacoon/utmos` → `utmos.jit` on demand via
`benchmarks.metrics.utmos.ensure_utmos_assets`, into
`/github/home/.cache/sglang-omni/utmos`. precheck does **not** verify it, so on an
overseas host (e.g. H100) `tts_utmos` fails **mid-run** because `hf-mirror.com`
can't serve the file. Warm it before TTS calibration with the same overseas
endpoint (`ensure_utmos_assets()` — a raw `huggingface-cli download` won't satisfy
its `.utmos_cache.json` marker):

```bash
HF_ENDPOINT=https://huggingface.co $VENV -c \
  "from benchmarks.metrics.utmos import ensure_utmos_assets; ensure_utmos_assets()"
```

---

## Gate 7 — `CAP_SYS_PTRACE` (Full CI / Qwen3 stage 11 only)

Skip for TTS-only calibration.

```bash
# /proc/self/status lists capabilities as hex bitmasks — never grep for
# the string "cap_sys_ptrace" there (always false). Use capsh instead:
capsh --print 2>/dev/null | rg -qi 'cap_sys_ptrace' && echo PASS ptrace || echo FAIL ptrace
```

**Pass:** `PASS ptrace` (output includes `cap_sys_ptrace=ep` or `cap_sys_ptrace` in Current).

**Fail → report**; stage `videoamme_talker_tp2` will fail. Calibrate other
stages only if user accepts partial scope.

---

## Gate 8 — Official precheck (mandatory)

Run for **each** model in Gate 0 scope:

```bash
cd /data/sglang-omni
export TUNE_HOST=sglang-h100-ci   # if Gate 1 autodetect failed

python .claude/skills/tune-ci-thresholds/tune.py --model tts precheck \
  --output-dir /tmp/precheck_tts

python .claude/skills/tune-ci-thresholds/tune.py --model qwen3-omni-v1 precheck \
  --output-dir /tmp/precheck_qwen3
```

Add `--host sglang-h100-ci` if autodetect failed in Gate 1.

**Pass — every line good, ends with `precheck OK`:**

```
host: sglang-h100-ci (repo=...)
venv_python: ... [ok]
  sglang: 0.5.12.post1 (pin ...) [ok]
  torch: 2.11.0+cu130 (pin ...) [ok]
  auto env: HF_HOME=/root/.cache/huggingface
  auto env: SEEDTTS_SIM_CACHE_DIR=/root/.cache/huggingface/speaker_sim
    ✓ model: ...
    ✓ dataset: ...
    ✓ speaker_sim: ... (wavlm_large.pt + wavlm_large_finetune.pth)
  GPUs: 2× NVIDIA H100 80GB HBM3 — 2/2 free

precheck OK
```

`auto env: HF_HOME` **must** match `physical.hf_hub` in host profile.

**Common misreads:**

| Symptom | Likely cause | Agent action |
|---------|--------------|--------------|
| Gate 7 FAIL ptrace but Docker has `--cap-add=SYS_PTRACE` | Used `grep cap_sys_ptrace /proc/self/status` (hex bitmasks only — always false) | Re-check with `capsh --print \| rg sys_ptrace` |
| HF ✗ but files under `/root/.cache/huggingface` | Host profile not loaded | `--host` / `$TUNE_HOST` |
| Wrong `HF_HOME` in `auto env` | Same | Same |
| speaker_sim ✗ | Gate 6 incomplete | Fix per Gate 6 or report |
| GPU busy | Gate 4 | Report |

Do **not** run `prepare_omni_venv.sh` or bulk `ensure_hf_models.sh` when precheck
is green or only reports one missing repo.

---

## Gate 9 — Optional smoke (after precheck OK)

Only if time permits or user requested; not a substitute for Gate 8.

```bash
cd /data/sglang-omni
source /github/home/calibration/omni/bin/activate
source .github/scripts/ci_env.sh
export GITHUB_ACTIONS=true RUNNER_TEMP=/tmp PYTHONPATH=$PWD
export NO_PROXY=localhost,127.0.0.1,::1
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x
```

Router/worker cold start **< ~60s** after FlashInfer compile. Much slower →
report env issue (`XDG_CACHE_HOME`, `HOME`, HF path split) before calibration.

---

## Proceed to calibration

**All mandatory gates (0–8 for your scope) pass** → follow `SKILL.md` for
`tune.py run` (dual-terminal tail, poll ≤120s, strict audit).

### Agent poll interval (P0 — every ≤120s while `tune.py run` is active)

Never blind-wait more than **2 minutes**. Each cycle:

```bash
python .claude/skills/tune-ci-thresholds/tune.py status --run-dir <run-dir>
python .claude/skills/tune-ci-thresholds/tune.py strict-audit --run-dir <run-dir>
```

Report **`strict-audit` N/N ✓** to the user, not `status ok/total` alone.

Before `run`:

- Confirm scope with user unless `handoff:` is explicit.
- Update `handoff:` in host profile when pausing mid-run.

**Forbidden:**

- Document or run `docker run` inside this skill
- `tune.py run` with pytest `-x`
- Symlinks for `HF_HOME` / `SEEDTTS_SIM_CACHE_DIR` when host profile is active
- `prepare_omni_venv.sh` / bulk downloads when precheck is green or one-repo ✗
- Start calibration while any Gate 8 model shows not `precheck OK`
- Proceed while strict audit has △/✗ repeats
- Blind-wait **>120s** without `status` + `strict-audit` during active calibration
- Fix env without reporting first (unless user explicitly asked)
