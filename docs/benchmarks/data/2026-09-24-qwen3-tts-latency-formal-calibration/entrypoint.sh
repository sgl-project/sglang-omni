#!/usr/bin/env bash
# note (luojiaxuan): container initial command for one native tune.py calibration
# (calibrate-h100-ci runbook steps 4 to 8). Every step appends to events.log; a
# failing evidence step records its exit code and the driver moves on only where
# the runbook says the result is informational.
set -uo pipefail
umask 077
: "${CALIB_ROOT:?}" "${TARGET_SHA:?}" "${STAGES:?}" "${TUNE_REPO_ROOT:?}" "${TUNE_SKILL_ROOT:?}"
RUN="$CALIB_ROOT/run_tts"
mkdir -p "$CALIB_ROOT"/{runtime,evidence,reports,home/.cache} /github
ln -sfn "$CALIB_ROOT/home" /github/home
ln -sfn /data/cache/huggingface "$CALIB_ROOT/home/.cache/huggingface"
# note (luojiaxuan): the sglang-h100-ci host profile pins the Hugging Face hub to
# /root/.cache/huggingface, the CI runner layout, and tune.py exports that path;
# both names resolve to the one cache under the personal root.
mkdir -p /root/.cache
ln -sfn /data/cache/huggingface /root/.cache/huggingface
exec > >(tee -a "$CALIB_ROOT/runtime/driver.log") 2>&1
ev() { echo "$(date -u +%FT%TZ) $*" | tee -a "$CALIB_ROOT/evidence/events.log"; }
fail() { ev "FAIL $*"; exit 1; }

export HOME=/github/home OMNI_CI_HOME=/github/home/calibration
export HF_HOME=/root/.cache/huggingface XDG_CACHE_HOME=/github/home/calibration/.cache
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=0
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple UV_CACHE_DIR=/github/home/.cache/uv
export TORCHINDUCTOR_CACHE_DIR=/github/home/calibration/.torchinductor
export SGLANG_CACHE_DIR=/github/home/sglang-cache FLASHINFER_WORKSPACE_BASE=/root
export FLASHINFER_JIT_DEBUG=0 FLASHINFER_DISABLE_VERSION_CHECK=1 SGLANG_OMNI_AUTO_CLONE=0

ev "driver start pid=$$ cpuset=$(taskset -pc $$ | awk '{print $NF}') OMNI_CI_CPUSET=${OMNI_CI_CPUSET:-unset}"
cd "$TUNE_REPO_ROOT" || fail "no checkout"
[ "$(git rev-parse HEAD)" = "$TARGET_SHA" ] || fail "checkout is $(git rev-parse HEAD), expected $TARGET_SHA"
git status --porcelain --untracked-files=no > "$CALIB_ROOT/evidence/dirty.txt"
ev "source $TARGET_SHA dirty_lines=$(wc -l < "$CALIB_ROOT/evidence/dirty.txt")"

bash .github/scripts/reconcile_omni_ci_env.sh omni || fail "env reconcile"
bash .github/scripts/validate_omni_env_reusable.sh omni > "$CALIB_ROOT/evidence/env-validate.txt" 2>&1 || fail "env validate"
ev "env $(tail -1 "$CALIB_ROOT/evidence/env-validate.txt")"
bash .github/scripts/prepare_rust_router.sh build || fail "router build"
SGLANG_OMNI_ROUTER_BIN="$(bash .github/scripts/prepare_rust_router.sh path)"
export SGLANG_OMNI_ROUTER_BIN
sha256sum "$SGLANG_OMNI_ROUTER_BIN" > "$CALIB_ROOT/evidence/router.sha256"
git rev-parse HEAD:sglang_omni_router/rust >> "$CALIB_ROOT/evidence/router.sha256"
ev "router $(head -c 16 "$CALIB_ROOT/evidence/router.sha256")"

PY="$OMNI_CI_HOME/omni/bin/python"
export TUNE_VENV_PYTHON="$PY"
"$PY" -c 'import torch; n=torch.cuda.device_count(); [torch.ones(1, device=f"cuda:{i}").sum().item() for i in range(n)]; print("CUDA_SMOKE_OK", n, torch.cuda.get_device_name(0))' \
  > "$CALIB_ROOT/evidence/cuda-smoke.log" 2>&1 || fail "cuda smoke"
ev "$(tail -1 "$CALIB_ROOT/evidence/cuda-smoke.log")"
"$PY" -m pip check > "$CALIB_ROOT/evidence/pip-check.txt" 2>&1
ev "pip check rc=$? (recorded, informational)"
"$PY" -m pytest -q -p no:cacheprovider tests/unit_test/benchmarks/test_benchmark_runner.py \
  tests/unit_test/benchmarks/test_tts_seedtts.py tests/unit_test/benchmarks/test_tts_seedtts_benchmark_config.py \
  tests/unit_test/ci/test_tts_ci_preset_contract.py > "$CALIB_ROOT/evidence/target-unit-tests.txt" 2>&1 \
  || fail "target unit tests"
ev "target unit tests $(tail -1 "$CALIB_ROOT/evidence/target-unit-tests.txt")"
"$PY" - > "$CALIB_ROOT/evidence/assets.txt" 2>&1 <<'PY' || fail "asset download"
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
    print(repo, snapshot_download(repo))
print("dataset", snapshot_download("zhaochenyang20/seed-tts-eval-arrow", repo_type="dataset"))
PY
ev "assets ready"
(cd "$TUNE_SKILL_ROOT/.." && "$PY" -m pytest -q calibration/tests/test_attempt_archive.py) \
  > "$CALIB_ROOT/evidence/attempt-retention.txt" 2>&1 || fail "attempt retention acceptance test"
ev "attempt retention $(tail -1 "$CALIB_ROOT/evidence/attempt-retention.txt")"
"$PY" "$TUNE_SKILL_ROOT/check_ci_coverage.py" --repo-root "$TUNE_REPO_ROOT" > "$CALIB_ROOT/evidence/ci-coverage.json" 2>&1 \
  || fail "ci coverage (evidence/ci-coverage.json)"
ev "ci coverage ok"
"$PY" "$TUNE_SKILL_ROOT/tune.py" --model tts stages-list > "$CALIB_ROOT/evidence/stages-list.txt" 2>&1 || fail "stages-list"

ev "sampling start stages=$STAGES"
"$PY" "$TUNE_SKILL_ROOT/tune.py" --model tts run --stages "$STAGES" --repeats 5 --output-dir "$RUN"
ev "run rc=$?"
for cmd in status strict-audit report apply-plan; do
  "$PY" "$TUNE_SKILL_ROOT/tune.py" "$cmd" --run-dir "$RUN" > "$CALIB_ROOT/reports/$cmd.txt" 2>&1
  ev "$cmd rc=$?"
done
ev "DRIVER_DONE"
