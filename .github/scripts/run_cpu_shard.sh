#!/usr/bin/env bash
set -euo pipefail
partition=$1
count=$2
shift 2
# The runner checkout is read-only; pytest and its child processes get a private tree.
git -c safe.directory=/ci-source clone -q --no-local /ci-source .
test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
export OMNI_CI_STAGE_LABEL="unit-test-$partition" RUNNER_TEMP=/tmp/omni-runner
export HOME=/github/home HF_HOME=/github/home/.cache/huggingface
export XDG_CACHE_HOME="$OMNI_CI_HOME/.cache" UV_CACHE_DIR=/data/omni-ci/uv-cache
export SGLANG_CACHE_DIR=/data/omni-ci/sglang-cache SGLANG_JIT_CACHE_DIR=/data/omni-ci/sglang-cache/jit
export FLASHINFER_WORKSPACE_BASE=/root FLASHINFER_JIT_DEBUG=0 SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0
export HF_HUB_DISABLE_XET=0 HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export OMNI_CI_PYTHON="$OMNI_CI_HOME/omni/bin/python" PYTHONPATH="$PWD"
source .github/scripts/pin_to_ci_cpuset.sh
bash .github/scripts/validate_omni_ci_home.sh
ln -sfn "$OMNI_CI_HOME/omni" omni
if ! bash .github/scripts/validate_omni_env_reusable.sh omni; then
    bash .github/scripts/reconcile_omni_ci_env.sh omni
    bash .github/scripts/validate_omni_env_reusable.sh omni
fi
export SGLANG_OMNI_ROUTER_BIN
SGLANG_OMNI_ROUTER_BIN=$(bash .github/scripts/prepare_rust_router.sh path)
source omni/bin/activate
exec timeout --kill-after=15s 30m bash .github/scripts/run_flaky_pytest.sh \
    pytest "$@" -v -m "not benchmark and not accelerator" -x \
    -p scripts.ci.utils.pytest_partition --auto-partition-id "$partition" \
    --auto-partition-size "$count" --partition-model-file .github/unit_test_durations.json
