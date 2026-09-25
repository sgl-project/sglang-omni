#!/usr/bin/env bash
# note (luojiaxuan): post-apply checks from the calibrate-h100-ci runbook: the
# reference-only commit must read back as equal against the calibration run,
# and the final commit must pass coverage and the target unit tests.
set -uo pipefail
umask 077
C=/data/calibrations/20260925T0135Z
P=$C/post
ev() { echo "$(date -u +%FT%TZ) $*" | tee -a "$C/evidence/events.log"; }
mkdir -p /github /root/.cache
ln -sfn "$C/home" /github/home
ln -sfn /data/cache/huggingface /root/.cache/huggingface
export HOME=/github/home OMNI_CI_HOME=/github/home/calibration HF_HOME=/root/.cache/huggingface
PY=/github/home/calibration/omni/bin/python
rm -rf "$P/repo" && git clone -q "$C/sglang-omni" "$P/repo" && git -C "$P/repo" fetch -q "$P/post.bundle" "refs/heads/*:refs/remotes/post/*"
cd "$P/repo"
git checkout -q 0e99d8ad || { ev "POST FAIL checkout 0e99d8ad"; exit 1; }
TUNE_REPO_ROOT="$P/repo" "$PY" /data/tools/calibrate-h100-ci-dda3c7c/calibration/tune.py apply-plan --run-dir "$C/run_tts" > "$C/reports/post-apply-plan-0e99d8ad.txt" 2>&1
ev "post apply-plan at 0e99d8ad rc=$? directions=$(grep -o '"direction": "[a-z]*"' "$C/reports/post-apply-plan-0e99d8ad.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
git checkout -q 4c72ead0 || { ev "POST FAIL checkout 4c72ead0"; exit 1; }
"$PY" /data/tools/calibrate-h100-ci-post/calibration/check_ci_coverage.py --repo-root "$P/repo" > "$C/reports/post-ci-coverage-4c72ead0.json" 2>&1
ev "post coverage at 4c72ead0 rc=$?"
"$PY" -m pytest -q -p no:cacheprovider tests/unit_test/benchmarks/test_benchmark_runner.py \
  tests/unit_test/benchmarks/test_tts_seedtts.py tests/unit_test/benchmarks/test_tts_seedtts_benchmark_config.py \
  tests/unit_test/ci/test_tts_ci_preset_contract.py > "$C/reports/post-unit-tests-4c72ead0.txt" 2>&1
ev "post unit tests at 4c72ead0 rc=$? $(tail -1 "$C/reports/post-unit-tests-4c72ead0.txt")"
ev "POST_DONE"
