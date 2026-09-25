#!/usr/bin/env bash
# note (luojiaxuan): reads the reference-only commit back against the run.
set -uo pipefail
umask 077
C=/data/calibrations/20260925T0135Z
mkdir -p /github /root/.cache
ln -sfn "$C/home" /github/home
ln -sfn /data/cache/huggingface /root/.cache/huggingface
export HOME=/github/home OMNI_CI_HOME=/github/home/calibration HF_HOME=/root/.cache/huggingface TUNE_HOST=sglang-h100-ci
git -C "$C/post/repo" checkout -q 0e99d8ad
TUNE_REPO_ROOT="$C/post/repo" /github/home/calibration/omni/bin/python /data/tools/calibrate-h100-ci-dda3c7c/calibration/tune.py apply-plan --run-dir "$C/run_tts" > "$C/reports/post-apply-plan-0e99d8ad.txt" 2>&1
echo "$(date -u +%FT%TZ) post apply-plan at 0e99d8ad rc=$? directions=$(grep -o '"direction": "[a-z]*"' "$C/reports/post-apply-plan-0e99d8ad.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')" | tee -a "$C/evidence/events.log"
