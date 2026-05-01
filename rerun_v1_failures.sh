#!/usr/bin/env bash
# Re-run the V1 CI stages that failed in the previous run, appending output to
# the existing log_CI.txt / log_CI_summary.txt.

set -uo pipefail

cd /data/chenyang/sglang-omni

export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export SGLANG_OMNI_SERVER_VERSION=v1
export SGLANG_SEEDTTS50_DIR=/data/chenyang/sglang-omni/.benchmark-data/seedtts50
export CUDA_VISIBLE_DEVICES=1,5

LOG=/data/chenyang/sglang-omni/log_CI.txt
SUMMARY=/data/chenyang/sglang-omni/log_CI_summary.txt

{
  echo ""
  echo "################################################################"
  echo "### RERUN of failed V1 stages started at $(date -Iseconds) ###"
  echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "################################################################"
  echo ""
} | tee -a "$LOG" "$SUMMARY"

run_stage() {
  local label="$1"
  shift
  local marker
  marker=$(mktemp)

  {
    echo ""
    echo "================================================================"
    echo "=== STAGE: $label (rerun)"
    echo "=== CMD: $*"
    echo "=== Started: $(date -Iseconds)"
    echo "================================================================"
  } | tee -a "$LOG"

  local t0=$(date +%s)
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local t1=$(date +%s)
  local dur=$((t1 - t0))

  {
    echo ""
    echo "=== EXIT CODE: $rc (duration ${dur}s)"
    echo "=== Finished: $(date -Iseconds)"
    echo ""
    echo "=== SERVER LOGS for $label (rerun) ==="
  } | tee -a "$LOG"

  while IFS= read -r f; do
    {
      echo ""
      echo "--- server log: $f ---"
      cat "$f"
    } >> "$LOG"
  done < <(find /tmp/pytest-of-root -name "server.log" -newer "$marker" -printf "%T@ %p\n" 2>/dev/null | sort -n | cut -d' ' -f2-)

  if [ "$rc" -eq 0 ]; then
    echo "[PASS rerun] $label  (${dur}s)" | tee -a "$SUMMARY"
  else
    echo "[FAIL rerun exit=$rc] $label  (${dur}s)" | tee -a "$SUMMARY"
  fi

  rm -f "$marker"
}

run_stage "stage-3-mmmu"            pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -v -s -x
run_stage "stage-6-mmsu-talker"     pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x
run_stage "stage-8-videomme-talker" pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py -v -s -x

{
  echo ""
  echo "=== RERUN ALL DONE at $(date -Iseconds) ==="
} | tee -a "$LOG" "$SUMMARY"
