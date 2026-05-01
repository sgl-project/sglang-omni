#!/usr/bin/env bash
# Retry the still-failing V1 stages multiple times until each passes
# (or exhausts MAX_ATTEMPTS). Append everything to log_CI.txt.

set -uo pipefail

cd /data/chenyang/sglang-omni

export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export SGLANG_OMNI_SERVER_VERSION=v1
export SGLANG_SEEDTTS50_DIR=/data/chenyang/sglang-omni/.benchmark-data/seedtts50
export CUDA_VISIBLE_DEVICES=1,5

LOG=/data/chenyang/sglang-omni/log_CI.txt
SUMMARY=/data/chenyang/sglang-omni/log_CI_summary.txt
MAX_ATTEMPTS=5

{
  echo ""
  echo "################################################################"
  echo "### RETRY-UNTIL-PASS started at $(date -Iseconds)"
  echo "### MAX_ATTEMPTS per stage: $MAX_ATTEMPTS"
  echo "################################################################"
  echo ""
} | tee -a "$LOG" "$SUMMARY"

retry_stage() {
  local label="$1"
  shift
  local cmd=("$@")

  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    local marker
    marker=$(mktemp)

    {
      echo ""
      echo "================================================================"
      echo "=== STAGE: $label (retry attempt $attempt/$MAX_ATTEMPTS)"
      echo "=== CMD: ${cmd[*]}"
      echo "=== Started: $(date -Iseconds)"
      echo "================================================================"
    } | tee -a "$LOG"

    local t0=$(date +%s)
    "${cmd[@]}" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    local dur=$((t1 - t0))

    {
      echo ""
      echo "=== EXIT CODE: $rc (duration ${dur}s, attempt $attempt)"
      echo "=== Finished: $(date -Iseconds)"
      echo ""
      echo "=== SERVER LOGS for $label (attempt $attempt) ==="
    } | tee -a "$LOG"

    while IFS= read -r f; do
      {
        echo ""
        echo "--- server log: $f ---"
        cat "$f"
      } >> "$LOG"
    done < <(find /tmp/pytest-of-root -name "server.log" -newer "$marker" -printf "%T@ %p\n" 2>/dev/null | sort -n | cut -d' ' -f2-)

    rm -f "$marker"

    if [ "$rc" -eq 0 ]; then
      echo "[PASS retry attempt=$attempt] $label  (${dur}s)" | tee -a "$SUMMARY"
      return 0
    else
      echo "[FAIL retry attempt=$attempt exit=$rc] $label  (${dur}s)" | tee -a "$SUMMARY"
    fi
  done

  echo "[GIVE UP after $MAX_ATTEMPTS attempts] $label" | tee -a "$SUMMARY"
  return 1
}

retry_stage "stage-6-mmsu-talker"     pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x
retry_stage "stage-8-videomme-talker" pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py -v -s -x

{
  echo ""
  echo "=== RETRY ALL DONE at $(date -Iseconds) ==="
} | tee -a "$LOG" "$SUMMARY"
