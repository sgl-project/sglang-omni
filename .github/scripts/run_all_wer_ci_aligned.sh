#!/usr/bin/env bash
# Full WER CI sweep (Qwen3 then TTS). One instance at a time (flock).
#
# Two-terminal contract (see tune-ci-thresholds § Two-terminal supervision):
#   Tab A (supervision) — tail -f /tmp/wer_ci_*.log  → detailed log
#   Tab B (job)         — this script                 → milestones only
# Verbose pytest output: >> LOG. Never tee to Tab B stdout.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

LOCK_FILE=/tmp/wer_ci_sweep.lock
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "ERROR: another WER sweep holds ${LOCK_FILE} — stop it first"
    exit 1
fi

log_line() {
    local log_path="$1"
    shift
    echo "$*" >> "${log_path}"
}

log_and_echo() {
    local log_path="$1"
    shift
    echo "$*"
    echo "$*" >> "${log_path}"
}

ensure_gpus_idle() {
    local label="$1"
    local log_path="$2"
    log_line "${log_path}" "=== ensure_gpus_idle before ${label} $(date -Is) ==="
    OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB=2048 \
    OMNI_CI_GPU_CLEAN_WAIT_SECONDS=600 \
    OMNI_CI_GPU_CLEAN_POLL_SECONDS=5 \
    bash .github/scripts/delete_gpu_process.sh --kill-orphans >> "${log_path}" 2>&1 || return 1
    sleep 3
}

setup_omni_env() {
    local log_path="$1"
    # shellcheck source=/dev/null
    source omni/bin/activate
    # shellcheck source=/dev/null
    source .github/scripts/ci_env.sh
    local env_msg
    env_msg="$(python -c "
import os
assert os.environ['TORCHINDUCTOR_CACHE_DIR'].startswith(os.environ['OMNI_CI_HOME']), (
    os.environ['TORCHINDUCTOR_CACHE_DIR'], os.environ['OMNI_CI_HOME']
)
print('env OK', os.environ['OMNI_CI_HOME'], os.environ['TORCHINDUCTOR_CACHE_DIR'])
")"
    log_and_echo "${log_path}" "${env_msg}"
}

run_one() {
    local label="$1"
    local log_path="$2"
    shift 2
    log_and_echo "${log_path}" "===== START ${label} $(date -Is) ====="
    echo "  [supervision] tail -f ${log_path}"
    if ! ensure_gpus_idle "${label}" "${log_path}"; then
        log_and_echo "${log_path}" "===== ABORT ${label} GPU not idle ====="
        return 1
    fi
    "$@" >> "${log_path}" 2>&1
    local exit_code=$?
    if [ "${exit_code}" -eq 0 ]; then
        log_and_echo "${log_path}" "===== PASS ${label} $(date -Is) ====="
    else
        log_and_echo "${log_path}" "===== FAIL ${label} $(date -Is) exit=${exit_code} ====="
    fi
    ensure_gpus_idle "post-${label}" "${log_path}" >> "${log_path}" 2>&1 || true
    sleep 5
    return "${exit_code}"
}

LOG=/tmp/wer_ci_qwen3.log
: > "${LOG}"
echo "WER sweep started — milestones here; full log: tail -f ${LOG}"

setup_omni_env "${LOG}"

run_one mmmu_talker_wer "${LOG}" \
    pytest tests/test_model/test_qwen3_omni_mmmu_talker_ci.py::test_mmmu_talker_wer -v -s

run_one videomme_talker_wer "${LOG}" \
    pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py::test_videomme_talker_wer -v -s

run_one videoamme_talker_tp2_wer "${LOG}" \
    pytest tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py::test_videoamme_talker_tp2_wer -v -s

run_one mmsu_talker_wer "${LOG}" \
    pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py::test_mmsu_talker_wer -v -s

run_one qwen3_tts_wer "${LOG}" \
    pytest tests/test_model/test_qwen3_omni_tts_ci.py::test_voice_cloning_wer -v -s

log_and_echo "${LOG}" "===== QWEN3 WER SECTION FINISHED $(date -Is) ====="

run_one tts_nonstream_wer "${LOG}" \
    pytest tests/test_model/test_tts_ci.py --tts-stage tts-stage-1-nonstream \
    -k "test_voice_cloning_non_streaming or test_voice_cloning_wer" -v -s

run_one tts_stream_wer "${LOG}" \
    pytest tests/test_model/test_tts_ci.py --tts-stage tts-stage-2-stream \
    -k "test_voice_cloning_streaming or test_voice_cloning_streaming_wer" -v -s

log_and_echo "${LOG}" "===== ALL WER CI FINISHED $(date -Is) ====="
