#!/usr/bin/env bash

set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pin_to_ci_cpuset.sh"

max_attempts="${OMNI_CI_MAX_ATTEMPTS:-3}"
retry_delay_seconds="${OMNI_CI_RETRY_DELAY_SECONDS:-10}"
stage_label="${OMNI_CI_STAGE_LABEL:-${GITHUB_JOB:-pytest stage}}"

if ! [[ "${max_attempts}" =~ ^[0-9]+$ ]] || [ "${max_attempts}" -lt 1 ]; then
  echo "::error::OMNI_CI_MAX_ATTEMPTS must be a positive integer; got '${max_attempts}'"
  exit 2
fi

if ! [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "::error::OMNI_CI_RETRY_DELAY_SECONDS must be a non-negative integer; got '${retry_delay_seconds}'"
  exit 2
fi

if [ "${1:-}" = "--" ]; then
  shift
fi

if [ "$#" -eq 0 ]; then
  echo "::error::run_flaky_pytest.sh requires a pytest command"
  exit 2
fi

if [ "$1" = "pytest" ]; then
  shift
  set -- "${OMNI_CI_PYTHON:?OMNI_CI_PYTHON is not set}" -m pytest "$@"
fi

slug="$(printf '%s' "${stage_label}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-//;s/-$//')"
if [ -z "${slug}" ]; then
  slug="pytest-stage"
fi

log_root="${RUNNER_TEMP:-/tmp}/omni-ci-retry"
mkdir -p "${log_root}"

should_retry() {
  local status="$1"
  local _log_file="$2"

  [ "${status}" -ne 0 ]
}

report_gpu_occupancy() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  echo "GPU occupancy at failure (index, memory.used, utilization):"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | sed 's/^/  gpu /'
  declare -A gpu_index_by_uuid=()
  local index uuid pid mem owner apps
  while IFS=, read -r index uuid; do
    index="${index// /}"
    uuid="${uuid// /}"
    [ -n "${uuid}" ] && gpu_index_by_uuid["${uuid}"]="${index}"
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
  apps="$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null)"
  if [ -z "${apps}" ]; then
    echo "  no compute processes visible"
    return 0
  fi
  while IFS=, read -r pid mem uuid; do
    pid="${pid// /}"
    uuid="${uuid// /}"
    [ -n "${pid}" ] || continue
    # Note (Jiaxin Deng): a PID nvidia-smi reports but /proc lacks lives in
    # another container or the host, so the card is shared with someone else.
    if [ -e "/proc/${pid}" ]; then owner="this-job"; else owner="EXTERNAL"; fi
    echo "  gpu ${gpu_index_by_uuid[${uuid}]:-?} pid=${pid} mem=${mem# } owner=${owner}"
  done <<< "${apps}"
}

cleanup_between_attempts() {
  echo "Cleaning GPU state before retry..."
  if ! bash .github/scripts/delete_gpu_process.sh --kill-orphans; then
    echo "::error::GPU cleanup failed before retry; not retrying with dirty GPU state"
    return 1
  fi
  if [ "${retry_delay_seconds}" -gt 0 ]; then
    sleep "${retry_delay_seconds}"
  fi
}

attempt=1
last_status=0

while [ "${attempt}" -le "${max_attempts}" ]; do
  log_file="${log_root}/${slug}-attempt-${attempt}.log"

  echo "::group::${stage_label} attempt ${attempt}/${max_attempts}"
  OMNI_CI_ATTEMPT="${attempt}" "$@" 2>&1 | tee "${log_file}"
  last_status="${PIPESTATUS[0]}"
  echo "::endgroup::"

  if [ "${last_status}" -eq 0 ]; then
    if [ "${attempt}" -gt 1 ]; then
      echo "::notice::${stage_label} passed on attempt ${attempt}/${max_attempts}"
    fi
    exit 0
  fi

  echo "${stage_label} attempt ${attempt}/${max_attempts} failed with exit code ${last_status}."
  echo "Attempt log: ${log_file}"
  report_gpu_occupancy

  if [ "${attempt}" -ge "${max_attempts}" ]; then
    break
  fi

  if should_retry "${last_status}" "${log_file}"; then
    echo "::warning::Retrying ${stage_label}; CI stages retry all failures by default."
    cleanup_between_attempts || exit "${last_status}"
    attempt=$((attempt + 1))
    continue
  fi

  echo "Not retrying ${stage_label}."
  exit "${last_status}"
done

echo "::error::${stage_label} failed after ${max_attempts} attempt(s)."
exit "${last_status}"
