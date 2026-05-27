#!/usr/bin/env bash

set -uo pipefail

max_attempts="${OMNI_CI_MAX_ATTEMPTS:-2}"
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

slug="$(printf '%s' "${stage_label}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-//;s/-$//')"
if [ -z "${slug}" ]; then
  slug="pytest-stage"
fi

log_root="${RUNNER_TEMP:-/tmp}/omni-ci-retry"
mkdir -p "${log_root}"

has_hard_failure() {
  local log_file="$1"

  grep -Eiq \
    '(CUDA out of memory|OutOfMemoryError|No space left on device|Segmentation fault|core dumped|exit code 13[79]|returned non-zero exit status 13[79]|SIGKILL|SIGTERM|Server failed to start|server process.*exit|Connection refused|connection refused|Health check failed)' \
    "${log_file}"
}

has_retryable_metric_assertion() {
  local log_file="$1"

  grep -Eiq \
    '(AssertionError|FAILED .* - AssertionError|E +assert).*(threshold|throughput_qps|tok_per_s_agg|latency_mean_s|rtf_mean|accuracy|WER|wer_|speaker_similarity|similarity)' \
    "${log_file}" ||
    grep -Eiq \
      '(threshold|throughput_qps|tok_per_s_agg|latency_mean_s|rtf_mean|accuracy|WER|wer_|speaker_similarity|similarity).*(AssertionError|FAILED|E +assert)' \
      "${log_file}"
}

should_retry() {
  local status="$1"
  local log_file="$2"

  # Pytest exits 1 for ordinary test failures. Exit codes 2+ indicate
  # interruption, internal errors, bad invocation, or collection problems.
  [ "${status}" -eq 1 ] || return 1
  grep -Eq '=+ FAILURES =+|FAILED .+ - AssertionError|AssertionError' "${log_file}" || return 1
  ! has_hard_failure "${log_file}" || return 1
  has_retryable_metric_assertion "${log_file}"
}

cleanup_between_attempts() {
  echo "Cleaning GPU state before retry..."
  bash .github/scripts/delete_gpu_process.sh || {
    echo "::warning::GPU cleanup failed before retry; continuing to preserve the retry signal"
  }
  rm -rf "${XDG_CACHE_HOME:-/github/home/.cache}/flashinfer" || true
  if [ "${retry_delay_seconds}" -gt 0 ]; then
    sleep "${retry_delay_seconds}"
  fi
}

attempt=1
last_status=0

while [ "${attempt}" -le "${max_attempts}" ]; do
  log_file="${log_root}/${slug}-attempt-${attempt}.log"

  echo "::group::${stage_label} attempt ${attempt}/${max_attempts}"
  "$@" 2>&1 | tee "${log_file}"
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

  if [ "${attempt}" -ge "${max_attempts}" ]; then
    break
  fi

  if should_retry "${last_status}" "${log_file}"; then
    echo "::warning::Retrying ${stage_label}; failure looks like a flaky metric assertion."
    cleanup_between_attempts
    attempt=$((attempt + 1))
    continue
  fi

  echo "Not retrying ${stage_label}; failure does not match the flaky metric assertion policy."
  exit "${last_status}"
done

echo "::error::${stage_label} failed after ${max_attempts} attempt(s)."
exit "${last_status}"
