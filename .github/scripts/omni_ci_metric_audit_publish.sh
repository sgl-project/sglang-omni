#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/omni_ci_metric_audit.py"
PYTHON_BIN="${OMNI_CI_AUDIT_PYTHON:-python3}"
STAGE_LABEL=""
MATCHES_FILE=""
ARTIFACT_SEARCH_ROOT=""
ARTIFACT_PATH_GLOBS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-label)
      STAGE_LABEL="$2"
      shift 2
      ;;
    --matches-file)
      MATCHES_FILE="$2"
      shift 2
      ;;
    --artifact-search-root)
      ARTIFACT_SEARCH_ROOT="$2"
      shift 2
      ;;
    --artifact-path-globs)
      ARTIFACT_PATH_GLOBS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${STAGE_LABEL}" ]]; then
  echo "--stage-label is required" >&2
  exit 2
fi

warn() {
  echo "::warning::Omni CI metric audit: $*" >&2
}

finish() {
  local status="$1"
  if [[ "${status}" -ne 0 && "${OMNI_CI_AUDIT_FAIL_ON_ERROR:-0}" == "1" ]]; then
    exit "${status}"
  fi
  exit 0
}

quote_sh() {
  local escaped=${1//\'/\'\\\'\'}
  printf "'%s'" "${escaped}"
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -z "${MATCHES_FILE}" || ! -f "${MATCHES_FILE}" ]]; then
  MATCHES_FILE="${TMP_DIR}/matches.txt"
  : > "${MATCHES_FILE}"
fi

CURRENT_JSON="${TMP_DIR}/current.json"
HISTORY_JSONL="${TMP_DIR}/history.jsonl"
ALERT_JSON="${TMP_DIR}/alert.json"
HISTORY_RECORD="${TMP_DIR}/history-record.jsonl"

if ! "${PYTHON_BIN}" "${PY_SCRIPT}" collect \
  --stage-label "${STAGE_LABEL}" \
  --matches-file "${MATCHES_FILE}" \
  --artifact-search-root "${ARTIFACT_SEARCH_ROOT}" \
  --artifact-path-globs "${ARTIFACT_PATH_GLOBS}" \
  --output "${CURRENT_JSON}"; then
  warn "failed to collect metrics"
  finish 1
fi

REMOTE="${OMNI_CI_AUDIT_REMOTE:-hyper00}"
REMOTE_DIR="${OMNI_CI_AUDIT_REMOTE_DIR:-/data02/jaxan/sglang-omni-ci-audit}"
MAIL_TO="${OMNI_CI_AUDIT_MAIL_TO:-luojiaxuan1215@gmail.com,notifications@github.com}"
THRESHOLD="${OMNI_CI_AUDIT_REGRESSION_THRESHOLD:-0.10}"
MIN_BASELINE_COUNT="${OMNI_CI_AUDIT_MIN_BASELINE_COUNT:-1}"
DOCKER_MODE="${OMNI_CI_AUDIT_REMOTE_DOCKER:-auto}"
DOCKER_MOUNT="${OMNI_CI_AUDIT_REMOTE_DOCKER_MOUNT:-/data02/jaxan:/data}"
DOCKER_IMAGE="${OMNI_CI_AUDIT_REMOTE_DOCKER_IMAGE:-busybox:latest}"
SSH_OPTS=(
  -T
  -o RemoteCommand=none
  -o RequestTTY=no
  -o BatchMode=yes
  -o ConnectTimeout="${OMNI_CI_AUDIT_SSH_CONNECT_TIMEOUT:-10}"
  -o StrictHostKeyChecking=accept-new
)

REMOTE_STORAGE_PATH="$("${PYTHON_BIN}" "${PY_SCRIPT}" storage-path --current "${CURRENT_JSON}")"

remote_direct() {
  local script="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE}" "${script}"
}

docker_container_path() {
  local host_mount="${DOCKER_MOUNT%%:*}"
  local container_mount="${DOCKER_MOUNT#*:}"
  container_mount="${container_mount%%:*}"
  if [[ "${REMOTE_DIR}" != "${host_mount}" && "${REMOTE_DIR}" != "${host_mount}/"* ]]; then
    return 1
  fi
  local suffix="${REMOTE_DIR#${host_mount}}"
  printf "%s%s" "${container_mount}" "${suffix}"
}

remote_docker() {
  local script="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE}" \
    "docker run --rm -i -v $(quote_sh "${DOCKER_MOUNT}") $(quote_sh "${DOCKER_IMAGE}") sh -lc $(quote_sh "${script}")"
}

USE_DOCKER=0
if [[ "${OMNI_CI_AUDIT_DISABLE_REMOTE:-0}" == "1" ]]; then
  warn "remote persistence disabled by OMNI_CI_AUDIT_DISABLE_REMOTE=1"
else
  mkdir_script="mkdir -p $(quote_sh "${REMOTE_DIR}/tmp") && test -w $(quote_sh "${REMOTE_DIR}/tmp")"
  if remote_direct "${mkdir_script}" >/dev/null 2>&1; then
    USE_DOCKER=0
  elif [[ "${DOCKER_MODE}" == "1" || "${DOCKER_MODE}" == "auto" ]]; then
    if CONTAINER_DIR="$(docker_container_path)"; then
      docker_script="mkdir -p $(quote_sh "${CONTAINER_DIR}/tmp") && test -w $(quote_sh "${CONTAINER_DIR}/tmp")"
      if remote_docker "${docker_script}" >/dev/null 2>&1; then
        USE_DOCKER=1
      else
        warn "cannot write ${REMOTE_DIR} directly or through remote Docker"
        finish 1
      fi
    else
      warn "remote dir ${REMOTE_DIR} is outside docker mount ${DOCKER_MOUNT}"
      finish 1
    fi
  else
    warn "cannot write ${REMOTE_DIR} on ${REMOTE}"
    finish 1
  fi
fi

if [[ "${OMNI_CI_AUDIT_DISABLE_REMOTE:-0}" != "1" ]]; then
  if [[ "${USE_DOCKER}" -eq 1 ]]; then
    CONTAINER_DIR="$(docker_container_path)"
    CONTAINER_EVENT_PATH="${CONTAINER_DIR}/${REMOTE_STORAGE_PATH}"
    CONTAINER_HISTORY="${CONTAINER_DIR}/metrics-history.jsonl"
    read_history_script="cat $(quote_sh "${CONTAINER_HISTORY}") 2>/dev/null || true"
    if ! remote_docker "${read_history_script}" > "${HISTORY_JSONL}"; then
      warn "failed to read metric history from ${REMOTE}:${REMOTE_DIR}"
      : > "${HISTORY_JSONL}"
    fi
  else
    REMOTE_EVENT_PATH="${REMOTE_DIR}/${REMOTE_STORAGE_PATH}"
    REMOTE_HISTORY="${REMOTE_DIR}/metrics-history.jsonl"
    read_history_script="cat $(quote_sh "${REMOTE_HISTORY}") 2>/dev/null || true"
    if ! remote_direct "${read_history_script}" > "${HISTORY_JSONL}"; then
      warn "failed to read metric history from ${REMOTE}:${REMOTE_DIR}"
      : > "${HISTORY_JSONL}"
    fi
  fi
else
  : > "${HISTORY_JSONL}"
fi

"${PYTHON_BIN}" "${PY_SCRIPT}" check \
  --current "${CURRENT_JSON}" \
  --history-jsonl "${HISTORY_JSONL}" \
  --alert-output "${ALERT_JSON}" \
  --threshold "${THRESHOLD}" \
  --min-baseline-count "${MIN_BASELINE_COUNT}" \
  --email-to "${MAIL_TO}" \
  --send-email || warn "metric regression check failed"

"${PYTHON_BIN}" "${PY_SCRIPT}" history-record \
  --current "${CURRENT_JSON}" \
  --output "${HISTORY_RECORD}" || {
  warn "failed to build history record"
  finish 1
}

if [[ "${OMNI_CI_AUDIT_DISABLE_REMOTE:-0}" == "1" ]]; then
  finish 0
fi

if [[ "${USE_DOCKER}" -eq 1 ]]; then
  event_dir="$(dirname "${CONTAINER_EVENT_PATH}")"
  write_event_script="mkdir -p $(quote_sh "${event_dir}") && cat > $(quote_sh "${CONTAINER_EVENT_PATH}.tmp") && mv $(quote_sh "${CONTAINER_EVENT_PATH}.tmp") $(quote_sh "${CONTAINER_EVENT_PATH}")"
  if ! remote_docker "${write_event_script}" < "${CURRENT_JSON}"; then
    warn "failed to write audit event to ${REMOTE}:${REMOTE_DIR}/${REMOTE_STORAGE_PATH}"
    finish 1
  fi
  append_script='
set -eu
mkdir -p "$1"
lock="$1/.history.lock"
i=0
while ! mkdir "$lock" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 30 ]; then
    echo "could not acquire history lock" >&2
    exit 1
  fi
  sleep 1
done
trap "rmdir \"$lock\"" EXIT
cat >> "$1/metrics-history.jsonl"
'
  if ! remote_docker "sh -c $(quote_sh "${append_script}") sh $(quote_sh "${CONTAINER_DIR}")" < "${HISTORY_RECORD}"; then
    warn "failed to append metric history on ${REMOTE}:${REMOTE_DIR}"
    finish 1
  fi
else
  event_dir="$(dirname "${REMOTE_EVENT_PATH}")"
  write_event_script="mkdir -p $(quote_sh "${event_dir}") && cat > $(quote_sh "${REMOTE_EVENT_PATH}.tmp") && mv $(quote_sh "${REMOTE_EVENT_PATH}.tmp") $(quote_sh "${REMOTE_EVENT_PATH}")"
  if ! remote_direct "${write_event_script}" < "${CURRENT_JSON}"; then
    warn "failed to write audit event to ${REMOTE}:${REMOTE_EVENT_PATH}"
    finish 1
  fi
  append_script='
set -eu
mkdir -p "$1"
lock="$1/.history.lock"
i=0
while ! mkdir "$lock" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 30 ]; then
    echo "could not acquire history lock" >&2
    exit 1
  fi
  sleep 1
done
trap "rmdir \"$lock\"" EXIT
cat >> "$1/metrics-history.jsonl"
'
  if ! remote_direct "sh -c $(quote_sh "${append_script}") sh $(quote_sh "${REMOTE_DIR}")" < "${HISTORY_RECORD}"; then
    warn "failed to append metric history on ${REMOTE}:${REMOTE_DIR}"
    finish 1
  fi
fi

echo "Omni CI metric audit persisted to ${REMOTE}:${REMOTE_DIR}/${REMOTE_STORAGE_PATH}"
finish 0
