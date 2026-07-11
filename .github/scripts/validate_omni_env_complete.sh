#!/usr/bin/env bash
# End-to-end setup completion gate (setup job after models + marker).
#
# Requires validate_omni_env_reusable plus a matching .omni-env-complete marker
# from a prior successful setup on this OMNI_CI_HOME.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME="$1"
MARKER="${OMNI_CI_HOME}/.omni-env-complete"

if ! bash "${SCRIPT_DIR}/validate_omni_env_reusable.sh" "${VENV_NAME}"; then
  exit 1
fi

if [ ! -f "${MARKER}" ]; then
  echo "missing completion marker ${MARKER}" >&2
  exit 1
fi

# shellcheck source=omni_ci_deps_hash.sh
source "${SCRIPT_DIR}/omni_ci_deps_hash.sh"
DEPS_HASH="$(omni_ci_deps_hash)"

MARKER_HASH="$(grep -E '^deps_hash=' "${MARKER}" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
MARKER_VENV="$(grep -E '^venv_name=' "${MARKER}" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
MARKER_AT="$(grep -E '^marked_at=' "${MARKER}" | head -1 | cut -d= -f2- | tr -d '[:space:]')"

if [ -z "${MARKER_HASH}" ] || [ "${MARKER_HASH}" != "${DEPS_HASH}" ]; then
  echo "marker deps_hash mismatch or missing: marker=${MARKER_HASH:-<empty>} current=${DEPS_HASH}" >&2
  exit 1
fi
if [ -z "${MARKER_VENV}" ] || [ "${MARKER_VENV}" != "${VENV_NAME}" ]; then
  echo "marker venv_name mismatch: marker=${MARKER_VENV:-<empty>} expected=${VENV_NAME}" >&2
  exit 1
fi
if [ -z "${MARKER_AT}" ]; then
  echo "marker missing marked_at in ${MARKER}" >&2
  exit 1
fi

echo "OMNI CI environment complete: ${OMNI_CI_HOME} (venv=${VENV_NAME}, deps_hash=${DEPS_HASH}, marked_at=${MARKER_AT})"
