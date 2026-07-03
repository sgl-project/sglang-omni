#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${OMNI_CI_HOME}/${VENV_NAME}/bin/python"
REPO_ROOT="$(pwd)"

if [ ! -x "${PYTHON}" ]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${PYTHON}" "${SCRIPT_DIR}/verify_omni_installed_pins.py" "${PYTHON}" "${REPO_ROOT}"
