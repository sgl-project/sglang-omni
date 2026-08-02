#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if [[ "${1:-}" == "--check" ]]; then
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py"
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py" --pre-install
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements/rocm.txt"
"${PYTHON_BIN}" -m pip install --no-deps -e "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py"
