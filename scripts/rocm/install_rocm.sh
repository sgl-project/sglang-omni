#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
UV_BIN="${UV:-uv}"

if [[ "${1:-}" == "--check" ]]; then
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py"
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py" --pre-install
if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi
"${UV_BIN}" pip install \
    --python "${PYTHON_BIN}" \
    --group "${REPO_ROOT}/pyproject.toml:rocm"
"${UV_BIN}" pip install \
    --python "${PYTHON_BIN}" \
    --no-deps \
    --editable "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_rocm.py"
