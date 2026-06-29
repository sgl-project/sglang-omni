#!/usr/bin/env bash
# Full rebuild of OMNI_CI_HOME (never reuses an existing tree).
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
HOST="${OMNI_CI_HOME}/${VENV_NAME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_HASH_FILE="${OMNI_CI_HOME}/.deps-hash"

# shellcheck source=omni_ci_deps_hash.sh
source "${SCRIPT_DIR}/omni_ci_deps_hash.sh"
DEPS_HASH="$(omni_ci_deps_hash)"

LOCK_DIR="${UV_CACHE_DIR:-/github/home/.cache/uv}"
mkdir -p "${LOCK_DIR}"
LOCK_FILE="${LOCK_DIR}/omni-venv-prepare-$(echo -n "${OMNI_CI_HOME}" | sha256sum | awk '{print $1}').lock"

exec 200>"${LOCK_FILE}"
if ! flock -w 3600 200; then
  echo "Timed out waiting for venv prepare lock: ${LOCK_FILE}" >&2
  exit 1
fi

echo "Preparing fresh ${HOST} (full rebuild)"
rm -f "${OMNI_CI_HOME}/.omni-env-complete"
rm -rf "${OMNI_CI_HOME}"
mkdir -p "${OMNI_CI_HOME}"
uv venv "${HOST}" -p 3.11

rm -rf "./${VENV_NAME}"
ln -sfn "${HOST}" "./${VENV_NAME}"
source "${VENV_NAME}/bin/activate"
uv pip install -e .

if ! python -c "import av" 2>/dev/null; then
  echo "PyAV native libraries corrupted in prepared venv, force-reinstalling..."
  uv pip install --force-reinstall --no-deps --no-cache av
fi

if ! python -c "from whisper.normalizers import EnglishTextNormalizer" 2>/dev/null; then
  echo "openai-whisper missing from prepared venv, installing pinned dependency..."
  uv pip install --force-reinstall --no-deps --no-cache openai-whisper==20250625
fi

if ! bash "${SCRIPT_DIR}/validate_omni_venv_imports.sh" "${VENV_NAME}"; then
  exit 1
fi

if ! bash "${SCRIPT_DIR}/verify_omni_installed_pins.sh" "${VENV_NAME}"; then
  echo "::error::Fresh venv does not match pyproject.toml pins" >&2
  exit 1
fi

echo "${DEPS_HASH}" > "${DEPS_HASH_FILE}"
echo "Fresh environment ready at ${HOST} (deps_hash=${DEPS_HASH})"
