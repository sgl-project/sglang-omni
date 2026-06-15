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
HOST="${OMNI_CI_HOME}/${VENV_NAME}"
DEPS_HASH_FILE="${OMNI_CI_HOME}/.deps-hash"
DEPS_HASH="$(sha256sum pyproject.toml | awk '{print $1}')"
LOCK_DIR="${UV_CACHE_DIR:-/github/home/.cache/uv}"
mkdir -p "${LOCK_DIR}"
LOCK_FILE="${LOCK_DIR}/omni-venv-prepare-$(echo -n "${OMNI_CI_HOME}" | sha256sum | awk '{print $1}').lock"

exec 200>"${LOCK_FILE}"
if ! flock -w 3600 200; then
  echo "Timed out waiting for venv prepare lock: ${LOCK_FILE}" >&2
  exit 1
fi

reuse_venv=false
if [ -f "${DEPS_HASH_FILE}" ] \
  && [ "$(cat "${DEPS_HASH_FILE}")" = "${DEPS_HASH}" ] \
  && [ -x "${HOST}/bin/python" ] \
  && "${HOST}/bin/python" -c "import torch; import av; from whisper.normalizers import EnglishTextNormalizer" 2>/dev/null; then
  reuse_venv=true
fi

if [ "${reuse_venv}" = true ]; then
  echo "Reusing ${HOST} (pyproject.toml unchanged); refreshing editable install and dependency versions"
  rm -rf "./${VENV_NAME}"
  ln -sfn "${HOST}" "./${VENV_NAME}"
  source "${VENV_NAME}/bin/activate"
  uv pip install --upgrade -e .
  exit 0
fi

echo "Preparing fresh ${HOST} (deps changed or venv missing/corrupt)"
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

python -c "from whisper.normalizers import EnglishTextNormalizer"
echo "${DEPS_HASH}" > "${DEPS_HASH_FILE}"
