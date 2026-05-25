#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

VENV_NAME="$1"
VENV_PATH="${VENV_NAME}"

if ! "${VENV_PATH}/bin/python" -c "import torch; import av; from whisper.normalizers import EnglishTextNormalizer" 2>/dev/null; then
  echo "::error::Prepared ${VENV_NAME} venv is incomplete; refusing to publish cache"
  exit 1
fi

HOST_CACHE="/github/home/${VENV_NAME}"
rm -rf "${HOST_CACHE}"
cp -p -r "${VENV_PATH}" "${HOST_CACHE}"
echo "Saved ${VENV_NAME} cache to ${HOST_CACHE}"
