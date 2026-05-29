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
HOST_CACHE="${OMNI_CI_HOME}/${VENV_NAME}"

if ! "${HOST_CACHE}/bin/python" -c "import torch; import av; from whisper.normalizers import EnglishTextNormalizer" 2>/dev/null; then
  echo "::error::Prepared ${VENV_NAME} at ${HOST_CACHE} is incomplete or corrupted"
  exit 1
fi

echo "Validated host venv cache: ${HOST_CACHE}"
