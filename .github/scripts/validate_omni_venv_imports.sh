#!/usr/bin/env bash
# Import probe for the Omni CI venv (matches packages exercised in real CI jobs).
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

if [ ! -x "${PYTHON}" ]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON}" -c "
import av
# Import before Torch to catch conflicting system NCCL libraries.
import llama_cpp
import torch
import transformers
import sglang
import zhon.hanzi
from whisper.normalizers import EnglishTextNormalizer
import shutil
from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
import dac
from neucodec import NeuCodec
assert shutil.which('sox'), 'Qwen3-TTS requires the system sox executable'
" 2>/dev/null; then
  echo "::error::${VENV_NAME} import probe failed at ${OMNI_CI_HOME}/${VENV_NAME}" >&2
  exit 1
fi

echo "Import probe ok: ${OMNI_CI_HOME}/${VENV_NAME}"
