#!/usr/bin/env bash
set -euo pipefail
venv=$(realpath "${1:?expected venv path}")
COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
COSYVOICE_PATH="${venv}/src/CosyVoice"
git clone --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_PATH}"
git -C "${COSYVOICE_PATH}" checkout --detach "${COSYVOICE_COMMIT}"
git -C "${COSYVOICE_PATH}" submodule update --init --depth=1 third_party/Matcha-TTS
# Note (Jiannan Li): TTS jobs replace PYTHONPATH, so persist both source paths in the venv.
printf '%s\n' "${COSYVOICE_PATH}" "${COSYVOICE_PATH}/third_party/Matcha-TTS" \
  > "${venv}/lib/python3.12/site-packages/cosyvoice.pth"
