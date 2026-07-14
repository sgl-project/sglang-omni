#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/intel/oneapi/setvars.sh && "${SETVARS_COMPLETED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi

MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-Transcribe-Diarize}"
PORT="${PORT:-8000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-11500}"

# gpu=None prevents Omni's CUDA-only placement wrapper from calling
# torch.cuda.set_device. The MOSS stage itself is explicitly placed on xpu:0.
exec sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --decode-mode sync \
  --stages.asr.gpu none \
  --stages.asr.factory_args.device xpu:0 \
  --stages.asr.factory_args.max_new_tokens "${MAX_NEW_TOKENS}" \
  --stages.asr.factory_args.encoder_cache_size_bytes 0
