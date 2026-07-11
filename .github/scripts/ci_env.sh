# Source CI-aligned env for all Omni benchmark tests (unit, Qwen3, TTS, Qwen3-ASR).
# Matches GitHub Actions omni-setup + tune-ci-thresholds auto_env.
set -a
export HOME=/github/home
export OMNI_CI_HOME="${OMNI_CI_HOME:-/github/home/calibration}"
export HF_HOME=/github/home/.cache/huggingface
export MODELSCOPE_CACHE=/github/home/.cache/modelscope
export XDG_CACHE_HOME="${OMNI_CI_HOME}/.cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export UV_CACHE_DIR=/github/home/.cache/uv
export TORCHINDUCTOR_CACHE_DIR="${OMNI_CI_HOME}/.torchinductor"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SEEDTTS_SIM_CACHE_DIR="${SEEDTTS_SIM_CACHE_DIR:-/github/home/seedtts-wavlm-sim}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
set +a
