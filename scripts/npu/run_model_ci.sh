#!/usr/bin/env bash
set -euo pipefail
: "${NPU_CI_IMAGE:?Set NPU_CI_IMAGE to a matching image digest}"
: "${NPU_CI_MODEL:?Set NPU_CI_MODEL to qwen3-tts or qwen3-asr}"
[[ "$NPU_CI_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'An image digest is required' >&2; exit 2; }
[[ "$NPU_CI_MODEL" == qwen3-tts || "$NPU_CI_MODEL" == qwen3-asr ]] || exit 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$repo_root/npu-ci-results"
run_dir="$(mktemp -d "$repo_root/npu-ci-results/$NPU_CI_MODEL-XXXXXX")"
exec > >(tee "$run_dir/runner.log") 2>&1
printf '%s\n' "$NPU_CI_IMAGE" > "$run_dir/image.txt"
git -C "$repo_root" rev-parse HEAD > "$run_dir/source-sha.txt"
git -C "$repo_root" status --short > "$run_dir/source-status.txt"
model_cache=/root/.cache/modelscope/hub/models/Qwen
if [[ "$NPU_CI_MODEL" == qwen3-tts ]]; then
  model_path="$model_cache/Qwen3-TTS-12Hz-0.6B-CustomVoice"
  serving_config="$repo_root/examples/configs/qwen3_tts_0_6b_customvoice_npu.yaml"
  [[ -d "$model_path/speech_tokenizer" ]] || { echo "Missing TTS speech_tokenizer: $model_path/speech_tokenizer" >&2; exit 2; }
  export OMNI_NPU_TTS_MODEL="$model_path"
  export OMNI_NPU_TTS_CONFIG="$serving_config"
  export OMNI_NPU_TTS_OUTPUT="$run_dir/tts"
  suite=tests/test_model/test_npu_tts.py
else
  model_path="$model_cache/Qwen3-ASR-1.7B"
  serving_config="$repo_root/examples/configs/qwen3_asr_npu.yaml"
  export OMNI_NPU_ASR_MODEL="$model_path"
  export OMNI_NPU_ASR_CONFIG="$serving_config"
  export OMNI_NPU_ASR_CASES="$repo_root/tests/data/npu_asr/cases.json"
  export OMNI_NPU_ASR_OUTPUT="$run_dir/asr"
  suite=tests/test_model/test_npu_asr.py
fi
for path in "$model_path/config.json" "$serving_config"; do
  [[ -f "$path" ]] || { echo "Missing test input: $path" >&2; exit 2; }
done

# Vendor environment scripts may reference unset shell variables.
set +u
# shellcheck source=/dev/null
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
set -u
source_dir="$(mktemp -d "${TMPDIR:-/tmp}/omni-ci-source-XXXXXX")"
tar -C "$repo_root" --exclude=.git --exclude=npu-ci-results --exclude=.venv \
  --exclude=__pycache__ -cf - . | tar -C "$source_dir" -xf -
cd "$source_dir"
bash scripts/npu/install_npu.sh --check 2>&1 | tee "$run_dir/environment-check.log"
cp pyproject_npu.toml pyproject.toml
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install pytest==8.3.2 jiwer==4.0.0 rapidfuzz==3.14.1
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OMNI_RUN_NPU_TESTS=1 HF_HUB_OFFLINE=1
python -m pip freeze > "$run_dir/packages.txt"
npu-smi info > "$run_dir/npu-smi.txt"
cp "$serving_config" "$run_dir/serving-config.yaml"
python -m pytest "$suite" -v -s --junitxml="$run_dir/junit.xml"
