#!/usr/bin/env bash
set -euo pipefail
: "${NPU_CI_IMAGE:?Set NPU_CI_IMAGE to a matching image digest}"
: "${NPU_CI_DATA_DIR:?Set NPU_CI_DATA_DIR to the data directory mounted in the test Pod}"
: "${NPU_CI_MODEL:?Set NPU_CI_MODEL to qwen3-tts or qwen3-asr}"
[[ "$NPU_CI_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'An image digest is required' >&2; exit 2; }
[[ "$NPU_CI_MODEL" == qwen3-tts || "$NPU_CI_MODEL" == qwen3-asr ]] || exit 2
[[ "$NPU_CI_DATA_DIR" == /* && "$NPU_CI_DATA_DIR" != / && -d "$NPU_CI_DATA_DIR" ]] || exit 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$repo_root/npu-ci-results"
run_dir="$(mktemp -d "$repo_root/npu-ci-results/$NPU_CI_MODEL-XXXXXX")"
exec > >(tee "$run_dir/runner.log") 2>&1
printf '%s\n' "$NPU_CI_IMAGE" > "$run_dir/image.txt"
git -C "$repo_root" rev-parse HEAD > "$run_dir/source-sha.txt"
git -C "$repo_root" status --short > "$run_dir/source-status.txt"
for path in "$NPU_CI_DATA_DIR/models/$NPU_CI_MODEL/config.json" "$NPU_CI_DATA_DIR/configs/$NPU_CI_MODEL.yaml"; do
  [[ -f "$path" ]] || { echo "Missing test input: $path" >&2; exit 2; }
done
if [[ "$NPU_CI_MODEL" == qwen3-tts ]]; then
  [[ -d "$NPU_CI_DATA_DIR/models/qwen3-tts/speech_tokenizer" ]] || { echo 'Missing TTS speech_tokenizer directory' >&2; exit 2; }
else
  [[ -f "$NPU_CI_DATA_DIR/asr/cases.json" ]] || { echo 'Missing ASR cases.json' >&2; exit 2; }
fi

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
cp "$NPU_CI_DATA_DIR/configs/$NPU_CI_MODEL.yaml" "$run_dir/serving-config.yaml"
if [[ "$NPU_CI_MODEL" == qwen3-tts ]]; then
  export OMNI_NPU_TTS_MODEL="$NPU_CI_DATA_DIR/models/qwen3-tts"
  export OMNI_NPU_TTS_CONFIG="$NPU_CI_DATA_DIR/configs/qwen3-tts.yaml"
  export OMNI_NPU_TTS_OUTPUT="$run_dir/tts"
  suite=tests/test_model/test_npu_tts.py
else
  export OMNI_NPU_ASR_MODEL="$NPU_CI_DATA_DIR/models/qwen3-asr"
  export OMNI_NPU_ASR_CONFIG="$NPU_CI_DATA_DIR/configs/qwen3-asr.yaml"
  export OMNI_NPU_ASR_CASES="$NPU_CI_DATA_DIR/asr/cases.json"
  export OMNI_NPU_ASR_OUTPUT="$run_dir/asr"
  suite=tests/test_model/test_npu_asr.py
fi
python -m pytest "$suite" -v -s --junitxml="$run_dir/junit.xml"
