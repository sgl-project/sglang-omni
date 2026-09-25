#!/usr/bin/env bash
# Trusted runner only: privileged Docker is not a device security boundary.
set -euo pipefail
: "${NPU_CI_IMAGE:?Set NPU_CI_IMAGE to a matching image digest}"
: "${NPU_CI_DEVICE:?Set NPU_CI_DEVICE to one reserved physical NPU}"
: "${NPU_CI_DATA_DIR:?Set NPU_CI_DATA_DIR to the prepared host data directory}"
: "${NPU_CI_MODEL:?Set NPU_CI_MODEL to qwen3-tts or qwen3-asr}"
[[ "$NPU_CI_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'An image digest is required' >&2; exit 2; }
[[ "$NPU_CI_DEVICE" =~ ^[0-9]+$ ]] || { echo 'NPU_CI_DEVICE must be one physical ID' >&2; exit 2; }
[[ "$NPU_CI_MODEL" == qwen3-tts || "$NPU_CI_MODEL" == qwen3-asr ]] || exit 2
[[ "$NPU_CI_DATA_DIR" == /* && "$NPU_CI_DATA_DIR" != / && -d "$NPU_CI_DATA_DIR" ]] || exit 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$repo_root/npu-ci-results"
run_dir="$(mktemp -d "$repo_root/npu-ci-results/run-XXXXXX")"
container_name="omni-npu-model-$(basename "$run_dir")-$$"
# shellcheck disable=SC2317
cleanup() {
  local status=$?
  trap - EXIT
  if docker inspect "$container_name" >/dev/null 2>&1; then
    docker logs "$container_name" > "$run_dir/container.log" 2>&1 || true
    docker cp "$container_name:/results/." "$run_dir/" || true
    docker rm -f "$container_name" >/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
device_args=()
for device in "/dev/davinci${NPU_CI_DEVICE}" /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "$device" ]] || { echo "Missing device: $device" >&2; exit 2; }
  device_args+=(--device "$device")
done
mount_args=()
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/sbin /etc/ascend_install.info; do
  [[ -e "$path" ]] || { echo "Missing runtime path: $path" >&2; exit 2; }
  mount_args+=(-v "$path:$path:ro")
done
if [[ -d /var/queue_schedule ]]; then
  mount_args+=(-v /var/queue_schedule:/var/queue_schedule)
fi
git -C "$repo_root" rev-parse HEAD > "$run_dir/source-sha.txt"
git -C "$repo_root" status --short > "$run_dir/source-status.txt"
docker image inspect "$NPU_CI_IMAGE" > "$run_dir/image.json" 2>/dev/null || {
  docker pull "$NPU_CI_IMAGE"
  docker image inspect "$NPU_CI_IMAGE" > "$run_dir/image.json"
}
docker create --name "$container_name" --privileged --network host --shm-size 8g \
  "${device_args[@]}" "${mount_args[@]}" \
  -v "$repo_root:/checkout:ro" -v "$NPU_CI_DATA_DIR:/data:ro" \
  -e "ASCEND_RT_VISIBLE_DEVICES=$NPU_CI_DEVICE" -e "NPU_CI_MODEL=$NPU_CI_MODEL" \
  -e OMP_NUM_THREADS=8 -e OMNI_RUN_NPU_TESTS=1 -e HF_HUB_OFFLINE=1 \
  "$NPU_CI_IMAGE" bash -c '
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mkdir -p /results /tmp/omni-ci-source
    tar -C /checkout --exclude=.git --exclude=npu-ci-results --exclude=.venv \
      --exclude=__pycache__ -cf - . | tar -C /tmp/omni-ci-source -xf -
    cd /tmp/omni-ci-source
    cp pyproject_npu.toml pyproject.toml
    python -m pip install --no-deps --no-build-isolation -e .
    python -m pip install pytest==8.3.5 jiwer==4.0.0 rapidfuzz==3.14.1
    export PYTHONPATH=/tmp/omni-ci-source
    python -m pip freeze > /results/packages.txt
    npu-smi info > /results/npu-smi.txt
    cp /data/configs/$NPU_CI_MODEL.yaml /results/serving-config.yaml
    if [[ "$NPU_CI_MODEL" == qwen3-tts ]]; then
      export OMNI_NPU_TTS_MODEL=/data/models/qwen3-tts
      export OMNI_NPU_TTS_CONFIG=/data/configs/qwen3-tts.yaml
      export OMNI_NPU_TTS_TASK=CustomVoice OMNI_NPU_TTS_OUTPUT=/results/tts
      suite=tests/test_model/test_npu_tts.py
    else
      export OMNI_NPU_ASR_MODEL=/data/models/qwen3-asr
      export OMNI_NPU_ASR_CONFIG=/data/configs/qwen3-asr.yaml
      export OMNI_NPU_ASR_CASES=/data/asr/cases.json
      export OMNI_NPU_ASR_OUTPUT=/results/asr
      suite=tests/test_model/test_npu_asr.py
    fi
    python -m pytest "$suite" -v -s --junitxml=/results/junit.xml
  ' > "$run_dir/container-id.txt"
docker start -a "$container_name"
exit "$(docker inspect --format '{{.State.ExitCode}}' "$container_name")"
