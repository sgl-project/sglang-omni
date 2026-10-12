#!/usr/bin/env bash
# Build the EasyMagpie TTS image if needed, start the server, and wait until it
# answers /health.
#
#   ./serve.sh MODEL_DIR [--port 8000] [--gpu 0] [--name easymagpie-tts]
#              [--ref feat/easymagpie-tts] [--rebuild] [-- EXTRA_SERVE_ARGS...]
#
# MODEL_DIR is a converted model directory (config.json, model.safetensors,
# codec_native/, speaker_embeddings/). Arguments after "--" go to
# `sglang_omni.cli serve`, for example `-- --vocoder.factory.freeze_gc false`.
set -euo pipefail

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
omni_repo="${OMNI_REPO:-https://github.com/yhayarannvidia/sglang-omni.git}"
port=8000
gpu=0
name=easymagpie-tts
ref=feat/easymagpie-tts
rebuild=0
model_dir=""
serve_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --port) port="$2"; shift 2 ;;
        --gpu) gpu="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --ref) ref="$2"; shift 2 ;;
        --rebuild) rebuild=1; shift ;;
        -h|--help) usage 0 ;;
        --) shift; serve_args=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *)
            if [ -n "${model_dir}" ]; then
                echo "only one MODEL_DIR is allowed" >&2
                usage 2
            fi
            model_dir="$1"; shift ;;
    esac
done

if [ -z "${model_dir}" ]; then
    usage 2
fi
model_dir="$(cd "${model_dir}" && pwd)"
for required in config.json model.safetensors codec_native/config.json speaker_embeddings; do
    if [ ! -e "${model_dir}/${required}" ]; then
        echo "${model_dir} is not a converted EasyMagpie model: missing ${required}" >&2
        exit 1
    fi
done

if ! docker info >/dev/null 2>&1; then
    echo "cannot reach the Docker daemon as $(id -un); use sudo or join the docker group" >&2
    exit 1
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia \
    && ! command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit not found; install it so containers can use the GPU" >&2
    exit 1
fi

# Tag the image by commit so a new push to the branch builds a new image.
commit="$(git ls-remote "${omni_repo}" "${ref}" | awk 'NR == 1 {print $1}')"
if [ -z "${commit}" ]; then
    if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
        commit="${ref}"
    else
        echo "cannot resolve ${ref} in ${omni_repo}" >&2
        exit 1
    fi
fi
image="easymagpie-tts:${commit:0:12}"

if [ "${rebuild}" = 1 ] || ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "==> building ${image} (sglang-omni ${commit})"
    DOCKER_BUILDKIT=1 docker build \
        --build-arg "OMNI_REPO=${omni_repo}" \
        --build-arg "OMNI_REF=${commit}" \
        --tag "${image}" \
        "${here}"
else
    echo "==> using existing ${image}"
fi

existing="$(docker ps -aq --filter "name=^${name}\$")"
if [ -n "${existing}" ]; then
    if [ "$(docker inspect --format '{{index .Config.Labels "easymagpie-tts"}}' "${name}")" != "1" ]; then
        echo "a container named ${name} exists and was not started by this script; pick another --name" >&2
        exit 1
    fi
    echo "==> replacing container ${name}"
    docker rm -f "${name}" >/dev/null
fi

# Compiled kernels, Triton autotuning and CUDA graphs' Inductor artifacts live
# under /root/.cache. A named volume keeps them, so only the first start pays
# for compilation.
echo "==> starting ${name} on port ${port} (GPU ${gpu})"
docker run -d \
    --name "${name}" \
    --label easymagpie-tts=1 \
    --gpus "device=${gpu}" \
    --ipc host \
    --restart unless-stopped \
    -p "${port}:8000" \
    -v "${model_dir}:/model:ro" \
    -v easymagpie-tts-cache:/root/.cache \
    "${image}" "${serve_args[@]}" >/dev/null

echo "==> waiting for the server; the first start compiles kernels and can take 20+ minutes"
start=$(date +%s)
while true; do
    if curl -fsS -m 3 -o /dev/null "http://localhost:${port}/health" 2>/dev/null; then
        break
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "${name}" 2>/dev/null)" != "true" ]; then
        echo "server exited during startup; last log lines:" >&2
        docker logs --tail 40 "${name}" >&2
        exit 1
    fi
    sleep 10
done
echo "==> ready after $(( $(date +%s) - start )) s"

cat <<EOF

  curl -X POST http://localhost:${port}/v1/audio/speech \\
    -H "Content-Type: application/json" \\
    -d '{"input": "Hello from EasyMagpie.", "response_format": "wav"}' \\
    --output hello.wav

  Logs: docker logs -f ${name}    Stop: docker rm -f ${name}
EOF
