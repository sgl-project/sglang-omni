#!/usr/bin/env bash
# Provision a fresh CUDA 13.0 host into the Higgs TTS inference base image.
#
# Run this ON the machine you intend to snapshot. It reproduces the layout the
# background service expects (see background/Dockerfile's
# INSTALL_HIGGS_DEPENDENCIES / HIGGS_SOURCE_DIR contract):
#
#   /root/miniconda3              the one interpreter every worker uses
#   /root/sglang-omni-fork        this fork, installed editable
#   /root/models/higgs-tts-3-4b   weights, fetched here rather than copied in
#   /etc/profile.d/higgs-runtime.sh   CUDA/PATH/LD_LIBRARY_PATH for JIT paths
#   /root/logs /root/pids         directories the service writes to
#
# Requirements: driver >= 580 (CUDA 13.0). torch is built for cu130; an older
# driver leaves torch.cuda.is_available() False and the pipeline cannot start.
#
# Dependencies are deliberately NOT installed from pyproject.toml's full
# dependency list: that list carries every model family sglang-omni supports
# (dots.tts, Ming-Omni, ZONOS2, gradio, whisper, nemo/pynini...), none of
# which the Higgs TTS pipeline imports, and some of which are painful to
# build. sglang is installed first and left to pin the numeric stack
# (torch / transformers / flashinfer[cu13]); the extras below are what
# higgs_tts itself needs.
#
# Usage:
#   SOURCE_ARCHIVE=/root/fork_deploy.tar.gz bash provision_higgs_base.sh
#   # or, if the source is already at /root/sglang-omni-fork, omit it.
set -euo pipefail

PY=/root/miniconda3/bin/python
SOURCE_DIR=${HIGGS_SOURCE_DIR:-/root/sglang-omni-fork}
MODEL_DIR=${HIGGS_MODEL_PATH:-/root/models/higgs-tts-3-4b}
MODEL_REPO=${HIGGS_MODEL_REPO:-bosonai/higgs-audio-v3-tts-4b}
REF_ASR_DIR=${HIGGS_REF_ASR_MODEL_PATH:-/root/models/faster-whisper-small}
REF_ASR_REPO=${HIGGS_REF_ASR_REPO:-Systran/faster-whisper-small}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}

echo "== 0. preflight =="
"$PY" -c "
import torch, sys
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'available', torch.cuda.is_available())
" || true
nvidia-smi --query-gpu=driver_version --format=csv,noheader

echo "== 1. directories =="
mkdir -p /root/models /root/logs /root/pids

echo "== 2. runtime environment =="
cat >/etc/profile.d/higgs-runtime.sh <<'PROFILE'
# CUDA 13.0 ships in the base image; flashinfer and torch.compile shell out to
# nvcc at runtime, and Miniconda must come first on PATH so subprocesses use
# the same interpreter as the worker that spawned them.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/root/miniconda3/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${CUDA_HOME}/targets/x86_64-linux/lib:/root/miniconda3/lib:${LD_LIBRARY_PATH:-}
export HIGGS_MODEL_PATH=/root/models/higgs-tts-3-4b
PROFILE

echo "== 2b. system packages =="
# ffmpeg is a *binary* dependency, and nothing in pip's world provides it.
# background's _transcode_to_mp3 calls ffmpeg-python, which shells out to
# /usr/bin/ffmpeg; the earlier version of this script installed the
# ffmpeg-python wrapper and stopped there, so every job died at the transcode
# step with "[Errno 2] No such file or directory: 'ffmpeg'" after generating
# audio successfully. The version here is the one the previous working image
# carried (7:4.4.2-0ubuntu0.22.04.1 on jammy); apt pulls the ~45-package
# libav*/libsndfile/libasound dependency tree with it.
#
# ninja-build is the other non-CUDA package that image had and a bare CUDA
# 13.0 host does not: flashinfer and torch.compile invoke it to build kernels.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg ninja-build
command -v ffmpeg >/dev/null || { echo "ffmpeg still missing after apt" >&2; exit 1; }
ffmpeg -version | head -1

echo "== 3. dependencies =="
"$PY" -m pip install --no-cache-dir "sglang==${SGLANG_VERSION}"
# setuptools<81: pyworld still imports pkg_resources at module load, which
# setuptools removed in 81. Without the pin the voice-fusion path dies at
# import time inside the tts_engine process -- and only there, so plain TTS
# keeps working and the breakage presents as a fusion bug.
"$PY" -m pip install --no-cache-dir \
    "pyworld>=0.3.4" "setuptools<81" "soundfile>=0.12.0" "scipy>=1.10.0" \
    msgspec xxhash librosa huggingface_hub

echo "== 4. fork source =="
if [ -n "${SOURCE_ARCHIVE:-}" ]; then
    rm -rf "${SOURCE_DIR}"
    mkdir -p "${SOURCE_DIR}"
    tar xzf "${SOURCE_ARCHIVE}" -C "${SOURCE_DIR}"
fi
test -d "${SOURCE_DIR}" || { echo "no source at ${SOURCE_DIR}" >&2; exit 1; }
# --no-deps: step 3 already resolved the stack. A plain `pip install -e` here
# would drag in the full multi-model dependency list described above.
"$PY" -m pip install --no-cache-dir --no-deps -e "${SOURCE_DIR}"

echo "== 5. weights =="
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
    HF_HUB_DISABLE_XET=1 \
    "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='${MODEL_REPO}',
                        local_dir='${MODEL_DIR}', max_workers=4))
"
else
    echo "weights already present at ${MODEL_DIR}"
fi

echo "== 5b. reference-ASR weights =="
# background/tasks/higgs_reference_asr.py transcribes clone references so the
# actor can pass the reference's own text along with its audio. Its docstring
# states the expectation plainly -- "image prep bakes the model in" -- and
# HIGGS_REF_ASR_MODEL_PATH defaults to this directory. It has an
# hf-mirror fallback, but that fallback hands a local path to
# huggingface_hub as a repo id and dies on HFValidationError, so a missing
# directory silently degrades every clone request to no reference text
# rather than failing loudly.
if [ ! -f "${REF_ASR_DIR}/model.bin" ]; then
    HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
    HF_HUB_DISABLE_XET=1 \
    "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='${REF_ASR_REPO}',
                        local_dir='${REF_ASR_DIR}', max_workers=4))
"
else
    echo "reference-ASR weights already present at ${REF_ASR_DIR}"
fi

echo "== 6. background service dependencies =="
# The deployment runs background/main.py straight out of this image, so the
# service's own dependencies have to be here -- without them it dies at import
# on python-dotenv before any Higgs code runs. They used to arrive from
# background/Dockerfile's `pip install -e .`.
#
# Deliberately NOT background's whole dependency list. That list also carries
# its other worker types (aider-chat, playwright, codewithgpu, volcengine),
# and aider-chat in particular pins an older universe -- huggingface-hub 0.31,
# numpy 1.26, openai 1.x -- which pip will happily install over this image's
# stack, leaving transformers unimportable and the pipeline dead. The Higgs
# worker imports none of them.
#
# But note how this list was originally built and what that cost: it was
# reasoned out from what the Higgs worker "obviously" needs, and it was wrong
# twice in opposite directions -- first too broad (installing background's
# whole list walked numpy and cryptography backwards), then too narrow
# (av and ffmpeg went missing, and production found both). The correct
# procedure is not to reason about it at all: boot the previous working
# image, capture `dpkg-query -W`, `pip freeze` and `conda list`, diff against
# the new host, and install the delta minus what the CUDA version forces to
# differ. Step 6a below is that delta, measured rather than guessed.
#
# The constraints cover every package the two sides share, not just torch:
# a narrower list still lets pip walk numpy and huggingface-hub backwards.
cat >/root/higgs-stack-constraints.txt <<'CONSTRAINTS'
torch==2.13.0+cu130
torchvision==0.28.0
sglang==0.5.18
transformers==5.12.1
flashinfer-python==0.6.17
numpy>=2.1
huggingface-hub>=0.36.0
openai==2.6.1
CONSTRAINTS
"$PY" -m pip install --no-cache-dir -c /root/higgs-stack-constraints.txt     "python-dotenv>=1.0.1,<2.0.0"     "dramatiq[redis,watch]>=2.0.0,<3.0.0"     "redis>=5.2.1,<6.0.0"     "oss2>=2.19.1,<3.0.0"     "opentelemetry-sdk>=1.30.0,<2.0.0"     "opentelemetry-api>=1.30.0,<2.0.0"     "opentelemetry-exporter-otlp-proto-grpc>=1.30.0,<2.0.0"     "opentelemetry-exporter-otlp-proto-http>=1.30.0,<2.0.0"     "opentelemetry-instrumentation-httpx>=0.51b0,<1.0.0"     "opentelemetry-instrumentation-requests>=0.51b0,<1.0.0"     "ffmpeg-python>=0.2.0,<0.3.0"     "mutagen>=1.47.0,<2.0.0"     "httpx[socks]>=0.28.1,<0.29.0"     "pyyaml>=6.0.2,<7.0.0"     "pillow>=11.2.1,<12.0.0"     "json-repair>=0.47.7"     "websockets>=14.1.0,<15.0.0"

echo "== 6a. packages the previous working image had and a bare host does not =="
# Measured, not reasoned: this is `pip freeze` on the last image that served
# traffic minus `pip freeze` here. Installed against a constraints file built
# from what is already present, so the step can only add packages -- it cannot
# walk an existing version backwards, which is what broke two earlier images.
#
#   av              reference decoding. HTTP(S)-URL references are the only
#                   shape the gateway sends, and non-wav containers reach
#                   torchaudio's av backend; without it every clone request
#                   failed with "No module named 'av'".
#   pyOpenSSL       the old image carried it alongside conda's cryptography
#                   42.0.5 with no conflict. The "module 'openssl' has no
#                   attribute 'ciphers'" outage came from pip pulling
#                   cryptography 44 over that 42, not from pyOpenSSL itself,
#                   so pin it to the version that coexisted and let the
#                   constraints hold cryptography where it is.
#   ctranslate2 /   the ASR workers that share this image. Not used by Higgs,
#   faster-whisper  but the image is copied wholesale to pods that run them.
#   onnxruntime
#
# Deliberately excluded from the delta: torchao and flashinfer-cubin (both
# built against the old image's CUDA 11.8 torch, which is the reason for this
# rebuild), and conda's own telemetry packages.
"$PY" -m pip freeze | grep -E '^[A-Za-z0-9._-]+==' > /root/higgs-installed-pins.txt
"$PY" -m pip install --no-cache-dir -c /root/higgs-installed-pins.txt \
    "av==18.1.0" \
    "pyOpenSSL==24.3.0" \
    "ctranslate2==4.8.1" \
    "faster-whisper==1.2.1" \
    "onnxruntime==1.29.0" \
    "flatbuffers==25.12.19" \
    "Cython==3.3.0" \
    "uv==0.12.5" \
    "volcengine-python-sdk==4.0.11" \
    "codewithgpu==0.2.8" \
    brotlicffi frozendict ruamel.yaml.clib pickleshare

echo "== 6b. drop conda extensions that pip's replacements cannot shadow =="
# A conda-installed package leaves a versioned extension
# (_rust.cpython-312-x86_64-linux-gnu.so) that takes import precedence over
# the abi3 .so pip ships. pip upgrading the package replaces the Python files
# and its own .so but never deletes conda's, so the new Python code ends up
# calling an old binary: cryptography 44 against a 2024 _rust gives
# "module 'openssl' has no attribute 'ciphers'", which surfaces far away as
# redis -> jwt failing to import, and only on hosts where that stale file
# exists.
"$PY" - <<'PY'
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"])
moved = 0
for abi3 in sp.rglob("*.abi3.so"):
    for rival in abi3.parent.glob(abi3.name.split(".")[0] + ".cpython-*.so"):
        backup = rival.with_suffix(rival.suffix + ".shadowed-by-abi3")
        print(f"  shadowing {rival.name} -> {backup.name}")
        rival.rename(backup)
        moved += 1
print(f"moved {moved} shadowed extension(s)")
PY

echo "== 7. verify =="
"$PY" -c "
import numpy, sglang, sglang_omni, torch, pyworld
import sglang_omni.models.higgs_tts.stages          # noqa: F401
import sglang_omni.models.higgs_tts.fusion_reference  # noqa: F401
# the service side must import too, in the same interpreter
import dotenv, dramatiq, redis, oss2, opentelemetry.sdk  # noqa: F401
import mutagen, ffmpeg, httpx, pydantic, yaml           # noqa: F401
print('numpy', numpy.__version__)
print('sglang_omni', sglang_omni.__version__, '|', sglang_omni.__file__)
print('sglang', sglang.__version__, '| torch', torch.__version__,
      '| cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'torch cannot see the GPU'
"
echo "PROVISION_DONE -- run docker/higgs_image_acceptance.py before snapshotting"
