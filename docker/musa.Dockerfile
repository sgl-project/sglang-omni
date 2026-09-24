# syntax=docker/dockerfile:1.7

# SGLang-Omni MUSA image. Build SGLang's MUSA image first:
#   SGLang tag: v0.5.20
#   docker build -f docker/musa.Dockerfile -t sglang:v0.5.20-musa520-s5000 <sglang-repo>
# Then build this image:
#   docker build -f docker/musa.Dockerfile -t sglang-omni:main-musa520-s5000 .

ARG SGLANG_MUSA_IMAGE=sglang:v0.5.20-musa520-s5000

FROM ${SGLANG_MUSA_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV SGLANG_OMNI_REPO_DIR=/workspace/sglang-omni

ARG UBUNTU_APT_MIRROR=

RUN if [[ -n "${UBUNTU_APT_MIRROR}" ]]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|http://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g" /etc/apt/sources.list; \
    fi \
    && apt-get -o Acquire::Retries=3 -o Acquire::ForceIPv4=true update \
    && apt-get install -y --no-install-recommends \
        libdav1d-dev \
        libmp3lame-dev \
        libsox-dev \
        libsox-fmt-all \
        libssl-dev \
        libx264-dev \
        nasm \
        pybind11-dev \
        yasm \
    && true

COPY . ${SGLANG_OMNI_REPO_DIR}

WORKDIR ${SGLANG_OMNI_REPO_DIR}

ARG MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG FFMPEG_MUSA_REPO=https://github.com/MooreThreads/FFmpeg.git
ARG FFMPEG_MUSA_REF=mt-7.0.2-public
ARG TORCHCODEC_INSTALL_SPEC="torchcodec @ git+https://github.com/MooreThreads/torchcodec.git@release/0.5-musa-public"

RUN python3 -m pip install --upgrade pip "setuptools<82" wheel \
    && cp pyproject_musa.toml pyproject.toml \
    && git clone --depth 1 --branch "${FFMPEG_MUSA_REF}" \
        "${FFMPEG_MUSA_REPO}" /tmp/FFmpeg \
    && mkdir -p /tmp/FFmpeg/build \
    && cd /tmp/FFmpeg/build \
    && ../configure \
        --enable-shared \
        --enable-libmp3lame \
        --enable-libdav1d \
        --enable-openssl \
        --enable-libx264 \
        --enable-gpl \
        --enable-nonfree \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && cd "${SGLANG_OMNI_REPO_DIR}" \
    && TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
        ENABLE_MUSA=1 \
        python3 -m pip install --no-build-isolation --no-deps "${TORCHCODEC_INSTALL_SPEC}" \
        --index-url "${PYPI_INDEX_URL}" \
        --extra-index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
    && python3 -m pip install --no-build-isolation -e . \
        --index-url "${PYPI_INDEX_URL}" \
        --extra-index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com

# Install qwen-tts without dependencies because it pins Transformers 4.57.3
# and accelerate 1.12.0, which would replace the inherited stack.
RUN python3 -m pip install --no-cache-dir --no-deps qwen-tts==0.1.1

RUN python3 - <<'PY'
import torch

assert getattr(torch.version, "musa", None), "the inherited PyTorch build is not MUSA-enabled"
assert hasattr(torch, "musa"), "torch.musa is unavailable"
import torchada  # noqa: F401
import triton
import triton.backends.mtgpu  # noqa: F401
import sglang  # noqa: F401
import sglang_omni  # noqa: F401
PY

CMD ["/bin/bash"]
