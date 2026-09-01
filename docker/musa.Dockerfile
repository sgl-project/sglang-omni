# syntax=docker/dockerfile:1.7

# SGLang-Omni MUSA image. Build SGLang's MUSA image first:
#   docker build -f docker/musa.Dockerfile -t sglang:main-musa520-s5000 <sglang-repo>
# Then build this image:
#   docker build -f docker/musa.Dockerfile -t sglang-omni:main-musa520-s5000 .

ARG SGLANG_MUSA_IMAGE=sglang:main-musa520-s5000

FROM ${SGLANG_MUSA_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV SGLANG_OMNI_REPO_DIR=/workspace/sglang-omni
ENV MATE_MUSA_ARCH_LIST=3.1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    TRITON_CACHE_DIR=/root/.triton/cache

ARG UBUNTU_APT_MIRROR=

RUN if [[ -n "${UBUNTU_APT_MIRROR}" ]]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|http://mirrors.aliyun.com/ubuntu|${UBUNTU_APT_MIRROR}|g" /etc/apt/sources.list; \
    fi \
    && apt-get -o Acquire::Retries=3 -o Acquire::ForceIPv4=true update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libdav1d-dev \
        libmp3lame-dev \
        libsndfile1 \
        libsox-dev \
        libsox-fmt-all \
        libssl-dev \
        libx264-dev \
        nasm \
        pybind11-dev \
        sox \
        vainfo \
        yasm \
    && true

COPY . ${SGLANG_OMNI_REPO_DIR}

WORKDIR ${SGLANG_OMNI_REPO_DIR}

ARG MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG FFMPEG_MUSA_REPO=https://github.com/MooreThreads/FFmpeg.git
ARG FFMPEG_MUSA_REF=mt-7.0.2-public
ARG TORCHCODEC_INSTALL_SPEC="torchcodec @ git+https://github.com/MooreThreads/torchcodec.git@release/0.5-musa-public"

COPY pyproject_musa.toml /tmp/pyproject_musa.toml
RUN python3 -m pip install --upgrade pip "setuptools<82" wheel \
    && pip_find_links=() \
    && pip_index_opts=(--index-url "${PYPI_INDEX_URL}" --extra-index-url "${MUSA_PIP_INDEX_URL}" --trusted-host dl.mthreads.com) \
    && if [[ -d third_party/musa_wheelhouse ]]; then \
        python3 -c 'from pathlib import Path; import zipfile; [wheel.unlink() for wheel in Path("third_party/musa_wheelhouse").glob("triton-*.whl") if not any(name.startswith("triton/backends/mtgpu/") for name in zipfile.ZipFile(wheel).namelist())]'; \
        pip_find_links=(--find-links third_party/musa_wheelhouse); \
        if [[ -f third_party/musa_wheelhouse/.omni-complete ]]; then \
            pip_index_opts=(--no-index); \
        fi; \
    fi \
    && cp pyproject.toml /tmp/pyproject_cuda.toml \
    && cp /tmp/pyproject_musa.toml pyproject.toml \
    && if [[ -f third_party/FFmpeg/configure ]]; then \
        ffmpeg_src="${SGLANG_OMNI_REPO_DIR}/third_party/FFmpeg"; \
    else \
        git clone --depth 1 --branch "${FFMPEG_MUSA_REF}" \
            "${FFMPEG_MUSA_REPO}" /tmp/FFmpeg; \
        ffmpeg_src="/tmp/FFmpeg"; \
    fi \
    && mkdir -p "${ffmpeg_src}/build" \
    && cd "${ffmpeg_src}/build" \
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
        python3 -m pip install --no-build-isolation "${TORCHCODEC_INSTALL_SPEC}" \
        "${pip_find_links[@]}" \
        --index-url "${PYPI_INDEX_URL}" \
        --extra-index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
    && python3 -m pip install --no-build-isolation -e . \
        "${pip_find_links[@]}" \
        "${pip_index_opts[@]}" \
    && cp /tmp/pyproject_cuda.toml pyproject.toml \
    && rm /tmp/pyproject_musa.toml /tmp/pyproject_cuda.toml

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
