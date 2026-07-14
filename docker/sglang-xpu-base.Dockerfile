# syntax=docker/dockerfile:1

# Reproducible SGLang XPU base for the SGLang-Omni compatibility matrix.
# Installing PyTorch XPU first prevents pip from pulling a CUDA torch stack
# through compressed-tensors and leaving that stack in an immutable layer.
FROM intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04 AS xpu-base

ARG PYTHON_VERSION=3.12
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_COMMIT=5a15cde858ea09b77116212a39356f2fc51b8584
ARG SGL_KERNEL_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
ARG SGL_KERNEL_COMMIT=2fcb04492ae79da88b7c3b1aeb88fb18e30aeaf8

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_COMMIT}" \
      qofe.runtime="xpu"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clinfo \
        intel-gsc \
        intel-media-va-driver-non-free \
        intel-metrics-discovery \
        intel-ocloc \
        intel-opencl-icd \
        libmfx-gen1 \
        libva-glx2 \
        libvpl-tools \
        libvpl2 \
        libze-dev \
        libze-intel-gpu1 \
        libze1 \
        python3-dev \
        va-driver-all \
        vainfo \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv venv --python "${PYTHON_VERSION}" --seed "${VIRTUAL_ENV}"

WORKDIR /sgl-workspace

RUN --mount=type=cache,id=sglang-xpu-pip,target=/root/.cache/pip \
    pip install --retries 20 --timeout 120 \
        torch==2.11.0+xpu \
        torchao==0.9.0+xpu \
        torchvision==0.26.0+xpu \
        torchaudio==2.11.0+xpu \
        --index-url https://download.pytorch.org/whl/xpu

FROM xpu-base AS kernel-builder

# Three compile jobs is the tested safe setting on a 26 GiB host. Higher
# values caused host OOM while compiling the XPU kernels.
ARG BUILD_JOBS=3

RUN --mount=type=cache,id=sglang-xpu-pip,target=/root/.cache/pip \
    --mount=type=cache,id=sglang-xpu-kernel-build,target=/sgl-kernel-build \
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" MAX_JOBS="${BUILD_JOBS}" \
    && git clone --filter=blob:none --depth=1 --branch pt2.11 \
        "${SGL_KERNEL_REPO}" sgl-kernel-xpu \
    && cd sgl-kernel-xpu \
    && git checkout "${SGL_KERNEL_COMMIT}" \
    && pip wheel --no-deps --wheel-dir /wheelhouse \
        --config-settings=build-dir=/sgl-kernel-build . \
    && cd /sgl-workspace \
    && rm -rf sgl-kernel-xpu

FROM xpu-base AS runtime

COPY --from=kernel-builder /wheelhouse/ /tmp/sgl-kernel-wheels/

RUN pip install --no-deps /tmp/sgl-kernel-wheels/*.whl \
    && rm -rf /tmp/sgl-kernel-wheels

RUN --mount=type=cache,id=sglang-xpu-pip,target=/root/.cache/pip \
    git clone --filter=blob:none --depth=1 --branch v0.5.12.post1 \
        "${SGLANG_REPO}" sglang \
    && cd sglang \
    && git checkout "${SGLANG_COMMIT}" \
    && cd python \
    && cp pyproject_xpu.toml pyproject.toml \
    && sed -i \
        's#sgl-kernel @ git+https://github.com/sgl-project/sgl-kernel-xpu.git#sgl-kernel==0.11.0#' \
        pyproject.toml \
    && pip install --retries 20 --timeout 120 . \
        --extra-index-url https://download.pytorch.org/whl/xpu \
    && pip install --no-deps xgrammar==0.1.33 \
    && python -c "from importlib.metadata import version; import torch, sglang, sgl_kernel, xgrammar; print(torch.__version__, sglang.__version__, sgl_kernel.__version__, version('xgrammar'))" \
    && cd /sgl-workspace \
    && rm -rf sglang

CMD ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec bash"]
