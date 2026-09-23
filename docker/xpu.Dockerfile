# Intel XPU image for sglang-omni. Mirrors SGLang's own XPU image, and swaps in
# pyproject_xpu.toml so no CUDA-only wheels are pulled. Keep this file in step with
# https://github.com/sgl-project/sglang/blame/main/docker/xpu.Dockerfile -- the blame
# view is the quickest way to see why a given pin below is there.
#   docker build -f docker/xpu.Dockerfile -t sglang-omni:xpu .
#   docker run -it --device /dev/dri --shm-size 32g sglang-omni:xpu

FROM intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 AS base

ARG SGLANG_XPU_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_XPU_BRANCH=v0.5.20

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV TORCH_XPU_INDEX=https://download.pytorch.org/whl/xpu
# files.pythonhosted.org occasionally throttles large wheels below pip's
# default 15s socket-read window; raise the timeout and retry budget so
# builds ride through transient CDN slowdowns instead of failing. 60s x
# 5 retries caps worst-case wait at 5 min per stalled file.
ENV PIP_DEFAULT_TIMEOUT=60
ENV PIP_RETRIES=5

# Level-Zero UMD + IGC, mirroring SGLang's image, which pins them because the rolling
# PPA once faulted libze on B580 (sgl-kernel-xpu#296). Keep in lockstep with the host
# xe KMD; override via --build-arg.
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG IGC_VERSION=2.34.4+21428
ARG GMM_VERSION=22.10.0

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 sox python3-dev build-essential python3.12-venv \
        software-properties-common curl ca-certificates

ARG RENDER_GID=110
RUN getent group render >/dev/null \
    || groupadd --system --gid ${RENDER_GID} render \
    || groupadd --system render

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    # Loader + media/metrics from the PPA; the GPU driver itself is pinned below.
    && apt-get install -y \
        libze1 libze-dev intel-metrics-discovery clinfo intel-gsc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 va-driver-all vainfo \
    && cd /tmp \
    && igc_url="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}" \
    && cr_url="https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}" \
    # IGC first: libze-intel-gpu1 / intel-opencl-icd depend on its exact version.
    && curl -fsSL -O "${igc_url}/intel-igc-core-2_${IGC_VERSION}_amd64.deb" \
    && curl -fsSL -O "${igc_url}/intel-igc-opencl-2_${IGC_VERSION}_amd64.deb" \
    && curl -fsSL -O "${cr_url}/libigdgmm12_${GMM_VERSION}_amd64.deb" \
    && curl -fsSL -O "${cr_url}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && curl -fsSL -O "${cr_url}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && curl -fsSL -O "${cr_url}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && apt-get install -y --allow-downgrades \
        ./intel-igc-core-2_${IGC_VERSION}_amd64.deb \
        ./intel-igc-opencl-2_${IGC_VERSION}_amd64.deb \
        ./libigdgmm12_${GMM_VERSION}_amd64.deb \
        ./libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb \
        ./intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb \
        ./intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb \
    && rm -f /tmp/*.deb \
    # Hold so a later apt upgrade cannot pull the rolling PPA version back over these.
    && apt-mark hold libze-intel-gpu1 intel-opencl-icd intel-ocloc libigdgmm12 \
        intel-igc-core-2 intel-igc-opencl-2 \
    && rm -rf /var/lib/apt/lists/*

# TorchCodec-XPU requires an FFmpeg build with VAAPI support. Device/driver
# initialization is checked at runtime because build stages do not expose /dev/dri.
RUN ffmpeg -hide_banner -hwaccels 2>&1 | grep -qx vaapi

# Minors differ on purpose: the XPU channel ships no torchaudio newer than 2.11+xpu.
RUN pip install --no-cache-dir --extra-index-url ${TORCH_XPU_INDEX} \
        torch==2.13.0+xpu \
        torchvision==0.28.0+xpu \
        torchaudio==2.11.0+xpu \
        torchcodec==0.15.0 \
        torchcodec-xpu==0.15.0

# SGLang's XPU manifest pins the SYCL kernel wheel itself. An isolated build would
# download torch again and compile Rust extensions this image never loads, so it
# builds against the torch installed above, without setuptools-rust.
RUN git clone --branch ${SGLANG_XPU_BRANCH} --single-branch ${SGLANG_XPU_REPO} sglang \
    && cd sglang/python \
    && cp pyproject_xpu.toml pyproject.toml \
    && pip install --no-cache-dir 'setuptools>=77.0.0' setuptools-scm wheel \
    && pip install --no-cache-dir . --no-build-isolation --extra-index-url ${TORCH_XPU_INDEX}

# --no-deps avoids installing NVIDIA Triton over the XPU Triton stack.
RUN pip install --no-cache-dir --no-deps xgrammar==0.1.33

# --no-build-isolation installs no build requirement, so setuptools is pinned here:
# below 77 it rejects the PEP 639 license metadata in pyproject_xpu.toml.
# note (Matrix Yao): --no-deps keeps openai-whisper from replacing triton-xpu.
COPY . /workspace/sglang-omni
RUN cd /workspace/sglang-omni \
    && pip install --no-cache-dir -U 'setuptools>=77.0.0' \
    && cp pyproject_xpu.toml pyproject.toml \
    && pip install --no-cache-dir -e . --no-build-isolation --extra-index-url ${TORCH_XPU_INDEX} \
    && pip install --no-cache-dir --no-deps openai-whisper==20250625

# --no-deps: qwen-tts pins Transformers 4.57.3, which would replace the stack above,
# and resolving sox lifts numpy past the numba==0.65.1 ceiling.
RUN pip install --no-cache-dir --no-deps sox einops \
    && pip install --no-cache-dir --no-deps qwen-tts==0.1.1

WORKDIR /workspace/sglang-omni

# Do NOT source /opt/intel/oneapi/setvars.sh: the `+xpu` wheels ship their own
# oneCCL/SYCL/Level-Zero, and a system oneAPI on the library path conflicts with
# the bundled libccl (multi-XPU xccl collectives crash).
CMD ["/bin/bash"]
