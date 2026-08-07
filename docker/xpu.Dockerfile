# Intel XPU image for sglang-omni. Mirrors SGLang's docker/xpu.Dockerfile, and
# swaps in pyproject_xpu.toml so no CUDA-only wheels are pulled.
#   docker build -f docker/xpu.Dockerfile -t sglang-omni:xpu .
#   docker run -it --device /dev/dri --shm-size 32g sglang-omni:xpu

FROM intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04 AS base

ARG SGLANG_XPU_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_XPU_BRANCH=v0.5.16

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV TORCH_XPU_INDEX=https://download.pytorch.org/whl/xpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch XPU stack (pins match SGLang's xpu.Dockerfile). Minors differ on
# purpose: the XPU channel ships no torchaudio 2.12+xpu.
RUN pip install --no-cache-dir --extra-index-url ${TORCH_XPU_INDEX} \
        torch==2.12.0+xpu \
        torchvision==0.27.0+xpu \
        torchaudio==2.11.0+xpu \
        torchcodec==0.12.0

# SGLang XPU build: install from source (its own XPU pyproject + sgl-kernel-xpu)
# rather than the CUDA wheel.
RUN git clone --branch ${SGLANG_XPU_BRANCH} --single-branch ${SGLANG_XPU_REPO} sglang \
    && cd sglang/python \
    && cp pyproject_xpu.toml pyproject.toml \
    && pip install --no-cache-dir . --extra-index-url ${TORCH_XPU_INDEX}

# sglang-omni (XPU variant). --no-build-isolation uses this image's setuptools so
# pip writes a PEP 660 editable install (a legacy egg-info is not importable).
COPY . /workspace/sglang-omni
RUN cd /workspace/sglang-omni \
    && cp pyproject_xpu.toml pyproject.toml \
    && pip install --no-cache-dir -e . --no-build-isolation --extra-index-url ${TORCH_XPU_INDEX}

WORKDIR /workspace/sglang-omni

# Do NOT source /opt/intel/oneapi/setvars.sh: the `+xpu` wheels ship their own
# oneCCL/SYCL/Level-Zero, and a system oneAPI on the library path conflicts with
# the bundled libccl (multi-XPU xccl collectives crash).
CMD ["/bin/bash"]
