# Intel XPU image for sglang-omni. Mirrors SGLang's docker/xpu.Dockerfile, and
# swaps in pyproject_xpu.toml so no CUDA-only wheels are pulled.
#   docker build -f docker/xpu.Dockerfile -t sglang-omni:xpu .
#   docker run -it --device /dev/dri --shm-size 32g sglang-omni:xpu

FROM intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04 AS base

ARG SGLANG_XPU_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_XPU_BRANCH=v0.5.16
# SGLang's XPU manifest requires sgl-kernel-xpu with no revision, so pinning SGLang
# alone leaves the SYCL kernels floating. Pinned to the revision this PR was built
# against; override only to move deliberately.
ARG SGL_KERNEL_XPU_REF=8c4f5e8c1f4ad10b3af9791e216295db3493c7d9

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV TORCH_XPU_INDEX=https://download.pytorch.org/whl/xpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

# Minors differ on purpose: the XPU channel ships no torchaudio 2.12+xpu.
RUN pip install --no-cache-dir --extra-index-url ${TORCH_XPU_INDEX} \
        torch==2.12.0+xpu \
        torchvision==0.27.0+xpu \
        torchaudio==2.11.0+xpu \
        torchcodec==0.12.0

# The grep guard fails the build if upstream reshapes that requirement line, since
# the sed would otherwise no-op and silently restore a floating kernel.
RUN git clone --branch ${SGLANG_XPU_BRANCH} --single-branch ${SGLANG_XPU_REPO} sglang \
    && cd sglang/python \
    && cp pyproject_xpu.toml pyproject.toml \
    && sed -i "s|\(sgl-kernel @ git+https://github.com/sgl-project/sgl-kernel-xpu.git\)\"|\1@${SGL_KERNEL_XPU_REF}\"|" pyproject.toml \
    && grep -q "sgl-kernel-xpu.git@${SGL_KERNEL_XPU_REF}\"" pyproject.toml \
    && pip install --no-cache-dir . --extra-index-url ${TORCH_XPU_INDEX}

# --no-build-isolation installs no build requirement, so setuptools is pinned here:
# below 77 it rejects the PEP 639 license metadata in pyproject_xpu.toml.
COPY . /workspace/sglang-omni
RUN cd /workspace/sglang-omni \
    && pip install --no-cache-dir -U 'setuptools>=77.0.0' \
    && cp pyproject_xpu.toml pyproject.toml \
    && pip install --no-cache-dir -e . --no-build-isolation --extra-index-url ${TORCH_XPU_INDEX}

# --no-deps: qwen-tts pins Transformers 4.57.3, which would replace the stack above,
# and resolving sox lifts numpy past the numba==0.65.1 ceiling.
RUN pip install --no-cache-dir --no-deps sox einops \
    && pip install --no-cache-dir --no-deps qwen-tts==0.1.1

WORKDIR /workspace/sglang-omni

# Do NOT source /opt/intel/oneapi/setvars.sh: the `+xpu` wheels ship their own
# oneCCL/SYCL/Level-Zero, and a system oneAPI on the library path conflicts with
# the bundled libccl (multi-XPU xccl collectives crash).
CMD ["/bin/bash"]
