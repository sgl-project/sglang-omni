# syntax=docker/dockerfile:1.7
# Ascend A3 runtime: SGLang 0.5.19, CANN 9.0.0, Python 3.11,
# torch/torch_npu 2.10.0, triton-ascend 3.2.1.dev20260530,
# NPU kernel release 20260826 (wheel version 2026.6.1).
# The digest pins the complete base dependency stack.
# Build from the repository root; source is installed from the build context.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.19-cann9.0.0-a3@sha256:55b9ca3b9f2bd67817054f51c55ff64603013f2b211535f77837ed47c3ae05d1
FROM ${SGLANG_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG FFMPEG_VERSION=7:4.4.2-0ubuntu0.22.04.1
ARG LIBSNDFILE_VERSION=1.0.31-2ubuntu0.2
ARG SOX_VERSION=14.4.2+git20190427-2+deb11u2ubuntu0.22.04.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=${FFMPEG_VERSION} \
        libsndfile1=${LIBSNDFILE_VERSION} \
        sox=${SOX_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Resolve no dependencies here: the lock includes additions to the pinned base.
# In particular, qwen-tts metadata pins a different Transformers version; Omni's
# compatibility layer supports the project's Transformers 5.12.1 instead.
COPY docker/requirements-npu.txt /tmp/requirements-npu.txt
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -r /tmp/requirements-npu.txt \
    && rm /tmp/requirements-npu.txt

COPY . /workspace/sglang-omni
WORKDIR /workspace/sglang-omni
RUN cp pyproject_npu.toml pyproject.toml \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && cd / \
    && python -c "import sglang_omni; import librosa; import soundfile" \
    && python -m pip freeze --all > /workspace/python-packages.txt \
    && dpkg-query -W > /workspace/system-packages.txt

# Qwen3-TTS audio input uses Omni's SoundFile fallback when TorchCodec is absent.
# Do not pull an incompatible CUDA-linked TorchCodec wheel into this CPU torch base.
ENTRYPOINT []
CMD ["/bin/bash"]
