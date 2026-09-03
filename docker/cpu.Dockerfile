# syntax=docker/dockerfile:1.7

# CPU image for sglang-omni.
#   docker build -f docker/cpu.Dockerfile -t sglang-omni:cpu .
#   docker run -it --shm-size 32g --ipc host --network host sglang-omni:cpu

ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.18-xeon@sha256:6d62b6fa73e4ddc90b46cbbd0081c6e25e46ea3884c6113a1b9399f70008a5d9

FROM ${SGLANG_IMAGE} AS runtime
SHELL ["/bin/bash", "-c"]

# torchcodec loads FFmpeg's shared libraries at import time, and Omni serves
# audio-input models, so FFmpeg and libsndfile are runtime dependencies.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

COPY pyproject_cpu.toml /tmp/pyproject.toml
# The inherited uv config prioritizes the PyTorch CPU index, whose packaging
# versions lag PyPI. Prefer PyPI only for build tooling; project dependencies
# below continue to use the inherited CPU index configuration.
RUN uv pip install --upgrade \
        --index https://pypi.org/simple \
        "packaging>=24.2" \
        "setuptools>=77.0.0" && \
    uv pip install --no-build-isolation -r /tmp/pyproject.toml

COPY . /sgl-workspace/sglang-omni
WORKDIR /sgl-workspace/sglang-omni

RUN cp /tmp/pyproject.toml pyproject.toml && \
    uv pip install --no-deps --no-build-isolation -e . && \
    rm /tmp/pyproject.toml
