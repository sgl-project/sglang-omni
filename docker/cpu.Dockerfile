# CPU image for sglang-omni.
#   docker build -f docker/cpu.Dockerfile -t sglang-omni:cpu .
#   docker run -it --shm-size 32g --ipc host --network host sglang-omni:cpu

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=v0.5.16

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    cmake \
    ninja-build \
    make \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl \
    sox

WORKDIR /opt

ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/ && \
    uv venv --python 3.12
ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

RUN uv pip install \
    --default-index https://pypi.org/simple \
    -U \
    "packaging>=24.2" \
    "setuptools>=77.0.0" \
    setuptools-scm \
    wheel \
    scikit-build-core

RUN uv pip install \
    --default-index https://download.pytorch.org/whl/cpu \
    "torch==2.12.0" \
    "torchvision==0.27.0" \
    "torchaudio==2.11.0"

WORKDIR /sgl-workspace
RUN source /opt/.venv/bin/activate && \
    git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . --no-build-isolation && \
    cd .. && \
    cd sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . --no-build-isolation
COPY . /sgl-workspace/sglang-omni

RUN cd /sgl-workspace/sglang-omni && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install -e . --no-build-isolation


RUN uv pip install --no-deps sox && \
    uv pip install --no-deps qwen-tts==0.1.1

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so:/opt/.venv/lib/libiomp5.so
ENV PATH="/opt/.venv/bin:$PATH"
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang-omni
