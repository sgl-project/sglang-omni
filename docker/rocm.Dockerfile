# syntax=docker/dockerfile:1.7

# Supply an architecture-compatible, digest-pinned SGLang ROCm base image:
#   gfx942: lmsysorg/sglang:v0.5.16-rocm720-mi30x@sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee
#   gfx950: lmsysorg/sglang:v0.5.16-rocm720-mi35x@sha256:54ac680bad1832b8acd469533ae66f608b525cec3449bbd5f3d0238351e9b965
ARG SGLANG_IMAGE
FROM ${SGLANG_IMAGE}

ARG UCX_REF=8a6b06fb880accbb933a79cda893883872c68d9d
ARG NIXL_REF=c0a1102b94d173049a5478c23e765ba37681e2ca
ARG NIXL_PREFIX=/usr/local/nixl
ARG GPU_ARCH
ARG INSTALL_AUDAR_TTS=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf automake build-essential git libasio-dev libibverbs-dev \
        libnl-3-dev libnl-route-3-dev libnuma-dev librdmacm-dev libtool \
        ffmpeg pkg-config python3-dev python3-pip sox \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        meson==1.8.3 meson-python==0.18.0 ninja==1.13.0 \
        patchelf==0.17.2.2 pybind11==3.0.1 tomli==2.4.0 \
        tomlkit==0.13.3 uv==0.8.24 \
    && rm -rf /var/lib/apt/lists/*

# NIXL's ROCm UCX plugin requires UCX itself to be built with ROCm memory support.
RUN git clone --filter=blob:none https://github.com/openucx/ucx.git /tmp/ucx \
    && git -C /tmp/ucx checkout "${UCX_REF}" \
    && cd /tmp/ucx \
    && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --with-rocm=/opt/rocm \
        --with-verbs \
        --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip \
    && ldconfig \
    && rm -rf /tmp/ucx

RUN git clone --filter=blob:none https://github.com/ai-dynamo/nixl.git /tmp/nixl \
    && python3 -m pip uninstall -y nixl nixl-cu12 nixl-cu13 \
    && git -C /tmp/nixl checkout "${NIXL_REF}" \
    && meson setup /tmp/nixl/build /tmp/nixl \
        --prefix="${NIXL_PREFIX}" \
        --buildtype=release \
        -Dbuild_docs=false \
        -Dbuild_examples=false \
        -Dbuild_tests=false \
        -Dinstall_headers=false \
        -Denable_plugins=UCX \
        -Ducx_path=/usr/local \
        -Dwheel_variant=rocm \
    && meson compile -C /tmp/nixl/build \
    && meson install -C /tmp/nixl/build \
    && echo "${NIXL_PREFIX}/lib/x86_64-linux-gnu" \
        > /etc/ld.so.conf.d/nixl.conf \
    && echo "${NIXL_PREFIX}/lib/x86_64-linux-gnu/plugins" \
        >> /etc/ld.so.conf.d/nixl.conf \
    && ldconfig \
    && nixl_python_site="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && if [ -e "${nixl_python_site}/nixl" ] \
        || [ -L "${nixl_python_site}/nixl" ]; then \
        rm -rf "${nixl_python_site}/nixl"; \
    fi \
    && ln -sT "${NIXL_PREFIX}/lib/python3/dist-packages/nixl_rocm" \
        "${nixl_python_site}/nixl" \
    && python3 -c "from pathlib import Path; from nixl import _api; assert Path(_api.__file__).resolve().is_relative_to(Path('${NIXL_PREFIX}'))" \
    && rm -rf /tmp/nixl

WORKDIR /workspace/sglang-omni
COPY pyproject_rocm.toml ./pyproject.toml
COPY README.md LICENSE ./
COPY sglang_omni ./sglang_omni
COPY sglang_omni_router ./sglang_omni_router
COPY scripts/rocm ./scripts/rocm

# The ROCm 7.2 Torch 2.9.1 image is not ABI-compatible with published
# TorchCodec wheels. Leave it absent so SGLang uses its decoder fallback.
RUN python3 -m pip install --no-deps --no-build-isolation -e . \
    && python3 scripts/rocm/install_dependencies.py --manifest pyproject.toml \
    && python3 -m pip uninstall -y torchcodec \
    && python3 -m pip install --no-deps qwen-tts==0.1.1 \
    && if [ "${INSTALL_AUDAR_TTS}" = "1" ]; then \
        CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=${GPU_ARCH}" \
        FORCE_CMAKE=1 python3 -m pip install llama-cpp-python==0.3.34 \
        && python3 -m pip install --no-deps \
            neucodec==0.0.6 torchtune==0.6.1 torchdata==0.11.0 \
            vector-quantize-pytorch==1.17.8 local_attention==1.11.2 \
            hyper-connections==0.4.11 \
        && python3 -c "from llama_cpp import Llama; from neucodec import NeuCodec"; \
    fi \
    && python3 -m compileall -q sglang_omni sglang_omni_router

ENV SGLANG_OMNI_ACCELERATOR=rocm \
    NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins

CMD ["/bin/bash"]
