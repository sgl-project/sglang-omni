# syntax=docker/dockerfile:1.7

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e

FROM ${SGLANG_IMAGE} AS runtime

ARG UCX_COMMIT=d8e50df6651b9ea5b76f23aee0aefbf053a4137a

RUN apt-get update \
    && apt-get install -y --no-install-recommends sox \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/openucx/ucx.git /tmp/ucx \
    && git -C /tmp/ucx checkout "${UCX_COMMIT}" \
    && cd /tmp/ucx \
    && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=/usr/local/cuda \
        --with-verbs \
        --with-dm \
        --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip \
    && ldconfig \
    && rm -rf /tmp/ucx

# Install only project dependencies. The repository itself is fetched when the
# container starts, so the image never contains a stale sglang-omni checkout.
# The base image installs its stack into the /opt/sglang virtualenv that PATH
# resolves. uv finds a virtualenv only through VIRTUAL_ENV or a .venv directory,
# so name the interpreter explicitly.
# Reuse the project's protobuf override for DAC, and include model extras so
# startup does not need to install packages into a different Python environment.
# The base image's FlashInfer AOT packages (cubin, the jit-cache umbrella and its
# per-architecture providers) carry the pinned version, so they stay: every kernel
# they hold loads without a compile, and only shapes outside them compile into
# FLASHINFER_WORKSPACE_BASE. Uninstalling the umbrella alone leaves the providers
# behind as a package without a version and breaks the flashinfer import.
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c 'import tomllib; print("\n".join(tomllib.load(open("/tmp/pyproject.toml", "rb"))["tool"]["uv"]["override-dependencies"]))' > /tmp/omni-overrides.txt \
    && uv pip install --python /opt/sglang/bin/python3 --no-build-isolation \
        --overrides /tmp/omni-overrides.txt -r /tmp/pyproject.toml \
        descript-audiotools==0.7.2 descript-audio-codec==1.0.0 \
    && CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89;90;100;120' \
        CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install --no-cache --python /opt/sglang/bin/python3 \
        --overrides /tmp/omni-overrides.txt -r /tmp/pyproject.toml \
        --extra audar-tts --extra fun-cosyvoice3 --extra minicpm-o \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps qwen-tts==0.1.1 \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps --reinstall flashinfer-python==0.7.1 \
    && rm /tmp/pyproject.toml /tmp/omni-overrides.txt

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0

# llama.cpp must load the same NCCL as Torch, not the older system copy.
ENV LD_LIBRARY_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

RUN <<'EOF'
cat >/usr/local/bin/sglang-omni-entrypoint <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SGLANG_OMNI_REPO_DIR:-/workspace/sglang-omni}"

if [ "${SGLANG_OMNI_AUTO_CLONE:-1}" = "1" ]; then
    if [ -d "${repo_dir}/.git" ]; then
        git -C "${repo_dir}" pull --ff-only origin main
    elif [ -e "${repo_dir}" ]; then
        echo "Repository path exists but is not a git checkout: ${repo_dir}" >&2
        exit 1
    else
        git clone --depth 1 --branch main \
            https://github.com/sgl-project/sglang-omni.git "${repo_dir}"
    fi
    python3 -m pip install --no-deps --no-build-isolation -e "${repo_dir}"
    cd "${repo_dir}"
fi

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
SCRIPT
chmod 0755 /usr/local/bin/sglang-omni-entrypoint
EOF

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/sglang-omni-entrypoint"]
CMD ["/bin/zsh"]
