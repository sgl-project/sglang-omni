# syntax=docker/dockerfile:1.7

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:02a85f00438c901c72a2eb2ef738974a807f63af3d13084445604f3344067b19

FROM ${FLASHINFER_CACHE_IMAGE} AS flashinfer-cache
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
COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --break-system-packages --no-build-isolation \
        -r /tmp/pyproject.toml \
    && uv pip install --system --break-system-packages --no-deps qwen-tts==0.1.1 \
    && uv pip install --system --break-system-packages --no-deps --reinstall flashinfer-python==0.6.17 \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm /tmp/pyproject.toml

# Docker builds do not have a GPU to regenerate architecture-specific JIT
# artifacts. Reuse the Python 3.12 cache validated on Ada (SM89) and Hopper
# (SM90a); other architectures compile into their own cache directory.
COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.17 /root/.cache/flashinfer/0.6.17

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0

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
