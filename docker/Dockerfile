# syntax=docker/dockerfile:1.7

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101

FROM ${FLASHINFER_CACHE_IMAGE} AS flashinfer-cache
FROM ${SGLANG_IMAGE} AS runtime

ARG UCX_COMMIT=d8e50df6651b9ea5b76f23aee0aefbf053a4137a

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
    && uv pip install --system --break-system-packages --no-deps --reinstall flashinfer-python==0.6.14 \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm /tmp/pyproject.toml

# Docker builds do not have a GPU to regenerate architecture-specific JIT
# artifacts. Reuse the Python 3.12 cache validated on Ada (SM89) and Hopper
# (SM90a); other architectures compile into their own cache directory.
COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.14 /root/.cache/flashinfer/0.6.14

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
