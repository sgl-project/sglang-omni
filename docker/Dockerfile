# syntax=docker/dockerfile:1.7

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df

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
# The base image installs its stack into the /opt/sglang virtualenv that PATH
# resolves. uv finds a virtualenv only through VIRTUAL_ENV or a .venv directory,
# so name the interpreter explicitly.
# Reuse the project's protobuf override for DAC, and include model extras so
# startup does not need to install packages into a different Python environment.
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c 'import tomllib; print("\n".join(tomllib.load(open("/tmp/pyproject.toml", "rb"))["tool"]["uv"]["override-dependencies"]))' > /tmp/omni-overrides.txt \
    && uv pip install --python /opt/sglang/bin/python3 --no-build-isolation \
        --overrides /tmp/omni-overrides.txt -r /tmp/pyproject.toml \
        descript-audiotools==0.7.2 descript-audio-codec==1.0.0 \
    && CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89;90;100;120' \
        CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install --no-cache --python /opt/sglang/bin/python3 \
        --overrides /tmp/omni-overrides.txt -r /tmp/pyproject.toml \
        --extra audar-tts --extra fun-cosyvoice3 \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps qwen-tts==0.1.1 \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps --reinstall flashinfer-python==0.6.18 \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm /tmp/pyproject.toml /tmp/omni-overrides.txt

# Reinstalling the same wheel changes header mtimes and invalidates Ninja's
# objects. Preserve donor mtimes only for byte-identical FlashInfer sources.
RUN --mount=type=bind,from=flashinfer-cache,source=/opt/sglang/lib/python3.12/site-packages/flashinfer/data,target=/tmp/flashinfer-data,ro <<'EOF'
python3 - <<'PY'
import filecmp
import os
from pathlib import Path

reference = Path("/tmp/flashinfer-data")
installed = Path("/opt/sglang/lib/python3.12/site-packages/flashinfer/data")
for source in reference.rglob("*"):
    target = installed / source.relative_to(reference)
    if source.is_file() and target.is_file() and filecmp.cmp(source, target, shallow=False):
        os.utime(target, ns=(target.stat().st_atime_ns, source.stat().st_mtime_ns))
PY
EOF

# Docker builds do not have a GPU to regenerate architecture-specific JIT
# artifacts. Reuse the Python 3.12 cache validated on Ada (SM89) and Hopper
# (SM90a); other architectures compile into their own cache directory.
COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.18 /root/.cache/flashinfer/0.6.18

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
