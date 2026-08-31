# syntax=docker/dockerfile:1.7

# Build on the validated SGLang ROCm stack without replacing GPU-coupled components.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.18-rocm720-mi35x@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72

FROM ${SGLANG_IMAGE} AS runtime

ARG OMNI_REPO=https://github.com/sgl-project/sglang-omni.git
ARG OMNI_REF=main

ENV PATH=/opt/ucx/bin:${PATH} \
    SGLANG_OMNI_REPO_DIR=/workspace/sglang-omni

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies in a separate cacheable layer.
RUN python3 -m pip install --no-cache-dir uv

COPY pyproject_rocm.toml /tmp/pyproject.toml
RUN uv pip install --no-build-isolation \
        -r /tmp/pyproject.toml

RUN git clone "${OMNI_REPO}" "${SGLANG_OMNI_REPO_DIR}" \
    && git -C "${SGLANG_OMNI_REPO_DIR}" checkout --detach "${OMNI_REF}"

WORKDIR ${SGLANG_OMNI_REPO_DIR}

# Keep ROCm project metadata at the source root for consistent editable installs.
RUN cp /tmp/pyproject.toml pyproject.toml \
    && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && rm /tmp/pyproject.toml

RUN python3 - <<'PY'
import torch

assert torch.version.hip, "the inherited PyTorch build is not ROCm-enabled"
import nixl  # noqa: F401
import sglang  # noqa: F401
import sglang_omni  # noqa: F401
PY

RUN ucx_info -v | grep -q -- '--with-rocm'

CMD ["/bin/bash"]
