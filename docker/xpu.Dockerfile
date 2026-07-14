# syntax=docker/dockerfile:1

ARG SGLANG_XPU_IMAGE=sglang-xpu:0.5.12.post1-pt2.11
FROM ${SGLANG_XPU_IMAGE}

ARG SGLANG_OMNI_COMMIT=1618b7ce26f5ea464e849cfcd943c0e429a711a6

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang-omni" \
      org.opencontainers.image.revision="${SGLANG_OMNI_COMMIT}" \
      qofe.runtime="xpu" \
      qofe.workflow="moss-transcribe-diarize"

ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

WORKDIR /opt/sglang-omni-src

COPY pyproject_xpu.toml pyproject.toml
COPY README.md LICENSE ./
COPY sglang_omni ./sglang_omni

RUN --mount=type=cache,id=sglang-omni-xpu-pip,target=/root/.cache/pip \
    pip install --retries 20 --timeout 120 . \
    # Docker builds do not expose /dev/dri, so importing the scheduler here
    # makes SGLang mis-detect the platform and request CUDA-only kvcacheio.
    # The device-aware scheduler import is exercised after the image starts.
    && python -c "import torch, sglang, sglang_omni; print(torch.__version__, sglang.__version__)" \
    && python -m py_compile sglang_omni/models/moss_transcribe_diarize/stages.py \
    && rm -rf /opt/sglang-omni-src

COPY scripts/serve_moss_xpu.sh /usr/local/bin/serve-moss-xpu

WORKDIR /workspace
CMD ["serve-moss-xpu"]
