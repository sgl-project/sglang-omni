# syntax=docker/dockerfile:1.7
# SGLANG_IMAGE selects a pinned A3 or 950 image or the locally built 910B base.
ARG SGLANG_IMAGE
FROM ${SGLANG_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple

COPY . /workspace/sglang-omni
WORKDIR /workspace/sglang-omni
RUN bash scripts/npu/install_npu.sh \
        --install-system-deps --with-qwen-tts --no-editable --skip-device-check \
    && python -m pip freeze --all > /workspace/python-packages.txt \
    && dpkg-query -W > /workspace/system-packages.txt

# Qwen3-TTS audio input uses Omni's SoundFile fallback when TorchCodec is absent.
# Do not pull an incompatible CUDA-linked TorchCodec wheel into this CPU torch base.
CMD ["/bin/bash"]
