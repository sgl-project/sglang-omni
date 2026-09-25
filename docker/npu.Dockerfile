# syntax=docker/dockerfile:1.7
# Base versions and digests are maintained in pyproject_npu.toml.
# The selected digest pins the complete Ascend dependency stack.
# Build from the repository root; source is installed from the build context.
# Pass SGLANG_IMAGE from scripts/npu/config.py --get base-image-a3 (or 910b).
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
