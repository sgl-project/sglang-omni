# syntax=docker/dockerfile:1.7
#
# EasyMagpie TTS server: SGLang-Omni on the SGLang v0.5.20 release image, with
# the EasyMagpie SGLang fixes applied to the installed SGLang package.
#
# The model is not baked in. Mount a converted model directory at /model.

# lmsysorg/sglang:v0.5.20, the same digest upstream SGLang-Omni pins.
ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
FROM ${SGLANG_IMAGE}

ARG OMNI_REPO=https://github.com/yhayarannvidia/sglang-omni.git
ARG OMNI_REF=feat/easymagpie-tts
ARG SGLANG_VERSION=0.5.20

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN git init -q /opt/sglang-omni \
    && git -C /opt/sglang-omni remote add origin "${OMNI_REPO}" \
    && git -C /opt/sglang-omni fetch -q --depth 1 origin "${OMNI_REF}" \
    && git -C /opt/sglang-omni checkout -q --detach FETCH_HEAD

# Dependencies come from the checkout's pyproject and its uv overrides, without
# the extras other models need. The base image's stack lives in the /opt/sglang
# virtualenv that PATH resolves. A leftover flashinfer-cubin wheel breaks
# FlashInfer's import, so the base image's copy is removed.
WORKDIR /opt/sglang-omni
RUN python3 -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["tool"]["uv"]["override-dependencies"]))' > /tmp/omni-overrides.txt \
    && uv pip install --python /opt/sglang/bin/python3 --no-build-isolation \
        --overrides /tmp/omni-overrides.txt -r pyproject.toml \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps --no-build-isolation -e . \
    && (python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true) \
    && rm /tmp/omni-overrides.txt

# The base image installs SGLang from its v0.5.20 source tree. The patch is
# exact against that release, so refuse any other version instead of applying
# it with offsets.
COPY sglang-0.5.20-easymagpie.patch /tmp/sglang.patch
RUN installed="$(python3 -c 'import importlib.metadata as m; print(m.version("sglang"))')" \
    && test "${installed}" = "${SGLANG_VERSION}" \
        || { echo "expected sglang ${SGLANG_VERSION}, found ${installed}" >&2; exit 1; } \
    && site="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("sglang").origin))')" \
    && patch -d "${site}" -p1 --forward --no-backup-if-mismatch -F0 < /tmp/sglang.patch \
    && rm /tmp/sglang.patch

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=3600s \
    CMD curl -fsS -o /dev/null http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "-m", "sglang_omni.cli", "serve", \
    "--config", "examples/configs/easymagpie_tts.yaml", \
    "--model-path", "/model", "--port", "8000"]
CMD []
