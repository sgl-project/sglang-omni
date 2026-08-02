# syntax=docker/dockerfile:1
# TODO: Override BASE_IMAGE with the matching upstream mi30x/mi35x and ROCm 7.0/7.2
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
FROM ${BASE_IMAGE}

ARG UV_VERSION=0.11.17
WORKDIR /sgl-workspace/sglang-omni
COPY pyproject.toml pyproject.toml
RUN python3 -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv pip install --python "$(command -v python3)" --no-cache --group rocm

COPY . .
RUN uv pip install \
    --python "$(command -v python3)" \
    --no-cache \
    --no-deps \
    --editable .

CMD ["sleep", "infinity"]
