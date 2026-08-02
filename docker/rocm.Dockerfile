# syntax=docker/dockerfile:1
# TODO: Override BASE_IMAGE with the matching upstream mi30x/mi35x and ROCm 7.0/7.2
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
FROM ${BASE_IMAGE}

WORKDIR /sgl-workspace/sglang-omni
COPY pyproject.toml pyproject.toml
RUN uv pip install --system --no-cache --group rocm

COPY . .
RUN uv pip install --system --no-cache --no-deps -e .

CMD ["sleep", "infinity"]
