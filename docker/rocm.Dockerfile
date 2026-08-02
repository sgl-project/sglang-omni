# syntax=docker/dockerfile:1
# Override BASE_IMAGE with the matching upstream mi30x/mi35x and ROCm 7.0/7.2
# tag when targeting a different accelerator generation.
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
FROM ${BASE_IMAGE}

WORKDIR /sgl-workspace/sglang-omni
COPY requirements/rocm.txt requirements/rocm.txt
RUN python3 -m pip install --no-cache-dir -r requirements/rocm.txt

COPY . .
RUN python3 -m pip install --no-cache-dir --no-deps -e .

CMD ["bash"]
