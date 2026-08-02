# Installation — AMD ROCm

AMD support initially targets Qwen3-ASR, Qwen3-TTS, and Qwen3-Omni on AMD
Instinct MI300/MI325 (`gfx942`) and MI350 (`gfx950`) GPUs.

ROCm PyTorch exposes AMD accelerators through `torch.cuda` and uses `cuda:N`
device strings. Do not rewrite pipeline configurations to use a `rocm` device.

## Container installation

The image inherits the matching, pinned SGLang ROCm runtime, including its
PyTorch, Triton, Aiter, and RCCL builds:

```bash
docker build -f docker/rocm.Dockerfile -t sglang-omni:rocm .
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 32g \
  sglang-omni:rocm
```

The default is ROCm 7.2 for MI300-class hardware. Select another published
SGLang image when required:

```bash
docker build -f docker/rocm.Dockerfile \
  --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-rocm720-mi35x \
  -t sglang-omni:rocm-mi35x .
```

## Existing SGLang ROCm environment

The normal project dependencies contain NVIDIA-only packages. The helper
installs the scoped `rocm` dependency group from `pyproject.toml` and then
installs Omni with `--no-deps`, so it cannot replace the working
PyTorch/SGLang ROCm stack. The Dockerfile installs a pinned `uv`; install `uv`
first when using the helper in another environment:

```bash
PYTHON=python3 scripts/rocm/install_rocm.sh
```

## Verification

Run the model-free primitive probe and platform-aware diagnostic first:

```bash
python3 scripts/rocm/verify_rocm.py
sgl-omni check-gpu --strict
```

The probe verifies the ROCm PyTorch build, the pinned SGLang version, GPU
visibility, a synchronized GPU matrix multiplication, and the Omni import. Each
model enablement adds its own correctness recipe and hardware test rather than
treating this environment check as proof of model correctness.

For multi-GPU serving, prefer `ROCR_VISIBLE_DEVICES` on Linux. Avoid setting
conflicting `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and
`CUDA_VISIBLE_DEVICES` masks.
