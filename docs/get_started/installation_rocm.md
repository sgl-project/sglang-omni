# Installation — AMD ROCm

ROCm support targets `gfx942` (MI300X) and `gfx950` (MI355X) with ROCm 7.2,
PyTorch 2.9.1, and SGLang 0.5.16. Use the image matching the GPU architecture;
mixing the `mi30x` and `mi35x` stacks is unsupported.

## Build the pinned image

Until release images are published, build from the repository. The Dockerfile
pins UCX and NIXL source commits and installs an accelerator-neutral Omni
dependency manifest without replacing the ROCm Torch/SGLang stack.

For MI300X/gfx942:

```bash
docker build -f docker/rocm.Dockerfile \
  --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi30x@sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee \
  --build-arg GPU_ARCH=gfx942 \
  -t sglang-omni:rocm720-gfx942 .
```

For MI355X/gfx950:

```bash
docker build -f docker/rocm.Dockerfile \
  --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi35x@sha256:54ac680bad1832b8acd469533ae66f608b525cec3449bbd5f3d0238351e9b965 \
  --build-arg GPU_ARCH=gfx950 \
  -t sglang-omni:rocm720-gfx950 .
```

Record the resolved base-image and result-image digests in deployment or CI
artifacts. A mutable tag alone is not reproducible evidence.

## Run

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1:-1 \
  --security-opt seccomp=unconfined \
  -v "$PWD:/workspace/sglang-omni" \
  sglang-omni:rocm720-gfx950
```

Use `sglang-omni:rocm720-gfx942` on MI300X. Then verify the resolved stack:

```bash
./scripts/rocm/install_rocm.sh --check
sgl-omni check-gpu --strict
```

The report must identify `rocm`, the expected `gfx942` or `gfx950`
architecture, an importable AITER/Triton or Torch-SDPA path, GPU IPC, and NIXL.

## Install in an existing SGLang ROCm environment

Do not run the default `uv pip install -e .` dependency resolution in a ROCm
environment during the transition release: the main manifest still pins CUDA
packages. Build NIXL with its ROCm UCX plugin, then run:

```bash
./scripts/rocm/install_rocm.sh
```

The installer uses `pyproject_rocm.toml` for top-level Omni dependencies and
installs the checkout with `--no-deps`. It refuses a non-ROCm Torch build,
SGLang other than 0.5.16, unsupported visible GPU architectures, or a missing
NIXL Python API.

## Audar-TTS optional runtime

Audar-TTS uses llama.cpp's HIP backend plus NeuCodec. To include it in the
image, add `--build-arg INSTALL_AUDAR_TTS=1` to the architecture-matched build
above. For an existing environment, build llama.cpp for the local target and
then install NeuCodec without changing the pinned ROCm stack:

```bash
export CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942"  # gfx950 on MI355X
export FORCE_CMAKE=1
python3 -m pip install llama-cpp-python==0.3.34 neucodec==0.0.6
```

## Runtime policy

- ROCm TP child processes honor and normalize `ROCR_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`. Conflicting inherited masks
  fail before worker startup.
- NVIDIA-only CUTLASS, FlashInfer, and DeepGEMM selections fail during backend
  policy resolution. Leave backends on `auto` or select AITER/Triton explicitly.
- Owned device-graph paths remain eager until separately qualified on both
  targets. Torch compilation may remain enabled where the model support matrix
  records it.
- Same-node GPU edges use PyTorch's CUDA-compatible HIP IPC APIs. Remote ROCm
  edges default to NIXL over UCX built with `--with-rocm`.

See [ROCm model support](../developer_reference/rocm_support.md) before choosing
a model. `preview` is not a support claim.
