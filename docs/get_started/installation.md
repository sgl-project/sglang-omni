# Installation — CUDA

Install SGLang-Omni on Linux with an NVIDIA GPU. Docker is recommended.
The current stable release is [v0.2.0 on PyPI](https://pypi.org/project/sglang-omni/).

## Other platforms

Using a different device? Choose the installation guide for your hardware:
[Apple Silicon](./installation_apple_silicon.md),
[Ascend NPU](./installation_npu.md),
[Intel XPU](./installation_xpu.md),
[Intel CPU](./installation_cpu.md), and
[Moore Threads MUSA](./installation_musa.md).

## Install SGLang-Omni

Choose [Docker](#docker-recommended) for a prebuilt CUDA environment,
[pip / uv](#pip-uv) to install into your own environment, or
[From source](#from-source) to work on the code.

(option-a-docker-recommended)=
### Docker (recommended)

**Prerequisites.** Use a CUDA-compatible NVIDIA driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image includes UCX, flash-attn, SGLang, and CUDA.

**Pull the image.**

```bash
docker pull hongccc/sglang-omni:dev
```

````{note}
The `:dev` tag moves with `main`. For reproducible runs, use the pulled image's
repository digest in place of `hongccc/sglang-omni:dev` in both the pull and run commands:

```bash
docker image inspect --format '{{index .RepoDigests 0}}' hongccc/sglang-omni:dev
```
````

**Start a container.**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

**Install SGLang-Omni** inside the container.

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.2.0"
```

(option-b-manual-install)=
### pip / uv

**Prerequisites.** Use a CUDA environment with glibc 2.34 or newer and install
these build dependencies first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.14.1` and SGLang 0.5.23's `nvidia-cutlass-dsl` 4.8.0 pin.

**Prepare the environment.**

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate
```

**Install SGLang-Omni.**

```bash
uv pip install --prerelease=allow "sglang-omni==0.2.0"
```

To install the latest version on the index, run `uv pip install --prerelease=allow sglang-omni`.

(install-from-source)=
### From source

Use this path for development or unreleased changes. Start with the
[pip / uv prerequisites](#pip-uv).

**Get the source.**

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni
```

**Prepare the environment** inside the checkout, as in [pip / uv](#pip-uv).

**Install in editable mode.**

```bash
uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```

## Next steps

[Start the Qwen3-ASR server, send a request, and check the transcript](apiserver_quickstart.md#install-and-start).
For model-specific configuration and other request formats, see the
[Qwen3-ASR Cookbook](../cookbook/qwen3_asr.md).

## Troubleshooting

See [Troubleshooting](./troubleshooting.md) for installation errors. Compare
`uv pip check` output against the [expected CUDA dependency messages](./troubleshooting.md#expected-output-of-uv-pip-check).
