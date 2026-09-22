# 🚀 Installation — Ascend NPU

## Prerequisites

Both installation methods require Ascend host drivers and firmware compatible
with your hardware and selected CANN release. Follow the
[HDK driver and firmware installation guide](https://www.hiascend.com/hardware/firmware-drivers/community)
and the [Ascend software compatibility guide](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu)
to select and install matching versions. For Docker, match the CANN version in
the image tag.

Verify the host installation before continuing:

```bash
npu-smi info
```

## 🐳 Option A: Docker (recommended)

The ARM64 image includes SGLang-Omni source, the SGLang/CANN/PyTorch runtime
selected by [`pyproject_npu.toml`](../../pyproject_npu.toml), and common audio
dependencies. Model weights, host drivers and TorchCodec are not included.
Model support on NPU and optional dependencies must be checked separately.

### A3 image installation

Use an ARM64 Linux host with Docker and Ascend A3 devices.

**1. Pull the image**

Pull the image matching your hardware from
[Docker Hub](https://hub.docker.com/r/lmsysorg/sglang-omni/tags):

- **Release:** use `v<version>-cann9.0.0-a3` for deployments, replacing
  `v<version>` with a published version tag from Docker Hub.
- **Development:** use `main-cann9.0.0-a3` for the latest published development
  build. Scheduled or manual publications update this rolling tag.

```bash
# Release (replace v<version> with a published version):
# IMAGE=lmsysorg/sglang-omni:v<version>-cann9.0.0-a3

# Development (A3):
IMAGE=lmsysorg/sglang-omni:main-cann9.0.0-a3

docker pull "$IMAGE"
```

For reproducible deployments, set `IMAGE` to
`lmsysorg/sglang-omni@sha256:<digest>`.

**2. Start the container**

Download your model weights into a host directory and mount it into the
container. The example exposes device 0; adjust device IDs and driver paths
for your host. `--privileged` grants broad host access and is only suitable for
a trusted host; production deployments should use restricted device access.

```bash
MODEL_DIR=<your_own_path>
docker run --rm -it --name omni-npu \
  --privileged --network host --shm-size 8g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_DIR:/models:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=8 \
  "$IMAGE" bash

# Inside the container:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

When selecting another device, update both `/dev/davinci0` and
`ASCEND_RT_VISIBLE_DEVICES`. Device indices in the pipeline configuration are
relative to the visible devices (`gpu: 0` selects the first visible device).

<details>
<summary>A2 image installation details (click)</summary>

Use an ARM64 Linux host with Docker and Ascend A2 / 910B devices.

**1. Pull the image**

Use `v<version>-cann9.0.0-910b` for a release, or
`main-cann9.0.0-910b` for the latest published development build.

```bash
# Release (replace v<version> with a published version):
# IMAGE=lmsysorg/sglang-omni:v<version>-cann9.0.0-910b

# Development (A2 / 910B):
IMAGE=lmsysorg/sglang-omni:main-cann9.0.0-910b

docker pull "$IMAGE"
```

For reproducible deployments, set `IMAGE` to
`lmsysorg/sglang-omni@sha256:<digest>`.

**2. Start the container**

Set `MODEL_DIR` to your model directory. The example exposes device 0;
adjust device IDs and driver paths for your host. Use `--privileged` only
on a trusted host; production deployments should use restricted device access.

```bash
MODEL_DIR=<your_own_path>
docker run --rm -it --name omni-npu \
  --privileged --network host --shm-size 8g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_DIR:/models:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=8 \
  "$IMAGE" bash

# Inside the container:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

When selecting another device, update both `/dev/davinci0` and
`ASCEND_RT_VISIBLE_DEVICES`. Pipeline device indices are relative to the
visible devices (`gpu: 0` selects the first visible device).

</details>

### Launch the server

The runtime is already installed. Skip the source installation below and do
not run the TorchCodec stack-upgrade script inside this image.
Qwen3-TTS dependencies are included; skip its cookbook's dependency installation.
Continue with the [API Server Quickstart](apiserver_quickstart.md) to launch
`sgl-omni serve` and send requests. Use the container model path under `/models`
and pass the selected model's NPU pipeline configuration with `--config` when
required. Model-specific inputs belong in the corresponding cookbook, such as
the [Qwen3-TTS guide](../cookbook/qwen3_tts.md); GPU-specific configurations
there should not be assumed to work unchanged on NPU.

## 🛠️ Option B: Manual install

Install the Ascend software stack and NPU build of SGLang before installing
`sglang-omni`. The helper script
[`install_npu.sh`](../../scripts/npu/install_npu.sh) installs only
`sglang-omni` and the Python dependencies declared in
[`pyproject_npu.toml`](../../pyproject_npu.toml), using the
same installation flow as Docker. Prepare
FFmpeg, libsndfile and SoX separately, or pass `--install-system-deps` as root.
The Ascend software stack and SGLang must already be installed.

### Prerequisites

Select mutually compatible versions for your Ascend hardware by following the
linked documentation. Python 3.11 is the verified configuration.

| Component | Version | Required | Manual installation | Installation |
|-----------|---------|----------|---------------------|--------------|
| CANN toolkit | Compatible release | Yes | Yes | [Official documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html) |
| PyTorch and `torch_npu` | Matching releases | Yes | Yes | [Official documentation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=177&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C) |
| `triton-ascend` | Match the selected PyTorch and CANN releases | Yes | Yes | [Official documentation](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/quick_start.md) |
| `sgl-kernel-npu` | Match PyTorch, Python, CANN, hardware, and architecture | Yes | Yes | [Official documentation](https://github.com/sgl-project/sgl-kernel-npu/releases) |
| `memfabric-hybrid` | Compatible release | No (PD disaggregation only) | Yes | [Official documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |
| SGLang for NPU | `sglang-version` in [`pyproject_npu.toml`](../../pyproject_npu.toml) | Yes | Yes | [Official documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |

### Install sglang-omni

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Check the environment and show the installation command without changing files.
bash scripts/npu/install_npu.sh --check

# Install sglang-omni in editable mode.
bash scripts/npu/install_npu.sh
```

The precheck reads the supported SGLang release from
`[tool.sglang-omni.npu]` in `pyproject_npu.toml` before swapping the install
manifest. This includes development, pre-release, post-release and local builds
of that release line. To print the configured release (Python 3.11+, or Python
3.10 with `tomli` installed), run `python scripts/npu/config.py --get sglang-version`.
It rejects other release lines,
including later releases. On a mismatch it reports both the supported line and
the installed version. The precheck also verifies the required Python packages,
matching `torch` and `torch_npu` major-minor versions, NPU availability, and a
small NPU matrix multiplication. Run `bash scripts/npu/install_npu.sh --help`
for optional extras, non-editable installation, and environments where devices
are intentionally not exposed during the build.

### TorchCodec installation for TTS models

Models in the TTS models utilize **`torchcodec`** for high-efficiency, native-streaming audio decoding directly into PyTorch tensors.

The helper script `scripts/npu/install_npu_torchcodec.sh` automatically installs:
* **Audio codec:** `ffmpeg`
* **CANN 9.1.0 stack:** `toolkit`, `A3-ops`, `nnal`
* **PyTorch 2.11 stack:** `torch`, `torchvision`, `torchaudio`, `torch_npu`, `torchcodec`

To run the installation:

```bash
# Specify your device type as the first argument (910b or A3)
bash scripts/npu/install_npu_torchcodec.sh A3
```
