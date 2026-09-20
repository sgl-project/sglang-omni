# Installation — Ascend NPU

Install the Ascend software stack and NPU build of SGLang before installing
`sglang-omni`. The helper script
[`install_npu.sh`](../../scripts/npu/install_npu.sh) installs only
`sglang-omni`; it does not install or change any prerequisite in the table below.

## Prerequisites

Select mutually compatible versions for your Ascend hardware by following the
linked documentation. Python 3.11 is the verified configuration.

| Component | Version | Required | Manual installation | Installation |
|-----------|---------|----------|---------------------|--------------|
| CANN toolkit | Compatible release | Yes | Yes | [Official documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html) |
| HDK (driver and firmware) | Match the hardware and CANN release | Yes | Yes | [Official documentation](https://www.hiascend.com/hardware/firmware-drivers/community) |
| PyTorch and `torch_npu` | Matching releases | Yes | Yes | [Official documentation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=177&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C) |
| `triton-ascend` | Match the selected PyTorch and CANN releases | Yes | Yes | [Official documentation](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/quick_start.md) |
| `sgl-kernel-npu` | Match PyTorch, Python, CANN, hardware, and architecture | Yes | Yes | [Official documentation](https://github.com/sgl-project/sgl-kernel-npu/releases) |
| `memfabric-hybrid` | Compatible release | No (PD disaggregation only) | Yes | [Official documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |
| SGLang for NPU | `v0.5.18` | Yes | Yes | [Official documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |

## Install sglang-omni

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Check the environment and show the installation command without changing files.
bash scripts/npu/install_npu.sh --check

# Install sglang-omni in editable mode.
bash scripts/npu/install_npu.sh
```

The precheck accepts the SGLang `0.5.18` release line. This includes development,
pre-release, post-release, and local builds whose numeric release segment starts
with `0.5.18`, such as `0.5.18.dev7+g<git-sha>`. It rejects other release lines,
including later releases. On a mismatch it reports both the supported line and
the installed version. The precheck also verifies the required Python packages,
matching `torch` and `torch_npu` major-minor versions, NPU availability, and a
small NPU matrix multiplication. Run `bash scripts/npu/install_npu.sh --help`
for optional extras, non-editable installation, and environments where devices
are intentionally not exposed during the build.

## Torcodec installation for TTS models

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
