# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.platforms.xpu import XpuSRTPlatform

import sglang_omni.platforms as platforms
import sglang_omni.platforms.xpu as xpu_platform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform


class _VendorDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "vendor"
    device_type = "vendor"

    def get_device(self, device_id: int = 0) -> str:
        return f"vendor:{device_id}"

    def set_device(self, device: torch.device) -> None:
        pass


class _VendorSRTPlatform(SRTPlatform, _VendorDeviceMixin):
    pass


def test_npu_probe_handles_torch_without_npu(monkeypatch) -> None:
    monkeypatch.delattr(torch, "npu", raising=False)

    assert platforms._is_npu_available() is False


def test_cpu_platform_needs_no_stage_process_env() -> None:
    spec = SimpleNamespace(stage_name="cpu", tp_size=2, gpu_id=None)

    assert CPUOmniPlatform().get_stage_process_env(spec, {}) == {}


def test_rocm_platform_keeps_cuda_compatible_tp_mapping() -> None:
    platform = platforms._as_omni_platform(RocmSRTPlatform())
    spec = SimpleNamespace(stage_name="thinker", tp_size=2, gpu_id=1)

    assert platform.is_rocm()
    assert not isinstance(platform, CUDAOmniPlatform)
    assert platform.device_type == "cuda"
    assert (
        platform.get_stage_process_env(spec, {"CUDA_VISIBLE_DEVICES": "3,4"})[
            "CUDA_VISIBLE_DEVICES"
        ]
        == "4"
    )


def test_rocm_platform_uses_conservative_omni_capabilities() -> None:
    from sglang_omni.comm.data_ref import TransportKind

    platform = ROCMOmniPlatform()

    assert platform.get_intra_node_transport() is TransportKind.SHM
    assert platform.get_fused_qk_norm_rope() is None


def test_rocm_talker_keeps_auto_moe_backend() -> None:
    server_args = SimpleNamespace(
        quantization=None,
        moe_runner_backend="auto",
    )
    model_config = SimpleNamespace(quantization=None)

    ROCMOmniPlatform().apply_model_worker_backend_policy(
        server_args,
        model_config,
        "Qwen3OmniTalker",
    )

    assert server_args.moe_runner_backend == "auto"


@pytest.mark.parametrize("backend", ["flashinfer_cutlass", "cutlass"])
def test_rocm_qwen3_omni_rejects_cutlass_moe_backends(backend: str) -> None:
    server_args = SimpleNamespace(quantization=None, moe_runner_backend=backend)
    model_config = SimpleNamespace(quantization=None)

    with pytest.raises(ValueError, match="NVIDIA CUDA-only"):
        ROCMOmniPlatform().apply_model_worker_backend_policy(
            server_args,
            model_config,
            "Qwen3OmniThinkerForCausalLM",
        )


def test_srt_plugin_identity_round_trips_to_spawned_process() -> None:
    qualname = f"{__name__}._VendorSRTPlatform"
    platform = platforms._load_platform_class(qualname)()

    restored = platforms._load_platform_class(platforms.get_platform_spec(platform))()

    assert isinstance(restored, OmniPlatform)
    assert restored.get_device(2) == "vendor:2"
    assert restored.get_stage_process_env(SimpleNamespace(), {}) == {}


@pytest.mark.parametrize(
    ("argument", "expected_index"),
    [
        (2, 2),
        (torch.device("xpu", 3), 3),
        (torch.device("xpu", 0), 0),  # index 0 must not read as "no index"
        (torch.device("xpu"), 0),
    ],
)
def test_xpu_set_device_accepts_an_index_or_a_device(
    monkeypatch, argument, expected_index
) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        xpu_platform.torch,
        "xpu",
        SimpleNamespace(set_device=seen.append),
        raising=False,
    )

    XPUOmniPlatform().set_device(argument)

    assert seen == [expected_index]


def test_xpu_platform_resolves_to_the_omni_xpu_platform() -> None:
    platform = platforms._as_omni_platform(XpuSRTPlatform())

    assert type(platform) is XPUOmniPlatform
    spec = SimpleNamespace(tp_size=2, gpu_id=0, stage_name="talker")
    assert platform.get_stage_process_env(spec, {"ZE_AFFINITY_MASK": "0,1"}) == {
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"
    }
