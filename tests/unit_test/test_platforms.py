# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.platforms.xpu import XpuSRTPlatform

import sglang_omni.platforms as platforms
import sglang_omni.platforms.xpu as xpu_platform
from sglang_omni.pipeline.stage_workers import StageLaunchConfig
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform


class VendorDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "vendor"
    device_type = "vendor"

    def get_device(self, device_id: int = 0) -> str:
        return f"vendor:{device_id}"

    def set_device(self, device: torch.device) -> None:
        pass


class VendorSRTPlatform(SRTPlatform, VendorDeviceMixin):
    pass


def test_cuda_tp_stage_env_is_the_narrowing_plus_nvls_off() -> None:
    spec = StageLaunchConfig(stage_name="thinker", tp_size=2, gpu_id=1)

    env = CUDAOmniPlatform().get_stage_process_env(
        spec, {"CUDA_VISIBLE_DEVICES": "3,4"}
    )

    assert env == {
        "CUDA_VISIBLE_DEVICES": "4",
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "NCCL_NVLS_ENABLE": "0",
    }


def test_rocm_platform_keeps_cuda_compatible_tp_mapping() -> None:
    platform = platforms.as_omni_platform(RocmSRTPlatform())
    spec = SimpleNamespace(
        stage_name="thinker", tp_size=2, allow_child_processes=False, gpu_id=1
    )

    assert platform.is_rocm()
    assert not isinstance(platform, CUDAOmniPlatform)
    assert platform.device_type == "cuda"
    assert (
        platform.get_stage_process_env(spec, {"CUDA_VISIBLE_DEVICES": "3,4"})[
            "CUDA_VISIBLE_DEVICES"
        ]
        == "4"
    )


def test_rocm_platform_maps_tp_rank_through_hip_visible_devices() -> None:
    platform = ROCMOmniPlatform()
    spec = SimpleNamespace(
        stage_name="thinker", tp_size=2, allow_child_processes=False, gpu_id=1
    )

    mapped_env = platform.get_stage_process_env(spec, {"HIP_VISIBLE_DEVICES": "3,4"})

    assert mapped_env["HIP_VISIBLE_DEVICES"] == "4"
    assert mapped_env["CUDA_VISIBLE_DEVICES"] == "4"
    assert mapped_env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_rocm_platform_prefers_hip_visibility_over_cuda_alias() -> None:
    """HIP_VISIBLE_DEVICES wins in the HIP runtime, so the rank maps through it;
    the CUDA alias mirrors the same physical id for the CUDA-named helpers."""
    platform = ROCMOmniPlatform()
    spec = SimpleNamespace(
        stage_name="thinker", tp_size=2, allow_child_processes=False, gpu_id=1
    )

    mapped_env = platform.get_stage_process_env(
        spec,
        {
            "HIP_VISIBLE_DEVICES": "3,5",
            "CUDA_VISIBLE_DEVICES": "6,7",
        },
    )

    assert mapped_env["HIP_VISIBLE_DEVICES"] == "5"
    assert mapped_env["CUDA_VISIBLE_DEVICES"] == "5"


def test_rocm_platform_without_hip_visibility_sets_only_the_cuda_alias() -> None:
    platform = ROCMOmniPlatform()
    spec = SimpleNamespace(
        stage_name="thinker", tp_size=2, allow_child_processes=False, gpu_id=0
    )

    assert "HIP_VISIBLE_DEVICES" not in platform.get_stage_process_env(spec, {})
    assert platform.get_stage_process_env(spec, {})["CUDA_VISIBLE_DEVICES"] == "0"


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

    assert resolution_result(server_args, "moe_runner_backend") == "auto"


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
    qualname = f"{__name__}.VendorSRTPlatform"
    platform = platforms.load_platform_class(qualname)()

    restored = platforms.load_platform_class(platforms.get_platform_spec(platform))()

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
    platform = platforms.as_omni_platform(XpuSRTPlatform())

    assert type(platform) is XPUOmniPlatform
    spec = SimpleNamespace(tp_size=2, gpu_id=0, stage_name="talker")
    assert platform.get_stage_process_env(spec, {"ZE_AFFINITY_MASK": "0,1"}) == {
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"
    }


def test_each_platform_names_the_graph_backend_its_hardware_uses() -> None:
    """The accelerators that capture name a backend; the rest answer None.

    CPU and Apple keep the base None.
    """
    from sglang_omni.platforms.apple import AppleOmniPlatform
    from sglang_omni.platforms.device_graph import (
        CudaDeviceGraphBackend,
        NpuDeviceGraphBackend,
        XpuDeviceGraphBackend,
    )
    from sglang_omni.platforms.musa import MUSAOmniPlatform

    expected = {
        CUDAOmniPlatform: CudaDeviceGraphBackend,
        ROCMOmniPlatform: CudaDeviceGraphBackend,
        MUSAOmniPlatform: CudaDeviceGraphBackend,
        xpu_platform.XPUOmniPlatform: XpuDeviceGraphBackend,
        NPUOmniPlatform: NpuDeviceGraphBackend,
        CPUOmniPlatform: None,
        AppleOmniPlatform: None,
        OmniPlatform: None,
    }
    for platform_class, backend_class in expected.items():
        platform = platform_class()
        device = SimpleNamespace(type=platform.device_type)
        backend = platform.get_device_graph_backend(device)
        if backend_class is None:
            assert backend is None, platform_class.__name__
        else:
            assert isinstance(backend, backend_class), platform_class.__name__


def test_npu_platform_opts_only_mimi_into_decode_graphs() -> None:
    platform = NPUOmniPlatform()
    assert platform.enable_mimi_decode_graph() is True
    # #1721: the Higgs TTS codec keeps its eager NPU vocoder policy.
    assert platform.enable_codec_decode_graph() is False


def test_a_platform_declines_a_device_that_is_not_its_own() -> None:
    """A caller holding a tensor's device does not have to check that first."""
    platform = CUDAOmniPlatform()

    assert platform.get_device_graph_backend(torch.device("xpu", 0)) is None
    assert platform.get_device_graph_backend(torch.device("meta")) is None
    assert platform.get_device_graph_backend(torch.device("cpu")) is None


def test_a_platform_that_names_no_sdpa_backend_never_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn.attention as attention

    calls: list[object] = []

    def recording_pin(backends):
        calls.append(backends)
        return nullcontext()

    monkeypatch.setattr(attention, "sdpa_kernel", recording_pin)

    with OmniPlatform().graph_capture_attention():
        pass

    assert calls == []


def test_the_pin_receives_exactly_the_backends_the_hook_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn.attention as attention
    from torch.nn.attention import SDPBackend

    calls: list[list[SDPBackend]] = []

    def recording_pin(backends):
        calls.append(list(backends))
        return nullcontext()

    monkeypatch.setattr(attention, "sdpa_kernel", recording_pin)
    platform = xpu_platform.XPUOmniPlatform()

    with platform.graph_capture_attention():
        pass

    assert calls == [list(platform.get_graph_capture_sdpa_backends())]
    assert calls[0], "an empty set would leave dispatch on the uncapturable default"


def test_cpu_cuda_rocm_xpu_and_musa_implement_the_shared_device_operations() -> None:
    for platform_type in (
        CPUOmniPlatform,
        CUDAOmniPlatform,
        ROCMOmniPlatform,
        XPUOmniPlatform,
        platforms.MUSAOmniPlatform,
    ):
        platform = platform_type()
        assert type(platform).empty_cache is not DeviceMixin.empty_cache, platform_type
        assert (
            type(platform).get_available_memory is not DeviceMixin.get_available_memory
        ), platform_type
