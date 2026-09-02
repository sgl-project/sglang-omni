# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Intel CPU platform layer (no accelerator required).

These instantiate ``CPUOmniPlatform`` directly rather than reading
``current_platform``, so the CPU contract is checked on every host — including
CUDA CI machines, where a ``current_platform``-based test would silently skip.
"""

from __future__ import annotations

import torch

import sglang_omni.utils.device as dev
from sglang_omni.platforms import _as_omni_platform, get_platform_spec
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform


def test_cpu_platform_identifies_as_cpu():
    """``device_type`` alone is not enough to tell CPU from a fallback platform:
    the generic SRT platform also reports 'cpu' while ``is_cpu()`` stays False.
    Both must agree, or code branching on ``is_cpu()`` takes the accelerator path.
    """
    platform = CPUOmniPlatform()

    assert platform.is_cpu()
    assert platform.device_type == "cpu"
    assert platform.device_name == "cpu"
    assert not platform.is_cuda()
    assert not platform.is_xpu()


def test_cpu_platform_is_an_omni_platform():
    """The stage layer calls Omni hooks on whatever ``current_platform`` returns."""
    assert isinstance(CPUOmniPlatform(), OmniPlatform)


def test_a_cpu_srt_platform_resolves_to_the_cpu_omni_platform():
    """Resolution wiring: an upstream CPU platform must map to the Omni subclass
    rather than fall through to the generic wrapper, which would lose every
    CPU-specific hook below.
    """

    class FakeCpuSRTPlatform:
        def is_cuda(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return False

        def is_cpu(self) -> bool:
            return True

        def is_xpu(self) -> bool:
            return False

    resolved = _as_omni_platform(FakeCpuSRTPlatform())

    assert isinstance(resolved, CPUOmniPlatform)
    assert get_platform_spec(resolved).endswith("CPUOmniPlatform")


def test_get_device_ignores_the_rank_and_returns_an_index_free_device():
    """Stages carry a placement id — the Qwen3-Omni talker is handed 1 — but a CPU
    torch.device may only have index -1 or 0 (c10 Device::validate), so the rank
    must not become an index. Per-rank isolation is NUMA/OpenMP binding, not the
    device object.
    """
    platform = CPUOmniPlatform()

    for local_rank in (0, 1, 7):
        device = platform.get_device(local_rank)
        assert device == torch.device("cpu")
        assert device.type == "cpu"
        assert device.index is None


def test_set_device_accepts_the_cpu_device_without_error():
    """Called for symmetry with the CUDA path; it must stay a harmless no-op and
    must not flip the process-wide default tensor device.
    """
    before = torch.empty(0).device

    CPUOmniPlatform().set_device(torch.device("cpu"))

    assert torch.empty(0).device == before


def test_code2wav_graph_capture_is_refused_on_cpu():
    """There is no graph to capture on CPU; leaving this True would fail during
    capture instead of at configuration time.
    """
    assert CPUOmniPlatform().enable_code2wav_graph() is False


def test_no_accelerator_only_fused_kernel_is_advertised():
    """These hooks return a kernel only where one exists. Wiring an accelerator
    kernel here would be a dispatch failure deep inside the model forward.
    """
    platform = CPUOmniPlatform()

    assert platform.get_fused_qk_norm_rope() is None
    assert platform.get_fused_qk_norm_rope_with_cos_sin_cache() is None


def test_cpu_stages_need_no_process_env_overrides():
    """Unlike XPU, which must widen ZE_AFFINITY_MASK across a TP group, CPU ranks
    share one device and need nothing injected before child startup.
    """
    from types import SimpleNamespace

    spec = SimpleNamespace(stage_name="thinker", tp_size=4, gpu_id=None)

    assert CPUOmniPlatform().get_stage_process_env(spec, {}) == {}


def test_a_cpu_device_spec_never_gains_a_placement_index():
    """``place_device_spec`` applies a stage's gpu_id to the caller's device. On
    CPU that index has to be dropped, not appended: 'cpu:2' is not a valid device.
    """
    assert dev.place_device_spec("cpu", 2) == "cpu"
    assert dev.place_device_spec("cpu") == "cpu"
    assert dev.resolve_device_spec("cpu", 5) == "cpu"
    assert dev.resolve_device_spec("cpu") == "cpu"
