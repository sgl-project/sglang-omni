# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the A100 / non-A100 dispatch and the
explicit-override contract in sglang_omni.models.qwen3_omni.compat.
"""

from __future__ import annotations

import unittest.mock as m

import pytest

from sglang_omni.models.qwen3_omni import compat
from sglang_omni.models.qwen3_omni.stages import (
    _SM80_INCOMPATIBLE_MOE_BACKENDS,
    _apply_compat_overrides,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    compat._device_capability.cache_clear()
    yield
    compat._device_capability.cache_clear()


def _patch_torch_caps(*caps):
    iterator = iter(caps)
    fake = m.MagicMock()
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = len(caps)
    fake.cuda.get_device_capability.side_effect = lambda i: next(iterator)
    return m.patch.dict("sys.modules", {"torch": fake})


def test_sm80_host_injects_triton():
    assert compat.get_qwen3_omni_compat_overrides(gpu_id=0) == {
        "moe_runner_backend": "triton"
    }


def test_non_sm80_host_is_noop():
    with _patch_torch_caps((9, 0)):
        assert compat.get_qwen3_omni_compat_overrides(gpu_id=0) == {}


def test_mixed_host_dispatches_per_gpu_id():
    with _patch_torch_caps((9, 0), (8, 0)):
        assert compat.get_qwen3_omni_compat_overrides(gpu_id=0) == {}
        assert compat.get_qwen3_omni_compat_overrides(gpu_id=1) == {
            "moe_runner_backend": "triton"
        }


def test_caller_unset_or_auto_receives_compat_on_sm80():
    for unset in ({}, {"moe_runner_backend": "auto"}):
        _apply_compat_overrides(stage_name="t", gpu_id=0, overrides=unset)
        assert unset == {"moe_runner_backend": "triton"}


def test_explicit_sm80_incompatible_backend_raises():
    overrides = {"moe_runner_backend": "flashinfer_cutlass"}
    with pytest.raises(ValueError, match="SM 80"):
        _apply_compat_overrides(stage_name="t", gpu_id=0, overrides=overrides)


def test_explicit_other_backend_preserved_on_sm80():
    for backend in ("triton", "flashinfer_trtllm", "cutlass"):
        overrides = {"moe_runner_backend": backend}
        _apply_compat_overrides(stage_name="t", gpu_id=0, overrides=overrides)
        assert overrides == {"moe_runner_backend": backend}, backend
