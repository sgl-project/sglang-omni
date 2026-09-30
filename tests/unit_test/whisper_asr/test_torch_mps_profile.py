# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from unittest import mock

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.models.whisper_asr.torch_mps_runner import WhisperTorchMpsModelRunner


@contextlib.contextmanager
def _torch_mps():
    with (
        mock.patch(
            "sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=False
        ),
        mock.patch(
            "sglang_omni.models.whisper_asr.engine_builder.current_platform"
        ) as platform,
    ):
        platform.is_mps.return_value = True
        yield


def _builder(device: str = "mps") -> WhisperASREngineBuilder:
    builder = WhisperASREngineBuilder(
        max_running_requests=8,
        max_new_tokens=64,
        mem_fraction_static=0.7,
    )
    builder.context_length = 1860
    builder.device = device
    return builder


def test_torch_mps_selects_a_non_triton_attention_backend() -> None:
    """flashinfer is absent on Metal."""
    with _torch_mps():
        defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["mm_attention_backend"] == "sdpa"


def test_torch_mps_bounds_the_kv_pool_to_the_model_context() -> None:
    """Unified memory over-reports free memory, so the pool needs a hard cap."""
    with _torch_mps():
        defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["max_total_tokens"] == 1860
    assert defaults["max_prefill_tokens"] == 1860
    assert defaults["max_running_requests"] == 1
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False


def test_torch_mps_runner_disables_grad() -> None:
    """Omni's scheduler loops lack SGLang's @DynamicGradMode()."""
    runner = object.__new__(WhisperTorchMpsModelRunner)
    observed: dict[str, bool] = {}

    def _record(*args, **kwargs):
        observed["grad_enabled"] = torch.is_grad_enabled()
        return None

    with mock.patch.object(ModelRunner, "prepare_and_forward", _record):
        with torch.enable_grad():
            assert torch.is_grad_enabled()
            runner.prepare_and_forward(None, None, [], True)

    assert observed["grad_enabled"] is False


def test_torch_mps_runner_uses_no_grad_not_inference_mode() -> None:
    """inference_mode taints its outputs; the sampler mutates these logits."""
    runner = object.__new__(WhisperTorchMpsModelRunner)

    def _make_tensor(*args, **kwargs):
        return torch.zeros(2)

    with mock.patch.object(ModelRunner, "prepare_and_forward", _make_tensor):
        with torch.enable_grad():
            out = runner.prepare_and_forward(None, None, [], True)

    # An inference-mode tensor cannot be mutated afterwards; a no_grad one can.
    out.add_(1.0)
    assert pytest.approx(out.tolist()) == [1.0, 1.0]


def test_torch_mps_path_builds_its_own_runner() -> None:
    """The guard lives in the runner, so the Torch/MPS path has to install it."""
    with (
        _torch_mps(),
        mock.patch(
            "sglang_omni.models.whisper_asr.torch_mps_runner.WhisperTorchMpsModelRunner"
        ) as runner_cls,
    ):
        made = _builder().make_model_runner(mock.Mock(), mock.Mock())

    assert made is runner_cls.return_value


def test_apple_paths_clamp_concurrency_to_one() -> None:
    """Both Apple runners decode one request at a time."""
    overrides = {"max_running_requests": 64, "chunked_prefill_size": 0}

    with _torch_mps():
        _builder().adjust_overrides(overrides)

    assert overrides["max_running_requests"] == 1


def test_cuda_concurrency_is_left_alone() -> None:
    overrides = {"max_running_requests": 64, "chunked_prefill_size": 0}

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=False
        ),
        mock.patch(
            "sglang_omni.models.whisper_asr.engine_builder.current_platform"
        ) as platform,
    ):
        platform.is_mps.return_value = False
        _builder(device="cuda:0").adjust_overrides(overrides)

    assert overrides["max_running_requests"] == 64


def test_mlx_rejects_sampling() -> None:
    server_args = type(
        "Args", (), {"max_running_requests": 1, "mlx_enable_sampling": True}
    )()

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=True
        ),
        mock.patch(
            "sglang_omni.models.whisper_asr.engine_builder.current_platform"
        ) as platform,
    ):
        platform.is_mps.return_value = True
        with pytest.raises(ValueError, match="mlx_enable_sampling=False"):
            _builder().validate_before_infrastructure(server_args)
