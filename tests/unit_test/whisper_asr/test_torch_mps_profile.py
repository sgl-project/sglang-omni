# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from unittest import mock

import pytest
import torch

from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.models.whisper_asr.sglang_model import WhisperForConditionalGeneration


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
    # AsrEngineBuilder.build assigns this; set it directly here.
    builder.device = device
    return builder


def test_torch_mps_selects_a_non_triton_attention_backend() -> None:
    """flashinfer is absent on Metal.

    Leaving the default in place fails at import with "name
    'BatchPrefillWithRaggedKVCacheWrapper' is not defined", and the scheduler's
    KV-index writer also takes its Triton path.
    """
    with _torch_mps():
        defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["mm_attention_backend"] == "sdpa"


def test_torch_mps_bounds_the_kv_pool_to_the_model_context() -> None:
    """Unified memory over-reports free memory, so the pool needs a hard cap.

    Without it the sizer walks past physical RAM until Metal refuses, which on
    a 24 GB host meant an unrecoverable OOM mid-request.
    """
    with _torch_mps():
        defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["max_total_tokens"] == 1860
    assert defaults["max_prefill_tokens"] == 1860
    assert defaults["max_running_requests"] == 1
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False


def test_forward_runs_with_grad_disabled() -> None:
    """Omni's scheduler loops lack SGLang's @DynamicGradMode().

    Without a guard here every request retains its autograd graph, which on
    Apple Metal is ~4.6 GB of live tensors per request for large-v3.
    """
    model = WhisperForConditionalGeneration.__new__(WhisperForConditionalGeneration)
    observed: dict[str, bool] = {}

    def _record(*args, **kwargs):
        observed["grad_enabled"] = torch.is_grad_enabled()
        return None

    model._forward = _record

    with torch.enable_grad():
        assert torch.is_grad_enabled()
        model.forward(input_ids=None, positions=None, forward_batch=None)

    assert observed["grad_enabled"] is False


def test_forward_guard_uses_no_grad_not_inference_mode() -> None:
    """inference_mode taints its outputs; the sampler mutates these logits.

    Using it here raises "Inplace update to inference tensor outside
    InferenceMode is not allowed" once sampling runs.
    """
    model = WhisperForConditionalGeneration.__new__(WhisperForConditionalGeneration)

    def _make_tensor(*args, **kwargs):
        return torch.zeros(2)

    model._forward = _make_tensor

    with torch.enable_grad():
        out = model.forward(input_ids=None, positions=None, forward_batch=None)

    # An inference-mode tensor cannot be mutated afterwards; a no_grad one can.
    out.add_(1.0)
    assert pytest.approx(out.tolist()) == [1.0, 1.0]


def test_apple_paths_clamp_concurrency_to_one() -> None:
    """Both Apple runners decode one request at a time.

    The clamp has to land in adjust_overrides: the stage's own EngineArgs carry
    the CUDA value of 64 and take precedence over generation_defaults, so a
    default launch would otherwise reach a runner that cannot serve it.
    """
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
