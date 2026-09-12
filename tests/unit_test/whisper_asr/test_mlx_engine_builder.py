# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder


@contextlib.contextmanager
def _backend(*, mlx: bool, mps: bool = True):
    with (
        mock.patch("sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=mlx),
        mock.patch(
            "sglang_omni.models.whisper_asr.engine_builder.current_platform"
        ) as platform,
    ):
        platform.is_mps.return_value = mps
        yield


def _builder(device: str = "mps") -> WhisperASREngineBuilder:
    builder = WhisperASREngineBuilder(
        max_running_requests=1,
        max_new_tokens=64,
        mem_fraction_static=0.7,
    )
    builder.context_length = 2048
    # AsrEngineBuilder.build assigns this; set it directly here.
    builder.device = device
    return builder


def test_mlx_defaults_disable_radix_and_graphs() -> None:
    """The encoder output lives in the cross-attention cache, not the KV pool.

    Radix reuse or a split prefill would hand a later chunk a cache that never
    saw the audio.
    """
    with _backend(mlx=True):
        defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["disable_radix_cache"] is True
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False
    assert defaults["chunked_prefill_size"] == 0
    assert "sampling_backend" not in defaults
    # Pinned rather than passed through, so the default launch does not need a
    # flag; an explicit override still reaches validate_before_infrastructure.
    assert defaults["max_running_requests"] == 1


def test_cuda_defaults_are_unchanged() -> None:
    with _backend(mlx=False, mps=False):
        defaults = _builder(device="cuda:0").generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is False
    assert defaults["enable_torch_compile"] is True
    assert defaults["sampling_backend"] == "pytorch"
    assert "disable_radix_cache" not in defaults


def test_mlx_requires_the_metal_platform() -> None:
    """SGLANG_USE_MLX=1 on a non-Apple host must fail loudly, not fall back."""
    with _backend(mlx=True, mps=False):
        with pytest.raises(RuntimeError, match="requires the Apple Metal platform"):
            _builder().generation_defaults(dtype="float16")


def test_mlx_does_not_advertise_custom_logit_processors() -> None:
    """The MLX prefill rejects logit editing, so the flag must stay off."""
    overrides: dict = {}
    with _backend(mlx=True):
        _builder().adjust_overrides(overrides)
    assert overrides["enable_custom_logit_processor"] is False

    overrides = {}
    with _backend(mlx=False):
        _builder().adjust_overrides(overrides)
    assert overrides["enable_custom_logit_processor"] is True


def test_mlx_skips_the_pre_lm_encoder_service() -> None:
    """That service caches Torch encoder states and drives encoder CUDA graphs."""
    builder = _builder()

    with _backend(mlx=True):
        builder.setup_runtime_resources(SimpleNamespace(), server_args=None)

    assert builder.audio_encoder_service is None


def test_mlx_uses_the_omni_mlx_scheduler_runner() -> None:
    from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner

    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(model=SimpleNamespace()),
    )

    with _backend(mlx=True):
        runner = _builder().make_model_runner(worker, SimpleNamespace())

    assert isinstance(runner, MlxSchedulerModelRunner)


def test_mlx_worker_dispatch_accepts_whisper_and_rejects_others() -> None:
    """create_mlx_model_worker gates on the architecture before doing any work.

    Whisper has to be listed there or the MLX path is unreachable, and the
    rejection message is what a new model's author reads first.
    """
    from sglang_omni.model_runner.mlx_model_worker import create_mlx_model_worker

    with pytest.raises(NotImplementedError) as excinfo:
        create_mlx_model_worker(
            config=SimpleNamespace(model_arch_override="SomeOtherModel"),
            server_args=SimpleNamespace(),
            gpu_id=0,
        )

    message = str(excinfo.value)
    assert "Qwen3ASRForConditionalGeneration" in message
    assert "WhisperForConditionalGeneration" in message


def test_mlx_worker_rejects_before_importing_the_mlx_backend() -> None:
    """The gate has to run before the MLX imports, not after.

    Those modules are absent on a non-Apple host, so checking afterwards turns a
    clear NotImplementedError into an ImportError for anyone who passes an
    unsupported architecture.
    """
    import builtins

    from sglang_omni.model_runner.mlx_model_worker import create_mlx_model_worker

    real_import = builtins.__import__

    def _no_mlx(name, *args, **kwargs):
        if "hardware_backend.mlx" in name or name.startswith("mlx"):
            raise ImportError(f"simulated missing backend: {name}")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", _no_mlx):
        with pytest.raises(NotImplementedError):
            create_mlx_model_worker(
                config=SimpleNamespace(model_arch_override="SomeOtherModel"),
                server_args=SimpleNamespace(),
                gpu_id=0,
            )
