# SPDX-License-Identifier: Apache-2.0
"""Engine option handling without model weights or CUDA initialization."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sglang_omni.models.nemotron_transcribe.engine_builder as engine_builder
from sglang_omni.models.nemotron_transcribe.config import (
    EncoderCompileMode,
    NemotronTranscribePipelineConfig,
)
from sglang_omni.models.nemotron_transcribe.engine_builder import (
    NemotronTranscribeEngineBuilder,
)
from sglang_omni.models.nemotron_transcribe.fp8_linear import Fp8Linear
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def make_builder(
    compile_perception: bool = False, cuda_graphs: bool = False
) -> NemotronTranscribeEngineBuilder:
    configuration = NemotronTranscribePipelineConfig(model_path="unused")
    parameters = configuration.stages[0].factory.model_dump(exclude_none=True)
    parameters.pop("dtype")
    parameters.update(
        compile_perception=compile_perception,
        perception_cuda_graphs=cuda_graphs,
        mem_fraction_static=None,
    )
    return NemotronTranscribeEngineBuilder(**parameters)


class RecordingEncoder(SpeechTransformerEncoder):
    def __init__(self) -> None:
        super().__init__(
            feat_in=4, d_model=16, n_heads=2, n_layers=1, subsampling_factor=2
        )
        self.dynamic_batches: bool | None = None

    def compile_layers(
        self, mode: EncoderCompileMode, cuda_graphs: bool, dynamic: bool
    ) -> None:
        self.dynamic_batches = dynamic


def test_decoder_defaults_keep_cuda_graphs_and_disable_compilation() -> None:
    defaults = make_builder().generation_defaults(dtype="bfloat16")
    assert defaults["enable_torch_compile"] is False
    assert defaults["disable_radix_cache"] is True


def test_loop_guard_enables_per_request_logit_processors() -> None:
    builder = make_builder()
    overrides: dict[str, object] = {}
    builder.adjust_overrides(overrides)
    assert overrides == {"enable_custom_logit_processor": True}

    builder.loop_guard = False
    overrides = {}
    builder.adjust_overrides(overrides)
    assert overrides == {}


def test_decoder_compilation_has_an_actionable_error() -> None:
    with pytest.raises(ValueError, match="compile_perception"):
        make_builder().customize_server_args(SimpleNamespace(enable_torch_compile=True))


def test_encoder_graphs_require_encoder_compilation() -> None:
    with pytest.raises(ValueError, match="requires compile_perception"):
        make_builder(cuda_graphs=True)


@pytest.mark.parametrize("decoder_slots,expected_dynamic", [(1, False), (16, True)])
def test_encoder_compilation_uses_resolved_decoder_concurrency(
    monkeypatch: pytest.MonkeyPatch, decoder_slots: int, expected_dynamic: bool
) -> None:
    monkeypatch.setattr(
        engine_builder, "init_mm_embedding_cache", lambda size_bytes: None
    )
    encoder = RecordingEncoder()
    model = SimpleNamespace(perception=nn.Sequential(encoder), encoder_max_batch_size=0)
    make_builder(compile_perception=True).setup_model_resources(
        model,
        SimpleNamespace(max_running_requests=decoder_slots),
        generation_cuda_graph_enabled=True,
    )
    assert encoder.dynamic_batches is expected_dynamic
    assert model.encoder_max_batch_size == 8


@pytest.mark.parametrize(
    "perception_fp8,compile_perception,gemm_supported,keeps_fp8",
    [
        (True, False, True, True),
        (True, True, True, True),
        (True, False, False, False),
        (False, False, True, False),
    ],
)
def test_fp8_encoder_linears_need_the_option_and_fp8_gemms(
    monkeypatch: pytest.MonkeyPatch,
    perception_fp8: bool,
    compile_perception: bool,
    gemm_supported: bool,
    keeps_fp8: bool,
) -> None:
    monkeypatch.setattr(
        engine_builder, "init_mm_embedding_cache", lambda size_bytes: None
    )
    monkeypatch.setattr(
        engine_builder, "fp8_gemm_supported", lambda device: gemm_supported
    )
    linear = Fp8Linear.replacing(nn.Linear(16, 16, dtype=torch.bfloat16))
    model = SimpleNamespace(perception=nn.Sequential(linear), encoder_max_batch_size=0)
    builder = make_builder(compile_perception=compile_perception)
    builder.perception_fp8 = perception_fp8
    builder.setup_model_resources(
        model,
        SimpleNamespace(max_running_requests=1),
        generation_cuda_graph_enabled=True,
    )
    assert isinstance(model.perception[0], Fp8Linear) is keeps_fp8
    assert isinstance(model.perception[0], nn.Linear) is not keeps_fp8


def test_pre_lm_encoder_starts_after_model_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class Service:
        def __init__(self, model: object, **kwargs: object) -> None:
            created["model"] = model
            created["kwargs"] = kwargs

        def close(self) -> None:
            created["closed"] = True

    monkeypatch.setattr(
        engine_builder, "NemotronTranscribePreLMEncoderService", Service
    )
    monkeypatch.setattr(
        engine_builder,
        "build_cache_namespace",
        lambda model, *, model_path: f"{model_path}:ns",
    )
    builder = make_builder()
    builder.checkpoint_dir = "checkpoint"
    attached: list[str] = []
    model = SimpleNamespace(attach_ctc_head=attached.append)
    builder.setup_runtime_resources(model, SimpleNamespace())
    assert attached == ["checkpoint"]
    assert created["model"] is model
    assert created["kwargs"] == {
        "cache_namespace": "checkpoint:ns",
        "cache_max_entries": 128,
        "cache_max_bytes": 2 * 1024**3,
        "max_batch_size": 8,
        "max_batch_wait_ms": 0,
    }
    builder.cleanup_build_failure()
    assert created["closed"] is True
    assert builder.audio_encoder_service is None


def test_disabled_pre_lm_encoder_does_not_start_a_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("encoder service must stay stopped")

    monkeypatch.setattr(engine_builder, "NemotronTranscribePreLMEncoderService", fail)
    builder = make_builder()
    builder.checkpoint_dir = "checkpoint"
    builder.enable_pre_lm_encoder = False
    attached: list[str] = []
    builder.setup_runtime_resources(
        SimpleNamespace(attach_ctc_head=attached.append), SimpleNamespace()
    )
    assert attached == ["checkpoint"]
    assert builder.audio_encoder_service is None


@pytest.mark.parametrize("decoder_slots", [None, 1, 16])
def test_engine_defaults_merge_with_operator_concurrency(
    decoder_slots: int | None,
) -> None:
    configuration = NemotronTranscribePipelineConfig(model_path="unused")
    expected_slots = (
        decoder_slots or configuration.stages[0].engine.max_running_requests
    )
    overrides = build_generation_batch_overrides(
        **make_builder().generation_defaults(dtype="bfloat16"),
        server_args_overrides=(
            {"max_running_requests": decoder_slots} if decoder_slots is not None else {}
        ),
    )
    assert overrides["max_running_requests"] == expected_slots
    assert overrides["cuda_graph_max_bs"] == expected_slots
    assert overrides["torch_compile_max_bs"] == expected_slots
