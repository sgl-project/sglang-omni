# SPDX-License-Identifier: Apache-2.0
"""Engine option handling without model weights or CUDA initialization."""

from types import SimpleNamespace

import pytest
from torch import nn

import sglang_omni.models.nemotron_transcribe.engine_builder as engine_builder
from sglang_omni.models.nemotron_transcribe.config import (
    EncoderCompileMode,
    NemotronTranscribePipelineConfig,
)
from sglang_omni.models.nemotron_transcribe.engine_builder import (
    NemotronTranscribeEngineBuilder,
)
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
    assert defaults["disable_cuda_graph"] is False
    assert defaults["disable_radix_cache"] is True


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
