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
    PINNED_CHECKPOINT_REVISION,
    NemotronTranscribeEngineBuilder,
    prepare_nemotron_speculative_options,
    validate_nemotron_mtp_metadata,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def make_builder(
    compile_perception: bool = False,
    cuda_graphs: bool = False,
    *,
    nextn_enabled: bool = False,
    checkpoint_revision: str | None = None,
) -> NemotronTranscribeEngineBuilder:
    configuration = NemotronTranscribePipelineConfig(model_path="unused")
    parameters = configuration.stages[0].factory.model_dump(exclude_none=True)
    parameters.pop("dtype")
    parameters.update(
        compile_perception=compile_perception,
        perception_cuda_graphs=cuda_graphs,
        mem_fraction_static=None,
        nextn_enabled=nextn_enabled,
        checkpoint_revision=checkpoint_revision,
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


def test_nextn_is_off_by_default_and_keeps_server_overrides_unchanged() -> None:
    original = {"revision": "operator-revision", "max_running_requests": 2}
    overrides, enabled = prepare_nemotron_speculative_options("model", original)

    assert overrides == original
    assert overrides is not original
    assert enabled is False
    assert make_builder().infra_kwargs() == {}


def test_nextn_disabled_preserves_async_and_non_speculative_overrides() -> None:
    original = {
        "revision": "operator-revision",
        "disable_overlap_schedule": False,
        "enable_mixed_chunk": True,
    }
    overrides, enabled = prepare_nemotron_speculative_options(
        "model",
        original,
        enable_async_decode=True,
    )

    assert overrides == original
    assert enabled is False


def test_nextn_registers_the_native_draft_with_shared_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_builder.ModelRegistry,
        "models",
        dict(engine_builder.ModelRegistry.models),
    )
    overrides, enabled = prepare_nemotron_speculative_options(
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        {"speculative_algorithm": "NEXTN"},
    )
    builder = make_builder(
        nextn_enabled=enabled,
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
    )

    assert enabled is True
    assert overrides["speculative_algorithm"] == "NEXTN"
    assert overrides["revision"] == PINNED_CHECKPOINT_REVISION
    assert overrides["speculative_eagle_topk"] == 1
    assert overrides["enable_mixed_chunk"] is False
    assert builder.model_arch_override == "NemotronTranscribeForConditionalGeneration"
    assert builder.infra_kwargs() == {"supports_speculative_decoding": True}
    assert (
        engine_builder.ModelRegistry.models["NemotronHForCausalLMMTP"]
        is engine_builder.NemotronTranscribeMTP
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"speculative_algorithm": "DFLASH"}, "bundled NEXTN only"),
        (
            {"speculative_algorithm": "NEXTN", "revision": "other"},
            "checkpoint revision",
        ),
        (
            {
                "speculative_algorithm": "NEXTN",
                "speculative_draft_model_path": "other-model",
            },
            "target checkpoint",
        ),
        (
            {
                "speculative_algorithm": "NEXTN",
                "speculative_draft_model_revision": "other",
            },
            "target checkpoint revision",
        ),
        (
            {"speculative_algorithm": "NEXTN", "speculative_eagle_topk": 2},
            "topk=1",
        ),
        (
            {"speculative_algorithm": "NEXTN", "disable_overlap_schedule": False},
            "synchronous",
        ),
        (
            {"speculative_algorithm": "NEXTN", "enable_mixed_chunk": True},
            "mixed-chunk",
        ),
        (
            {"speculative_algorithm": "NEXTN", "enable_multi_layer_eagle": True},
            "one logical MTP step",
        ),
        (
            {"speculative_algorithm": "NEXTN", "tp_size": 2},
            "tp_size=1",
        ),
    ],
)
def test_unsupported_nextn_options_fail_before_checkpoint_loading(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_nemotron_speculative_options("model", overrides)


def test_async_decode_is_rejected_for_nextn() -> None:
    with pytest.raises(ValueError, match="async decode"):
        prepare_nemotron_speculative_options(
            "model", {"speculative_algorithm": "NEXTN"}, enable_async_decode=True
        )


def test_nextn_post_resolution_validation_checks_only_resolved_draft_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = make_builder(
        nextn_enabled=True,
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
    )
    builder.checkpoint_dir = "resolved-target"
    monkeypatch.setattr(
        engine_builder.AsrEngineBuilder,
        "validate_before_infrastructure",
        lambda self, server_args: None,
    )
    monkeypatch.setattr(
        engine_builder,
        "resolved_view",
        lambda server_args: SimpleNamespace(speculative_draft_model_path="user-target"),
    )

    with pytest.raises(ValueError, match="resolved target checkpoint"):
        builder.validate_before_infrastructure(SimpleNamespace())


def test_nextn_requires_the_known_two_layer_physical_pattern() -> None:
    validate_nemotron_mtp_metadata(
        {
            "llm_config": {"num_nextn_predict_layers": 1},
            "mtp": {
                "enabled": True,
                "num_nextn_predict_layers": 2,
                "use_repeated_layer": True,
                "hybrid_override_pattern": "*E",
            },
        }
    )
    with pytest.raises(ValueError, match="two physical MTP layers"):
        validate_nemotron_mtp_metadata(
            {
                "llm_config": {"num_nextn_predict_layers": 1},
                "mtp": {
                    "enabled": True,
                    "num_nextn_predict_layers": 1,
                    "use_repeated_layer": True,
                    "hybrid_override_pattern": "*E",
                },
            }
        )


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
    model = SimpleNamespace()
    builder.setup_runtime_resources(model, SimpleNamespace())
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
    builder.enable_pre_lm_encoder = False
    builder.setup_runtime_resources(SimpleNamespace(), SimpleNamespace())
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
