# SPDX-License-Identifier: Apache-2.0
"""Engine option handling without model weights or CUDA initialization."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.server_args import ServerArgs
from torch import nn

import sglang_omni.models.nemotron_transcribe.engine_builder as engine_builder
from sglang_omni.models.nemotron_transcribe.config import (
    EncoderCompileMode,
    NemotronTranscribePipelineConfig,
)
from sglang_omni.models.nemotron_transcribe.engine_builder import (
    NemotronTranscribeEngineBuilder,
    prepare_nemotron_speculative_options,
    validate_nemotron_mtp_metadata,
)
from sglang_omni.models.nemotron_transcribe.hf_config import (
    NemotronTranscribeConfig,
    register_nemotron_transcribe_hf_config,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def synthetic_checkpoint_config(
    *, pattern: str = "*E", physical_depth: int = 2
) -> dict[str, object]:
    return {
        "llm_config": {"num_nextn_predict_layers": 1},
        "eos_token_id": [11],
        "perception": {},
        "pe_encoder_config": {},
        "mtp": {
            "enabled": True,
            "num_nextn_predict_layers": physical_depth,
            "use_repeated_layer": True,
            "hybrid_override_pattern": pattern,
        },
    }


def write_synthetic_checkpoint(
    checkpoint_directory: Path,
    *,
    pattern: str = "*E",
    physical_depth: int = 2,
) -> Path:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    (checkpoint_directory / "config.json").write_text(
        json.dumps(
            synthetic_checkpoint_config(
                pattern=pattern,
                physical_depth=physical_depth,
            )
        )
    )
    (checkpoint_directory / "weights.safetensors").write_bytes(b"checkpoint")
    return checkpoint_directory


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
    builder = make_builder(nextn_enabled=enabled)

    assert enabled is True
    assert overrides["speculative_algorithm"] == "NEXTN"
    assert "revision" not in overrides
    assert "speculative_draft_model_revision" not in overrides
    assert overrides["speculative_eagle_topk"] == 1
    assert overrides["speculative_num_steps"] == 2
    assert overrides["speculative_num_draft_tokens"] == 3
    assert overrides["enable_mixed_chunk"] is False
    assert builder.model_arch_override == "NemotronTranscribeForConditionalGeneration"
    assert builder.infra_kwargs() == {"supports_speculative_decoding": True}
    assert (
        engine_builder.ModelRegistry.models["NemotronHForCausalLMMTP"]
        is engine_builder.NemotronTranscribeMTP
    )


def test_nextn_preserves_an_operator_selected_revision() -> None:
    overrides, enabled = prepare_nemotron_speculative_options(
        "org/nemotron-transcribe",
        {"speculative_algorithm": "NEXTN", "revision": "release-2026-10"},
    )

    assert enabled is True
    assert overrides["revision"] == "release-2026-10"
    assert overrides["speculative_draft_model_revision"] == "release-2026-10"


def test_nextn_preserves_explicit_steps_and_derives_matching_draft_tokens() -> None:
    overrides, enabled = prepare_nemotron_speculative_options(
        "org/nemotron-transcribe",
        {
            "speculative_algorithm": "NEXTN",
            "speculative_num_steps": 7,
            "speculative_num_draft_tokens": 8,
        },
    )

    assert enabled is True
    assert overrides["speculative_num_steps"] == 7
    assert overrides["speculative_num_draft_tokens"] == 8


def test_nextn_defaults_resolve_through_sglang_server_args(
    tmp_path: Path,
) -> None:
    register_nemotron_transcribe_hf_config()
    NemotronTranscribeConfig.from_nemo_dict(
        synthetic_checkpoint_config()
    ).save_pretrained(tmp_path)
    overrides, enabled = prepare_nemotron_speculative_options(
        str(tmp_path), {"speculative_algorithm": "NEXTN"}
    )
    assert enabled is True

    server_args_overrides = dict(overrides, disable_overlap_schedule=True)
    server_args = ServerArgs(
        model_path=str(tmp_path), device="cpu", **server_args_overrides
    )
    server_args.resolve_once()
    resolved = resolved_view(server_args)

    assert resolved.speculative_num_steps == 2
    assert resolved.speculative_eagle_topk == 1
    assert resolved.speculative_num_draft_tokens == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"speculative_algorithm": "DFLASH"}, "bundled NEXTN only"),
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
            {
                "speculative_algorithm": "NEXTN",
                "revision": "target-revision",
                "speculative_draft_model_revision": "draft-revision",
            },
            "target checkpoint revision",
        ),
        (
            {"speculative_algorithm": "NEXTN", "speculative_eagle_topk": 2},
            "topk=1",
        ),
        (
            {
                "speculative_algorithm": "NEXTN",
                "speculative_num_steps": 4,
                "speculative_num_draft_tokens": 4,
            },
            "speculative_num_draft_tokens=5",
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
    builder = make_builder(nextn_enabled=True)
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


@pytest.mark.parametrize(("pattern", "physical_depth"), [("E*", 2), ("E", 1)])
def test_nextn_accepts_supported_physical_mtp_patterns(
    pattern: str, physical_depth: int
) -> None:
    validate_nemotron_mtp_metadata(
        synthetic_checkpoint_config(pattern=pattern, physical_depth=physical_depth)
    )


def test_nextn_rejects_unsupported_mtp_metadata() -> None:
    for unsupported_pattern in ("?E", "M", "-"):
        checkpoint_config = synthetic_checkpoint_config(
            pattern=unsupported_pattern,
            physical_depth=len(unsupported_pattern),
        )
        with pytest.raises(ValueError, match="supports only"):
            validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config(pattern="E*", physical_depth=1)
    with pytest.raises(ValueError, match="pattern length"):
        validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config(pattern="E", physical_depth=1)
    llm_config = checkpoint_config["llm_config"]
    assert isinstance(llm_config, dict)
    llm_config["num_nextn_predict_layers"] = 2
    with pytest.raises(ValueError, match="one logical MTP step"):
        validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config(pattern="E", physical_depth=1)
    mtp_config = checkpoint_config["mtp"]
    assert isinstance(mtp_config, dict)
    mtp_config["enabled"] = False
    with pytest.raises(ValueError, match="does not enable bundled MTP"):
        validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config(pattern="E", physical_depth=1)
    mtp_config = checkpoint_config["mtp"]
    assert isinstance(mtp_config, dict)
    mtp_config["use_repeated_layer"] = False
    with pytest.raises(ValueError, match="repeated MTP layers"):
        validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config(pattern="", physical_depth=0)
    with pytest.raises(ValueError, match="non-empty MTP pattern"):
        validate_nemotron_mtp_metadata(checkpoint_config)

    checkpoint_config = synthetic_checkpoint_config()
    checkpoint_config.pop("mtp")
    with pytest.raises(ValueError, match="target and MTP metadata"):
        validate_nemotron_mtp_metadata(checkpoint_config)


def test_converted_target_config_keeps_zero_layers_and_validates_original_mtp() -> None:
    converted_config = NemotronTranscribeConfig.from_nemo_dict(
        synthetic_checkpoint_config()
    ).to_dict()

    assert converted_config["num_nextn_predict_layers"] == 0
    validate_nemotron_mtp_metadata(converted_config)

    converted_config["num_nextn_predict_layers"] = 1
    with pytest.raises(ValueError, match="converted Nemotron Transcribe target config"):
        validate_nemotron_mtp_metadata(converted_config)


@pytest.mark.parametrize("nextn_enabled", [False, True])
@pytest.mark.parametrize("cache_hit", [False, True])
def test_remote_revision_resolves_the_requested_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nextn_enabled: bool,
    cache_hit: bool,
) -> None:
    snapshot_directory = write_synthetic_checkpoint(tmp_path / "snapshot")
    requested: list[tuple[str, str | None, bool]] = []

    def download_snapshot(
        model_path: str,
        *,
        revision: str | None,
        local_files_only: bool,
    ) -> str:
        requested.append((model_path, revision, local_files_only))
        if local_files_only and not cache_hit:
            raise LocalEntryNotFoundError("cache miss")
        else:
            pass
        return str(snapshot_directory)

    monkeypatch.setattr(engine_builder, "snapshot_download", download_snapshot)
    builder = make_builder(
        nextn_enabled=nextn_enabled,
        checkpoint_revision="operator-revision",
    )
    resolved_directory = Path(builder.resolve_checkpoint("org/model"))

    expected_requests = [("org/model", "operator-revision", True)]
    if not cache_hit:
        expected_requests.append(("org/model", "operator-revision", False))
    else:
        pass
    assert requested == expected_requests
    assert (resolved_directory / "weights.safetensors").resolve() == (
        snapshot_directory / "weights.safetensors"
    ).resolve()
    if nextn_enabled:
        builder.checkpoint_dir = str(resolved_directory)
        overrides = {"revision": "operator-revision"}
        builder.adjust_overrides(overrides)
        assert overrides["revision"] == "operator-revision"
        assert overrides["speculative_draft_model_path"] == str(resolved_directory)
        assert overrides["speculative_draft_model_revision"] == "operator-revision"
    else:
        pass


def test_remote_nextn_uses_default_revision_without_a_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_directory = write_synthetic_checkpoint(
        tmp_path / "snapshot", pattern="E", physical_depth=1
    )
    resolved: list[str] = []
    local_requests: list[tuple[str, bool]] = []

    def find_cached_snapshot(
        model_path: str,
        *,
        revision: str | None,
        local_files_only: bool,
    ) -> str:
        local_requests.append((model_path, local_files_only))
        assert revision is None
        assert local_files_only is True
        raise LocalEntryNotFoundError("cache miss")

    def resolve_default_revision(model_path: str) -> Path:
        resolved.append(model_path)
        return snapshot_directory

    monkeypatch.setattr(engine_builder, "snapshot_download", find_cached_snapshot)
    monkeypatch.setattr(engine_builder, "resolve_model_path", resolve_default_revision)
    builder = make_builder(nextn_enabled=True)

    resolved_directory = Path(builder.resolve_checkpoint("org/model"))

    assert resolved == ["org/model"]
    assert local_requests == [("org/model", True)]
    assert builder.checkpoint_revision is None
    assert (resolved_directory / "weights.safetensors").resolve() == (
        snapshot_directory / "weights.safetensors"
    ).resolve()


def test_local_nextn_checkpoint_uses_same_resolved_target_and_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_checkpoint = write_synthetic_checkpoint(
        tmp_path / "local-checkpoint", pattern="E", physical_depth=1
    )

    def fail_download(
        model_path: str,
        *,
        revision: str | None,
        local_files_only: bool,
    ) -> str:
        del model_path, revision, local_files_only
        raise AssertionError("local checkpoint must not be downloaded")

    monkeypatch.setattr(engine_builder, "snapshot_download", fail_download)
    builder = make_builder(
        nextn_enabled=True,
        checkpoint_revision="operator-revision",
    )

    resolved_directory = Path(builder.resolve_checkpoint(str(local_checkpoint)))
    builder.checkpoint_dir = str(resolved_directory)
    overrides = {"revision": "operator-revision"}
    builder.adjust_overrides(overrides)

    assert resolved_directory != local_checkpoint
    assert (resolved_directory / "weights.safetensors").resolve() == (
        local_checkpoint / "weights.safetensors"
    ).resolve()
    assert overrides["revision"] == "operator-revision"
    assert overrides["speculative_draft_model_path"] == str(resolved_directory)
    assert overrides["speculative_draft_model_revision"] == "operator-revision"


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
