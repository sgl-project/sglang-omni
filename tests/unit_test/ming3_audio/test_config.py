# SPDX-License-Identifier: Apache-2.0
"""Ming 3 audio placement defaults and explicit tensor parallelism."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sglang_omni.config import runtime
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.config.sources import dump_user_config
from sglang_omni.models.ming3_audio.config import (
    Ming3AudioPipelineConfig,
    Ming3TalkerFactoryArgs,
)


@dataclass
class FakeServerArgs:
    tp_size: int


@dataclass
class FactoryCapture:
    tp_size: int | None = None
    infrastructure_server_args: FakeServerArgs | None = None


def stage_named(config: Ming3AudioPipelineConfig, name: str) -> StageConfig:
    return next(stage for stage in config.stages if stage.name == name)


def fake_talker_factory(
    model_path: str,
    *,
    max_decode_steps: int,
    seed: int,
    enable_cuda_graph: bool,
    enable_text_segmentation: bool,
    max_semantic_length: int,
    max_fragment_characters: int,
    max_fragments: int,
    max_input_characters: int,
    max_total_decode_steps: int,
    gpu_id: int | None = None,
) -> tuple[str, int, int, bool, bool, int, int, int, int, int, int | None]:
    return (
        model_path,
        max_decode_steps,
        seed,
        enable_cuda_graph,
        enable_text_segmentation,
        max_semantic_length,
        max_fragment_characters,
        max_fragments,
        max_input_characters,
        max_total_decode_steps,
        gpu_id,
    )


def test_talker_segmentation_defaults_are_disabled_and_bounded() -> None:
    talker = stage_named(Ming3AudioPipelineConfig(model_path="dummy"), "talker")

    assert isinstance(talker.factory, FactoryArgs)
    assert talker.factory.model_extra is not None
    assert talker.factory.enable_text_segmentation is False
    assert talker.factory.max_semantic_length == 48
    assert talker.factory.max_fragment_characters == 96
    assert talker.factory.max_fragments == 32
    assert talker.factory.max_input_characters == 8192
    assert talker.factory.max_total_decode_steps == 16384


def test_talker_factory_controls_survive_pipeline_dump_and_rebuild() -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    serialized = config.model_dump()
    serialized_talker = next(
        stage for stage in serialized["stages"] if stage["name"] == "talker"
    )

    expected_factory_arguments = {
        "max_decode_steps": 512,
        "seed": 123,
        "enable_cuda_graph": True,
        "enable_text_segmentation": False,
        "max_semantic_length": 48,
        "max_fragment_characters": 96,
        "max_fragments": 32,
        "max_input_characters": 8192,
        "max_total_decode_steps": 16384,
    }
    assert all(
        serialized_talker["factory"][name] == value
        for name, value in expected_factory_arguments.items()
    )

    rebuilt = Ming3AudioPipelineConfig.model_validate(serialized)
    assert stage_named(rebuilt, "talker").factory.model_extra == (
        expected_factory_arguments
    )


def test_explicit_segmentation_override_survives_user_config_export(
    tmp_path: Path,
) -> None:
    config_manager = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy"))
    parsed_arguments = config_manager.parse_extra_args(
        ["--talker.factory.enable_text_segmentation", "true"]
    )
    config = config_manager.merge_config(parsed_arguments)
    configuration_path = tmp_path / "ming3-export.yaml"
    configuration_path.write_text(
        yaml.safe_dump(dump_user_config(config), sort_keys=False),
        encoding="utf-8",
    )

    rebuilt = ConfigManager.from_file(str(configuration_path)).config

    assert stage_named(rebuilt, "talker").factory.enable_text_segmentation is True
    assert stage_named(rebuilt, "talker").factory.seed == 123
    assert stage_named(rebuilt, "talker").factory.enable_cuda_graph is True


def test_original_talker_factory_arguments_get_new_defaults() -> None:
    serialized = Ming3AudioPipelineConfig(model_path="dummy").model_dump()
    serialized_talker = next(
        stage for stage in serialized["stages"] if stage["name"] == "talker"
    )
    serialized_talker["factory"] = {
        "max_decode_steps": 512,
        "seed": 123,
        "enable_cuda_graph": True,
    }

    rebuilt = Ming3AudioPipelineConfig.model_validate(serialized)
    factory = stage_named(rebuilt, "talker").factory

    assert factory.max_decode_steps == 512
    assert factory.seed == 123
    assert factory.enable_cuda_graph is True
    assert factory.enable_text_segmentation is False
    assert factory.max_semantic_length == 48
    assert factory.max_fragment_characters == 96
    assert factory.max_fragments == 32
    assert factory.max_input_characters == 8192
    assert factory.max_total_decode_steps == 16384


def test_explicit_none_factory_extras_survive_config_normalization() -> None:
    serialized = Ming3AudioPipelineConfig(model_path="dummy").model_dump()
    serialized_talker = next(
        stage for stage in serialized["stages"] if stage["name"] == "talker"
    )
    serialized_talker["factory"]["custom_factory_option"] = None

    rebuilt = Ming3AudioPipelineConfig.model_validate(serialized)
    factory = stage_named(rebuilt, "talker").factory
    extra_factory_arguments = factory.model_extra

    assert extra_factory_arguments is not None
    assert "custom_factory_option" in extra_factory_arguments
    assert extra_factory_arguments["custom_factory_option"] is None
    assert factory.model_extra["custom_factory_option"] is None


def test_config_manager_merge_and_launcher_resolution_keep_talker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "import_string", lambda factory_path: fake_talker_factory
    )
    merged = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy")).merge_config(
        []
    )
    talker = stage_named(merged, "talker")

    assert isinstance(talker.factory, FactoryArgs)
    assert talker.factory.model_extra is not None
    resolved_arguments = resolve_stage_factory_args(talker, merged, gpu_id=1)
    assert fake_talker_factory(**resolved_arguments) == (
        "dummy",
        512,
        123,
        True,
        False,
        48,
        96,
        32,
        8192,
        16384,
        1,
    )


def test_talker_segmentation_off_switch_is_preserved_by_cli_merge() -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    config_manager = ConfigManager(config)
    parsed_arguments = config_manager.parse_extra_args(
        [
            "--talker.factory.enable_text_segmentation",
            "false",
            "--talker.factory.max_fragments",
            "7",
        ]
    )

    merged = config_manager.merge_config(parsed_arguments)
    talker = stage_named(merged, "talker")

    assert talker.factory.enable_text_segmentation is False
    assert talker.factory.max_fragments == 7


def test_cli_can_enable_segmentation_and_stage_factory_receives_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "import_string", lambda factory_path: fake_talker_factory
    )
    config_manager = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy"))
    parsed_arguments = config_manager.parse_extra_args(
        ["--talker.factory.enable_text_segmentation", "true"]
    )
    merged = config_manager.merge_config(parsed_arguments)
    talker = stage_named(merged, "talker")
    resolved_arguments = resolve_stage_factory_args(talker, merged, gpu_id=1)

    assert talker.factory.enable_text_segmentation is True
    assert fake_talker_factory(**resolved_arguments)[4] is True


def test_cli_can_disable_segmentation_over_explicit_pipeline_true() -> None:
    serialized = Ming3AudioPipelineConfig(model_path="dummy").model_dump()
    serialized_talker = next(
        stage for stage in serialized["stages"] if stage["name"] == "talker"
    )
    serialized_talker["factory"]["enable_text_segmentation"] = True
    config = Ming3AudioPipelineConfig.model_validate(serialized)
    config_manager = ConfigManager(config)
    parsed_arguments = config_manager.parse_extra_args(
        ["--talker.factory.enable_text_segmentation", "false"]
    )

    merged = config_manager.merge_config(parsed_arguments)

    assert stage_named(merged, "talker").factory.enable_text_segmentation is False


def test_cli_merge_preserves_and_validates_all_talker_controls() -> None:
    config_manager = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy"))
    parsed_arguments = config_manager.parse_extra_args(
        [
            "--talker.factory.enable_text_segmentation",
            "false",
            "--talker.factory.max_fragments",
            "7",
            "--talker.factory.max_decode_steps",
            "256",
        ]
    )

    merged = config_manager.merge_config(parsed_arguments)
    talker = stage_named(merged, "talker")

    assert talker.factory.enable_text_segmentation is False
    assert talker.factory.max_fragments == 7
    assert talker.factory.max_decode_steps == 256
    assert talker.factory.seed == 123
    assert talker.factory.enable_cuda_graph is True
    assert talker.factory.max_total_decode_steps == 16384


def test_yaml_factory_overrides_round_trip(tmp_path: Path) -> None:
    configuration_path = tmp_path / "ming3.yaml"
    configuration_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Ming3AudioPipelineConfig",
                "model_path": "dummy",
                "stages": {
                    "talker": {
                        "factory": {
                            "enable_text_segmentation": False,
                            "max_fragments": 7,
                            "max_decode_steps": 256,
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = ConfigManager.from_file(str(configuration_path)).config
    talker = stage_named(config, "talker")

    assert talker.factory.enable_text_segmentation is False
    assert talker.factory.max_fragments == 7
    assert talker.factory.max_decode_steps == 256
    assert talker.factory.seed == 123
    assert talker.factory.enable_cuda_graph is True
    assert talker.factory.max_total_decode_steps == 16384


@pytest.mark.parametrize(
    "cli_arguments",
    [
        ["--talker.factory.max_decode_steps", "0"],
        ["--talker.factory.max_semantic_length", "0"],
        ["--talker.factory.max_fragment_characters", "513"],
        ["--talker.factory.max_fragments", "0"],
        ["--talker.factory.max_input_characters", "0"],
        ["--talker.factory.max_total_decode_steps", "0"],
        ["--talker.factory.max_decode_steps", "not-an-integer"],
        ["--talker.factory.enable_text_segmentation", "not-a-boolean"],
    ],
)
def test_invalid_talker_factory_overrides_fail_during_config_resolution(
    cli_arguments: list[str],
) -> None:
    config_manager = ConfigManager(Ming3AudioPipelineConfig(model_path="dummy"))

    with pytest.raises(ValueError):
        config_manager.merge_config(config_manager.parse_extra_args(cli_arguments))


def test_talker_fragment_character_limit_is_validated() -> None:
    with pytest.raises(ValueError, match="max_fragment_characters"):
        Ming3TalkerFactoryArgs(max_fragment_characters=513)


def test_talker_fragment_limit_must_fit_worst_case_decode_budget() -> None:
    with pytest.raises(ValueError, match="conservative decode-budget ceiling of 16"):
        Ming3TalkerFactoryArgs(
            max_decode_steps=1024,
            max_total_decode_steps=16384,
            max_fragments=32,
            enable_text_segmentation=True,
        )


def test_segmentation_off_does_not_enforce_fragment_decode_budget() -> None:
    factory = Ming3TalkerFactoryArgs(
        max_decode_steps=32768,
        max_total_decode_steps=16384,
        max_fragments=32,
        enable_text_segmentation=False,
    )

    assert factory.max_decode_steps == 32768
    assert factory.enable_text_segmentation is False


def test_default_configuration_does_not_pin_thinker_parallelism() -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    thinker = stage_named(config, "thinker")
    assert thinker.tp_size == 1
    assert thinker.gpu is None


def test_thinker_tp_size_places_logical_gpus() -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    merged = ConfigManager(config).merge_config([("thinker.tp_size", "2")])
    assert stage_named(merged, "thinker").tp_size == 2
    assert stage_named(merged, "thinker").gpu == [0, 1]
    assert stage_named(merged, "audio_encoder").gpu == 0
    assert stage_named(merged, "talker").gpu == 1


def test_cli_tp2_configuration_preserves_encoder_and_talker_placement() -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    config_manager = ConfigManager(config)
    cli_arguments = config_manager.parse_extra_args(
        ["--thinker.tp_size", "2", "--thinker.gpu", "[0,1]"]
    )
    merged = config_manager.merge_config(cli_arguments)
    assert stage_named(merged, "thinker").tp_size == 2
    assert stage_named(merged, "thinker").gpu == [0, 1]
    assert stage_named(merged, "audio_encoder").gpu == 0
    assert stage_named(merged, "talker").gpu == 1


@pytest.mark.parametrize(
    ("cli_arguments", "error_message"),
    [
        (["--thinker.tp_size", "2", "--thinker.gpu", "0"], "TP placement"),
        (["--thinker.tp_size", "1", "--thinker.gpu", "[0,1]"], "tp_size=1"),
        (["--thinker.tp_size", "2", "--thinker.gpu", "[0,0]"], "unique"),
    ],
)
def test_invalid_thinker_placement_is_rejected_before_model_loading(
    cli_arguments: list[str], error_message: str
) -> None:
    config = Ming3AudioPipelineConfig(model_path="dummy")
    config_manager = ConfigManager(config)
    parsed_arguments = config_manager.parse_extra_args(cli_arguments)
    with pytest.raises(ValueError, match=error_message):
        config_manager.merge_config(parsed_arguments)


def test_thinker_factory_forwards_tp_size_to_sglang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = pytest.importorskip("sglang_omni.models.ming3_audio.stages")
    factory_capture = FactoryCapture()

    def build_server_args(
        model_path: str, context_length: int | None, **overrides: object
    ) -> FakeServerArgs:
        tp_size = overrides["tp_size"]
        assert isinstance(tp_size, int)
        factory_capture.tp_size = tp_size
        return FakeServerArgs(tp_size=tp_size)

    def create_infrastructure(
        server_args: FakeServerArgs, gpu_id: int, **kwargs: object
    ) -> tuple[str, str, str, str, SimpleNamespace]:
        factory_capture.infrastructure_server_args = server_args
        return (
            "worker",
            "tree_cache",
            "request_pool",
            "token_pool",
            SimpleNamespace(vocab_size=32),
        )

    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda device, gpu_id: SimpleNamespace(index=gpu_id),
    )
    monkeypatch.setattr(stages, "build_sglang_server_args", build_server_args)
    monkeypatch.setattr(stages, "create_sglang_infrastructure", create_infrastructure)
    monkeypatch.setattr(
        stages, "get_tokenizer", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(stages, "SGLangOutputProcessor", lambda: "processor")
    monkeypatch.setattr(
        stages,
        "Ming3AudioModelRunner",
        lambda worker, processor: "model_runner",
    )
    monkeypatch.setattr(stages, "OmniScheduler", lambda **kwargs: kwargs)

    stages.create_thinker_executor(
        "dummy",
        max_seq_len=8192,
        enable_async_decode=False,
        gpu_id=0,
        tp_size=2,
    )

    assert factory_capture.tp_size == 2
    assert factory_capture.infrastructure_server_args is not None
    assert factory_capture.infrastructure_server_args.tp_size == 2
