# SPDX-License-Identifier: Apache-2.0
"""Ming 3 audio placement defaults and explicit tensor parallelism."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.ming3_audio.config import Ming3AudioPipelineConfig


@dataclass
class FakeServerArgs:
    tp_size: int


@dataclass
class FactoryCapture:
    tp_size: int | None = None
    infrastructure_server_args: FakeServerArgs | None = None


def stage_named(config: Ming3AudioPipelineConfig, name: str) -> StageConfig:
    return next(stage for stage in config.stages if stage.name == name)


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
