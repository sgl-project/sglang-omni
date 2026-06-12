# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from sglang_omni.cli.serve import (
    apply_cuda_graph_cli_overrides,
    apply_encoder_mem_reserve_cli_override,
    apply_mem_fraction_cli_overrides,
    apply_parallelism_cli_overrides,
    apply_partial_start_cli_overrides,
    apply_thinker_server_args_cli_overrides,
    apply_torch_compile_cli_overrides,
)
from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.ming_omni.config import (
    MingOmniPipelineConfig,
    MingOmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.registry import (
    PIPELINE_CONFIG_REGISTRY,
    import_pipeline_configs,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config: PipelineConfig, name: str) -> dict[str, object]:
    return dict(_stage(config, name).factory_args.get("server_args_overrides") or {})


def test_ming_config_manager_resolves_top_level_hf_architecture(monkeypatch) -> None:
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        )

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        fake_from_pretrained,
    )

    config_manager = ConfigManager.from_model_path("inclusionAI/Ming-flash-omni-2.0")

    assert calls == [(("inclusionAI/Ming-flash-omni-2.0",), {})]
    assert isinstance(config_manager.config, MingOmniSpeechPipelineConfig)
    assert config_manager.config.model_path == "inclusionAI/Ming-flash-omni-2.0"
    assert config_manager.config.terminal_stages == ["decode", "talker"]


def test_ming_config_manager_resolves_single_architecture_attribute(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architecture="BailingMM2NativeForConditionalGeneration"
        ),
    )

    config_manager = ConfigManager.from_model_path("dummy-ming")

    assert isinstance(config_manager.config, MingOmniSpeechPipelineConfig)


def test_ming_registry_keeps_thinker_architecture_alias() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("BailingMM2NativeForConditionalGeneration")
        is MingOmniSpeechPipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("BailingMoeV2ForCausalLM")
        is MingOmniSpeechPipelineConfig
    )


def test_ming_hf_config_registration_does_not_import_thinker() -> None:
    import sys

    from sglang_omni.models.ming_omni import registration

    sys.modules.pop("sglang_omni.models.ming_omni.thinker", None)
    registration._ming_hf_config_registered = False

    registration.register_ming_hf_config()

    assert "sglang_omni.models.ming_omni.thinker" not in sys.modules


def test_ming_text_variant_uses_text_image_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    config_manager = ConfigManager.from_model_path("dummy-ming", variant="text")

    assert isinstance(config_manager.config, MingOmniPipelineConfig)
    assert [stage.name for stage in config_manager.config.stages] == [
        "preprocessing",
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert config_manager.config.terminal_stages == ["decode"]


def test_ming_cli_applies_tp_gpus_and_disable_custom_all_reduce() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=4,
        thinker_gpus="0,1,2,3",
        talker_gpu=None,
        code2wav_gpu=None,
    )

    thinker = _stage(config, "thinker")
    assert thinker.tp_size == 4
    assert thinker.gpu == [0, 1, 2, 3]
    assert (
        _server_args_overrides(config, "thinker")["disable_custom_all_reduce"] is True
    )


def test_ming_cli_applies_image_encoder_tp_and_gpus() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        image_encoder_tp_size=2,
        image_encoder_gpus="4,5",
        talker_gpu=None,
        code2wav_gpu=None,
    )

    image_encoder = _stage(config, "image_encoder")
    assert image_encoder.tp_size == 2
    assert image_encoder.gpu == [4, 5]
    assert image_encoder.parallelism.tp == 2


def test_ming_cli_image_encoder_tp1_collapses_to_scalar_gpu() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        image_encoder_tp_size=1,
        image_encoder_gpus="4",
        talker_gpu=None,
        code2wav_gpu=None,
    )

    assert _stage(config, "image_encoder").gpu == 4


def test_ming_cli_rejects_image_encoder_gpu_count_mismatch() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            image_encoder_tp_size=2,
            image_encoder_gpus="4",
            talker_gpu=None,
            code2wav_gpu=None,
        )


def test_ming_cli_leaves_image_encoder_untouched_when_flags_omitted() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")
    before_tp = _stage(config, "image_encoder").tp_size
    before_gpu = _stage(config, "image_encoder").gpu

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
    )

    image_encoder = _stage(config, "image_encoder")
    assert image_encoder.tp_size == before_tp
    assert image_encoder.gpu == before_gpu


def test_ming_cli_applies_tp_server_args_for_config_mutated_tp() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
    )

    assert (
        _server_args_overrides(config, "thinker")["disable_custom_all_reduce"] is True
    )


def test_ming_cli_applies_thinker_sglang_server_args() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )
    apply_thinker_server_args_cli_overrides(
        config,
        cpu_offload_gb=0,
        quantization="fp8",
    )

    thinker = _stage(config, "thinker")
    overrides = _server_args_overrides(config, "thinker")
    assert thinker.runtime.sglang_server_args.mem_fraction_static == 0.80
    assert overrides["cpu_offload_gb"] == 0
    assert overrides["quantization"] == "fp8"


@pytest.mark.parametrize(
    "config_cls",
    [MingOmniPipelineConfig, MingOmniSpeechPipelineConfig],
)
def test_ming_cli_rejects_encoder_mem_reserve_with_stable_message(config_cls) -> None:
    config = config_cls(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match=(f"--encoder-mem-reserve is not supported by {config_cls.__name__}"),
    ):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.05,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )

    assert "encoder_mem_reserve" not in _stage(config, "thinker").factory_args


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (
            MingOmniPipelineConfig,
            {"mem_fraction_static": 0.80, "thinker_mem_fraction_static": None},
        ),
        (
            MingOmniPipelineConfig,
            {"mem_fraction_static": None, "thinker_mem_fraction_static": 0.80},
        ),
        (
            MingOmniSpeechPipelineConfig,
            {"mem_fraction_static": 0.80, "thinker_mem_fraction_static": None},
        ),
        (
            MingOmniSpeechPipelineConfig,
            {"mem_fraction_static": None, "thinker_mem_fraction_static": 0.80},
        ),
    ],
)
def test_ming_cli_rejects_encoder_mem_reserve_as_unsupported_before_exclusive_flags(
    config_cls,
    kwargs,
) -> None:
    config = config_cls(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match=(f"--encoder-mem-reserve is not supported by {config_cls.__name__}"),
    ):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.05,
            **kwargs,
        )

    assert "encoder_mem_reserve" not in _stage(config, "thinker").factory_args


def test_ming_speech_cli_rejects_talker_thinker_gpu_collision() -> None:
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="talker.*thinker.*colli"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=2,
            thinker_gpus="0,1",
            talker_gpu=None,
            code2wav_gpu=None,
        )


def test_ming_cli_talker_gpu_targets_talker_stage() -> None:
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=3,
        code2wav_gpu=None,
    )

    assert _stage(config, "thinker").gpu == [0, 1]
    assert _stage(config, "talker").gpu == 3


@pytest.mark.parametrize("mode", ["on", "off"])
def test_ming_cli_rejects_talker_partial_start_before_mutating_factory_args(
    mode: str,
) -> None:
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match="--talker-partial-start currently supports only Qwen3-Omni talker",
    ):
        apply_partial_start_cli_overrides(config, talker_partial_start=mode)

    assert "enable_partial_start" not in _stage(config, "talker").factory_args


def test_ming_text_cli_rejects_talker_gpu_with_stable_message() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match="--talker-gpu is not supported by MingOmniPipelineConfig",
    ):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=3,
            code2wav_gpu=None,
        )


@pytest.mark.parametrize(
    "config_cls",
    [MingOmniPipelineConfig, MingOmniSpeechPipelineConfig],
)
def test_ming_cli_rejects_code2wav_gpu_with_stable_message(config_cls) -> None:
    config = config_cls(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match=f"--code2wav-gpu is not supported by {config_cls.__name__}",
    ):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=5,
        )


@pytest.mark.parametrize(
    "config_cls",
    [MingOmniPipelineConfig, MingOmniSpeechPipelineConfig],
)
def test_ming_cli_rejects_talker_cuda_graph_with_stable_message(config_cls) -> None:
    config = config_cls(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match=f"--talker-cuda-graph is not supported by {config_cls.__name__}",
    ):
        apply_cuda_graph_cli_overrides(
            config,
            thinker_cuda_graph="default",
            talker_cuda_graph="on",
        )


@pytest.mark.parametrize(
    "config_cls,kwargs,flag_name",
    [
        (
            MingOmniPipelineConfig,
            {"talker_torch_compile": "on", "talker_torch_compile_max_bs": None},
            "--talker-torch-compile",
        ),
        (
            MingOmniSpeechPipelineConfig,
            {"talker_torch_compile": "on", "talker_torch_compile_max_bs": None},
            "--talker-torch-compile",
        ),
        (
            MingOmniPipelineConfig,
            {"talker_torch_compile": "default", "talker_torch_compile_max_bs": 2},
            "--talker-torch-compile-max-bs",
        ),
        (
            MingOmniSpeechPipelineConfig,
            {"talker_torch_compile": "default", "talker_torch_compile_max_bs": 2},
            "--talker-torch-compile-max-bs",
        ),
    ],
)
def test_ming_cli_rejects_talker_torch_compile_with_stable_message(
    config_cls,
    kwargs,
    flag_name,
) -> None:
    config = config_cls(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match=f"{flag_name} is not supported by {config_cls.__name__}",
    ):
        apply_torch_compile_cli_overrides(
            config,
            thinker_torch_compile="default",
            thinker_torch_compile_max_bs=None,
            **kwargs,
        )


def test_qwen_cli_talker_gpu_still_targets_talker_ar_stage() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=4,
        code2wav_gpu=5,
    )

    assert _stage(config, "talker_ar").gpu == 4
    assert _stage(config, "code2wav").gpu == 5


def test_registry_rejects_duplicate_architecture_aliases(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "fake_models"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    for model_name, architecture in (
        ("model_a", "FakeArchA"),
        ("model_b", "FakeArchB"),
    ):
        model_dir = package_dir / model_name
        model_dir.mkdir()
        (model_dir / "__init__.py").write_text("", encoding="utf-8")
        (model_dir / "config.py").write_text(
            "\n".join(
                [
                    "from typing import ClassVar",
                    "from sglang_omni.config import PipelineConfig",
                    "",
                    f"class FakeConfig(PipelineConfig):",
                    f"    architecture: ClassVar[str] = {architecture!r}",
                    "    architecture_aliases: ClassVar[tuple[str, ...]] = ("
                    "'SharedArch',)",
                    "",
                    "EntryClass = FakeConfig",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    import_pipeline_configs.cache_clear()

    with pytest.raises(ValueError, match="SharedArch"):
        import_pipeline_configs("fake_models", "config")

    import_pipeline_configs.cache_clear()


def test_omni_serve_builds_ming_text_config_without_launching(monkeypatch) -> None:
    from typer.testing import CliRunner

    from sglang_omni.cli import app

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", fake_launch_server)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            "inclusionAI/Ming-flash-omni-2.0",
            "--text-only",
            "--thinker-tp-size",
            "4",
            "--thinker-gpus",
            "0,1,2,3",
            "--cpu-offload-gb",
            "0",
            "--mem-fraction-static",
            "0.8",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-name",
            "ming-omni",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert type(config).__name__ == "MingOmniPipelineConfig"
    assert _stage(config, "thinker").tp_size == 4
    assert _stage(config, "thinker").gpu == [0, 1, 2, 3]
    overrides = _server_args_overrides(config, "thinker")
    assert overrides["cpu_offload_gb"] == 0
    assert overrides["disable_custom_all_reduce"] is True
    assert (
        _stage(config, "thinker").runtime.sglang_server_args.mem_fraction_static == 0.8
    )
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8000
    assert captured["kwargs"]["model_name"] == "ming-omni"


def test_omni_serve_builds_ming_speech_config_by_default(monkeypatch) -> None:
    from typer.testing import CliRunner

    from sglang_omni.cli import app

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", fake_launch_server)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            "inclusionAI/Ming-flash-omni-2.0",
            "--thinker-tp-size",
            "2",
            "--thinker-gpus",
            "0,1",
            "--talker-gpu",
            "3",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-name",
            "ming-omni",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert isinstance(config, MingOmniSpeechPipelineConfig)
    assert config.terminal_stages == ["decode", "talker"]
    assert _stage(config, "thinker").tp_size == 2
    assert _stage(config, "thinker").gpu == [0, 1]
    assert _stage(config, "talker").gpu == 3
    assert (
        _server_args_overrides(config, "thinker")["disable_custom_all_reduce"] is True
    )
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8000
    assert captured["kwargs"]["model_name"] == "ming-omni"
