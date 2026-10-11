# SPDX-License-Identifier: Apache-2.0
"""Native config loading must not import checkpoint Python through HF blob links."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import Result
from pydantic import JsonValue, ValidationError
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from typer.testing import CliRunner

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.minicpm_o import native_stages, stages
from sglang_omni.models.minicpm_o.components import audio_encoder, image_encoder
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.engine_builder import MiniCPMOThinkerEngineBuilder
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.minicpm_o.native_config import (
    TALKER_CONTEXT_LENGTH,
    THINKER_CONTEXT_LENGTH,
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexVision,
)
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.scheduling.session import BatchedSessionHooks
from sglang_omni.scheduling.stage_kv_budget import consume_stage_kv_cache_bytes
from sglang_omni.utils.gpu_memory import GpuDeviceInfo


class ConfigLoaded(Exception):
    """Stop at the configuration boundary before allocating any model or GPU."""


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    config = {
        "model_type": "minicpmo",
        "architectures": ["MiniCPMO"],
        "auto_map": {"AutoConfig": "configuration_minicpmo.MiniCPMOConfig"},
        "attention_bias": False,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 8,
        "num_hidden_layers": 1,
        "vision_config": {"hidden_size": 32},
        "audio_config": {"d_model": 32},
        "tts_config": {"hidden_size": 16},
    }
    files = {
        "config.json": json.dumps(config),
        "configuration_minicpmo.py": "from .modeling_navit_siglip import Config\n",
        "modeling_navit_siglip.py": "class Config: pass\n",
    }
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for name, contents in files.items():
        blob = blobs / hashlib.sha256(contents.encode()).hexdigest()
        blob.write_text(contents)
        (snapshot / name).symlink_to(blob)
    return snapshot


@pytest.mark.parametrize("encoder", ["image", "audio"])
def test_encoder_loads_native_config_from_snapshot_links(
    encoder: str, snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop_before_weights(*args):
        raise ConfigLoaded

    if encoder == "image":
        monkeypatch.setattr(image_encoder, "init_sglang_tp", stop_before_weights)
        constructor = image_encoder.MiniCPMOImageEncoder
    else:
        monkeypatch.setattr(audio_encoder, "audio_config_object", stop_before_weights)
        constructor = audio_encoder.MiniCPMOAudioEncoder

    with pytest.raises(ConfigLoaded):
        constructor(str(snapshot), device="cpu")


def test_native_config_preserves_component_dictionaries(snapshot: Path) -> None:
    config = MiniCPMOConfig.from_pretrained(snapshot)
    raw = json.loads((snapshot / "config.json").read_text())
    for name in ("vision_config", "audio_config", "tts_config"):
        assert getattr(config, name) == raw[name]
    assert image_encoder.vision_config_object(config).hidden_size == 32
    assert audio_encoder.audio_config_object(config).d_model == 32
    assert config.get_text_config().hidden_size == 64


@pytest.mark.parametrize("stage", ["thinker", "talker"])
@pytest.mark.parametrize("trust_override", [None, False, True])
def test_engine_factory_resolves_native_config_before_server_args(
    stage: str,
    trust_override: bool | None,
    snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = dict(
        CONFIG_MAPPING._extra_content
    )  # noqa: leading-underscore  # upstream name
    mapping.pop("minicpmo", None)
    monkeypatch.setattr(CONFIG_MAPPING, "_extra_content", mapping)
    monkeypatch.setattr(stages, "resolved_view", lambda args: args)

    def build_overrides(*, server_args_overrides=None, **defaults):
        return {**defaults, **(server_args_overrides or {})}

    def build_server_args(model_path, **kwargs):
        trust = kwargs.get("trust_remote_code", True)
        if trust_override is True:
            assert trust is True, "An explicit remote-code override must be preserved"
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust)
            assert isinstance(config, MiniCPMOConfig)
        raise ConfigLoaded

    monkeypatch.setattr(stages, "build_generation_batch_overrides", build_overrides)
    monkeypatch.setattr(
        stages, "validate_generation_batch_policy", lambda **kwargs: None
    )
    monkeypatch.setattr(stages, "build_sglang_server_args", build_server_args)
    factory = getattr(stages, f"create_sglang_{stage}_executor_from_config")
    overrides = {} if trust_override is None else {"trust_remote_code": trust_override}
    with pytest.raises(ConfigLoaded):
        factory(str(snapshot), server_args_overrides=overrides)


@pytest.fixture
def stub_stage_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AutoTokenizer",
        "AutoProcessor",
        "LogMelFilterBank",
        "MiniCPMOAudioEncoder",
        "MiniCPMOImageEncoder",
        "MiniCPMOCode2Wav",
        "MiniCPMOVocoderRuntime",
    ):
        monkeypatch.setattr(native_stages, name, Mock())
    monkeypatch.setattr(
        native_stages,
        "PerceptionHooks",
        Mock(
            return_value=Mock(spec=native_stages.PerceptionHooks, gather_window_ms=0.0)
        ),
    )
    monkeypatch.setattr(
        native_stages, "SpeechHooks", Mock(return_value=BatchedSessionHooks())
    )


@pytest.mark.parametrize(
    ("settings", "sessions", "state_bytes", "thinker", "talker", "talker_compile"),
    [
        ("", 2, 2 << 30, 3, 3, False),
        (
            "max_sessions: 64\nspeech_state_bytes_per_session: 1024\nstages:\n"
            "  talker:\n    engine:\n      max_running_requests: 3\n"
            "      enable_torch_compile: true\n",
            64,
            1024,
            65,
            3,
            True,
        ),
    ],
)
def test_duplex_yaml_builds_session_stages(
    settings: str,
    sessions: int,
    state_bytes: int,
    thinker: int,
    talker: int,
    talker_compile: bool,
    tmp_path: Path,
    stub_stage_models: None,
) -> None:
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference")
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: MiniCPMODuplexPipelineConfig\n"
        f"model_path: unused\nreference_audio: {reference_path}\n" + settings
    )
    config = ConfigManager.from_file(str(config_path)).config
    native_stages.MiniCPMOCode2Wav.return_value.default_prompt_wav = str(reference_path)
    perception = native_stages.create_perception_scheduler(
        config.model_path,
        device="cpu",
        dtype="float32",
        **config.stage_factory_kwargs("perception"),
    )
    speech = native_stages.create_speech_scheduler(
        config.model_path, device="cpu", **config.stage_factory_kwargs("speech")
    )
    for scheduler in (perception, speech):
        assert scheduler.max_open_sessions == sessions
        assert scheduler.max_concurrency == 1
    assert speech.max_state_bytes_per_session == state_bytes
    assert build_realtime_deployment(Mock(), config).max_connections == sessions
    for stage_name, factory, expected, compile_enabled in (
        ("thinker", native_stages.create_thinker_scheduler, thinker, False),
        (
            "talker",
            stages.create_sglang_session_talker_executor_from_config,
            talker,
            talker_compile,
        ),
    ):
        kwargs = apply_typed_stage_kwargs(
            factory,
            config.stage_factory_kwargs(stage_name),
            resolve_stage_typed_kwargs(config.stage_named(stage_name)),
            stage_name=stage_name,
        )
        assert kwargs["server_args_overrides"]["max_running_requests"] == expected
        assert (
            kwargs["server_args_overrides"]["enable_torch_compile"] is compile_enabled
        )
    for encoder in (
        native_stages.MiniCPMOAudioEncoder,
        native_stages.MiniCPMOImageEncoder,
    ):
        encoder.assert_called_once_with("unused", device="cpu", dtype="float32")
    assert (
        native_stages.PerceptionHooks.call_args.kwargs["image_encoder"]
        is native_stages.MiniCPMOImageEncoder.return_value
    )
    native_stages.PerceptionHooks.return_value.warm_up.assert_called_once_with(sessions)
    native_stages.AutoProcessor.from_pretrained.assert_called_once()


@pytest.mark.parametrize(
    ("settings", "sessions", "thinker_tokens_per_session", "derived_talker"),
    [
        ("max_sessions: 64\n", 64, THINKER_CONTEXT_LENGTH, True),
        (
            "stages:\n  thinker:\n    gpu_memory_fraction: 0.4\n"
            "  talker:\n    engine:\n      max_total_tokens: 1000\n",
            2,
            None,
            False,
        ),
        (
            "stages:\n  thinker:\n    engine:\n      context_length: 16384\n"
            "  talker:\n    engine:\n      mem_fraction_static: 0.1\n",
            2,
            16384,
            False,
        ),
    ],
)
def test_duplex_engine_memory_follows_max_sessions(
    settings: str,
    sessions: int,
    thinker_tokens_per_session: int | None,
    derived_talker: bool,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: MiniCPMODuplexPipelineConfig\nmodel_path: unused\n" + settings
    )
    config = ConfigManager.from_file(str(config_path)).config
    request_slots = sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
    thinker = config.stage_factory_kwargs("thinker")
    talker = config.stage_factory_kwargs("talker")["server_args_overrides"]
    assert thinker["server_args_overrides"]["max_running_requests"] == request_slots
    assert talker["max_running_requests"] == request_slots
    if thinker_tokens_per_session is None:
        assert "kv_cache_tokens" not in thinker
    else:
        assert thinker["kv_cache_tokens"] == request_slots * thinker_tokens_per_session
    if derived_talker:
        assert talker["max_total_tokens"] == request_slots * TALKER_CONTEXT_LENGTH
    else:
        assert "max_total_tokens" not in talker


def test_thinker_kv_pool_holds_the_derived_tokens_within_its_card_share(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MiniCPMOThinkerEngineBuilder,
        "build",
        lambda self, *args, **kwargs: consume_stage_kv_cache_bytes(),
    )
    # note (Junnan Li): The snapshot thinker has 1 layer of 8 KV heads of width 8 in bfloat16.
    kv_cache_bytes = 1000 * 2 * 1 * 8 * 8 * 2
    assert (
        native_stages.create_thinker_scheduler(
            str(snapshot),
            device="cpu",
            kv_cache_tokens=1000,
            total_gpu_memory_fraction=0.4,
        )
        == kv_cache_bytes
    )
    monkeypatch.setattr(
        native_stages,
        "get_gpu_device_info",
        lambda gpu_id: GpuDeviceInfo(
            logical_gpu_id=gpu_id,
            device_id=None,
            name=None,
            total_memory_bytes=2 * kv_cache_bytes,
        ),
    )
    with pytest.raises(ValueError, match="max_sessions needs"):
        native_stages.create_thinker_scheduler(
            str(snapshot),
            device="cpu",
            kv_cache_tokens=1000,
            total_gpu_memory_fraction=0.4,
        )


@pytest.mark.parametrize(
    ("cli_flags", "disable_cuda_graph"),
    [
        ([], False),
        (
            [
                "--thinker.engine.disable_cuda_graph",
                "true",
                "--talker.engine.disable_cuda_graph",
                "true",
            ],
            True,
        ),
    ],
)
def test_duplex_settings_select_decode_graphs_without_compile(
    cli_flags: list[str], disable_cuda_graph: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = dict(
        CONFIG_MAPPING._extra_content
    )  # noqa: leading-underscore  # upstream name
    monkeypatch.setattr(CONFIG_MAPPING, "_extra_content", mapping)
    manager = ConfigManager(MiniCPMODuplexPipelineConfig(model_path="unused"))
    config = manager.merge_config(manager.parse_extra_args(cli_flags))
    talker_server_args: dict[str, JsonValue] = {}

    def capture_server_args(model_path: str, **server_args: JsonValue) -> None:
        talker_server_args.update(server_args)
        raise ConfigLoaded

    monkeypatch.setattr(stages, "build_sglang_server_args", capture_server_args)
    overrides = {
        stage_name: apply_typed_stage_kwargs(
            factory,
            config.stage_factory_kwargs(stage_name),
            resolve_stage_typed_kwargs(config.stage_named(stage_name)),
            stage_name=stage_name,
        )["server_args_overrides"]
        for stage_name, factory in (
            ("thinker", native_stages.create_thinker_scheduler),
            ("talker", stages.create_sglang_session_talker_executor_from_config),
        )
    }
    for stage_overrides in overrides.values():
        assert stage_overrides["enable_torch_compile"] is False
        assert ("disable_cuda_graph" in stage_overrides) is disable_cuda_graph
    with pytest.raises(ConfigLoaded):
        stages.create_sglang_session_talker_executor_from_config(
            "unused", device="cpu", server_args_overrides=overrides["talker"]
        )
    assert talker_server_args["disable_cuda_graph"] is disable_cuda_graph
    assert (
        "disable_cuda_graph"
        not in MiniCPMOThinkerEngineBuilder().generation_defaults(dtype="bfloat16")
    )


def test_minicpmo_configs_load_without_sglang(tmp_path: Path) -> None:
    config_path = tmp_path / "duplex.yaml"
    script = """
import sys
from pathlib import Path
sys.modules["sglang"] = None
from sglang_omni.config.manager import ConfigManager
for name in ("MiniCPMODuplexPipelineConfig", "MiniCPMOPipelineConfig", "MiniCPMOSpeechPipelineConfig"):
    Path(sys.argv[1]).write_text(f"config_cls: {name}\\nmodel_path: unused\\n")
    config = ConfigManager.from_file(sys.argv[1]).config
    assert type(config).__name__ == name
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "limits",
    [
        {"max_slice_nums": 4, "max_slice_nums_limit": 2},
        {"max_tiles_per_unit": 9, "max_slice_nums_limit": 9},
    ],
)
def test_vision_limits_must_fit_one_frame(limits: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        MiniCPMODuplexVision(**limits)


def test_duplex_deployment_grants_images_by_slice_count() -> None:
    capabilities = build_realtime_deployment(
        Mock(), MiniCPMODuplexPipelineConfig(model_path="unused")
    ).capabilities
    assert capabilities.input_modalities == ("audio", "image")
    assert capabilities.image_frames_per_unit == (4, 3, 2, 2, 1, 1, 1, 1, 1)
    assert capabilities.default_max_slice_nums == 1


def invoke_variant_session(
    monkeypatch: pytest.MonkeyPatch, serve_arguments: list[str]
) -> tuple[Result, Mock]:
    monkeypatch.setattr(
        "sglang_omni.config.manager.resolve_config_cls_for_model_path",
        lambda model_path: MiniCPMOSpeechPipelineConfig,
    )
    launch_server = Mock()
    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", launch_server)
    result = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"}).invoke(
        app,
        ["serve", "--model-path", "unused", "--variant", "session", *serve_arguments],
    )
    return result, launch_server


def test_variant_session_serves_duplex(monkeypatch: pytest.MonkeyPatch) -> None:
    result, launch_server = invoke_variant_session(monkeypatch, ["--enable-realtime"])

    assert result.exit_code == 0, result.output
    assert type(launch_server.call_args.args[0]) is MiniCPMODuplexPipelineConfig


def test_variant_session_requires_enable_realtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, launch_server = invoke_variant_session(monkeypatch, [])

    assert result.exit_code == 2
    assert "--enable-realtime" in result.output
    launch_server.assert_not_called()


def test_duplex_speech_settings_reach_the_vocoder(
    tmp_path: Path, stub_stage_models: None
) -> None:
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference")
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: MiniCPMODuplexPipelineConfig\nmodel_path: unused\n"
        "speech:\n  dtype: float16\n  enable_dit_torch_compile: true\n"
        "  n_timesteps: 6\n"
    )
    config = ConfigManager.from_file(str(config_path)).config
    native_stages.MiniCPMOCode2Wav.return_value.default_prompt_wav = str(reference_path)
    native_stages.create_speech_scheduler(
        config.model_path, device="cpu", **config.stage_factory_kwargs("speech")
    )
    codec_kwargs = native_stages.MiniCPMOCode2Wav.call_args.kwargs
    assert codec_kwargs["dtype"] == "float16"
    assert codec_kwargs["enable_dit_torch_compile"] is True
    assert codec_kwargs["n_timesteps"] == 6
    native_stages.MiniCPMOVocoderRuntime.return_value.warm_up.assert_called_once_with(
        b"reference"
    )
    runtime_kwargs = native_stages.MiniCPMOVocoderRuntime.call_args.kwargs
    assert runtime_kwargs["max_open_sessions"] == config.max_sessions
