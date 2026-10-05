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
from pydantic import ValidationError
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.minicpm_o import native_stages, stages
from sglang_omni.models.minicpm_o.components import audio_encoder, image_encoder
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexVision,
)
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.scheduling.session import BatchedSessionHooks, SessionHooks


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
def stub_stage_models(monkeypatch: pytest.MonkeyPatch) -> SessionHooks:
    hooks = SessionHooks()
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
    ("settings", "sessions", "state_bytes", "thinker", "talker"),
    [
        ("", 2, 2 << 30, 3, 3),
        (
            "max_sessions: 64\nspeech_state_bytes_per_session: 1024\nstages:\n"
            "  talker:\n    engine:\n      max_running_requests: 3\n",
            64,
            1024,
            65,
            3,
        ),
    ],
)
def test_duplex_yaml_builds_session_stages(
    settings: str,
    sessions: int,
    state_bytes: int,
    thinker: int,
    talker: int,
    tmp_path: Path,
    stub_stage_models: SessionHooks,
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
    for stage_name, factory, expected in (
        ("thinker", native_stages.create_thinker_scheduler, thinker),
        ("talker", stages.create_sglang_session_talker_executor_from_config, talker),
    ):
        kwargs = apply_typed_stage_kwargs(
            factory,
            config.stage_factory_kwargs(stage_name),
            resolve_stage_typed_kwargs(config.stage_named(stage_name)),
            stage_name=stage_name,
        )
        assert kwargs["server_args_overrides"]["max_running_requests"] == expected
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
