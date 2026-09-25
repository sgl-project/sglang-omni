# SPDX-License-Identifier: Apache-2.0
"""Native config loading must not import checkpoint Python through HF blob links."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components import audio_encoder, image_encoder
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig


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
