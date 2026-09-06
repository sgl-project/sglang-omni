# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


@pytest.mark.parametrize("size", ["0.6B", "1.7B"])
def test_custom_voice_config_uses_engine_checkpoint_metadata(
    tmp_path, monkeypatch, size
) -> None:
    root = tmp_path / f"Qwen3-TTS-12Hz-{size}-Base"
    engine = tmp_path / "renamed-checkpoint"
    root.mkdir()
    engine.mkdir()
    (root / "config.json").write_text('{"tts_model_type": "base"}')
    metadata = {
        "tts_model_type": "CustomVoice",
        "talker_config": {"spk_id": {"First": 21, "Second": 34}},
    }
    (engine / "config.json").write_text(json.dumps(metadata))
    download = Mock(side_effect=AssertionError("Local checkpoint must not use HF"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    config = Qwen3TTSPipelineConfig(model_path=str(root))
    config.stages[1].factory = config.stages[1].factory.model_copy(
        update={"model_path": str(engine)}
    )
    monkeypatch.setattr(
        Qwen3TTSPipelineConfig,
        "stage_factory_kwargs",
        lambda self, name: {"model_path": str(root)},
    )

    custom_voice_config = config.resolve_custom_voice_config()

    assert custom_voice_config.speakers == ("First", "Second")
    assert custom_voice_config.task_type == "CustomVoice"
    download.assert_not_called()


def test_custom_voice_config_reads_hub_checkpoint_metadata(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tts_model_type": "custom_voice",
                "talker_config": {"spk_id": {"speaker": 7}},
            }
        )
    )
    download = Mock(return_value=str(config_path))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    config = Qwen3TTSPipelineConfig(model_path="org/model")

    assert config.resolve_custom_voice_config().speakers == ("speaker",)
    download.assert_called_once_with(repo_id="org/model", filename="config.json")


@pytest.mark.parametrize("spk_id", [None, {}, {"": 1}])
def test_custom_voice_config_rejects_missing_or_invalid_speakers(
    tmp_path, spk_id
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "tts_model_type": "custom_voice",
                "talker_config": {"spk_id": spk_id},
            }
        )
    )
    config = Qwen3TTSPipelineConfig(model_path=str(tmp_path))

    with pytest.raises(ValueError, match="spk_id"):
        config.resolve_custom_voice_config()


@pytest.mark.parametrize(
    ("model_type", "directory", "uploaded_voices"),
    [
        ("base", "Qwen3-TTS-12Hz-0.6B-Base", True),
        ("base", "Qwen3-TTS-12Hz-1.7B-Base", True),
        ("voice_design", "Qwen3-TTS-12Hz-1.7B-VoiceDesign", False),
        ("base", "renamed-checkpoint", False),
    ],
)
def test_non_custom_voice_config_preserves_existing_hooks(
    tmp_path, model_type, directory, uploaded_voices
) -> None:
    checkpoint = tmp_path / directory
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"tts_model_type": model_type}))
    config = Qwen3TTSPipelineConfig(model_path=str(checkpoint))

    assert config.resolve_custom_voice_config() is None
    assert config.requires_uploaded_voice_for_named_voice() is uploaded_voices
    assert config.supports_uploaded_voice_references() is uploaded_voices
