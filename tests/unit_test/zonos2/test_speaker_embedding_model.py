# SPDX-License-Identifier: Apache-2.0
"""Tests for selecting the ZONOS2 speaker-embedding checkpoint."""

import pytest

pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

import torch  # noqa: E402

import sglang_omni.models.zonos2.components.speaker_encoder as speaker_encoder  # noqa: E402
from sglang_omni.config.runtime import resolve_stage_typed_kwargs  # noqa: E402
from sglang_omni.config.schema import StageConfig  # noqa: E402
from sglang_omni.models.zonos2.components.speaker_encoder import (  # noqa: E402
    Qwen3SpeakerEmbedding,
)
from sglang_omni.models.zonos2.stages import (  # noqa: E402
    create_speaker_encode_executor,
)


class FakeAutoModel:
    def __init__(self) -> None:
        self.received_model_names_or_paths: list[str] = []

    def from_pretrained(
        self,
        model_name_or_path: str,
        *,
        trust_remote_code: bool,
    ) -> "FakeModel":
        self.received_model_names_or_paths.append(model_name_or_path)
        assert trust_remote_code is True
        return FakeModel()


class FakeModel:
    def to(self, device: str) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self


@pytest.mark.parametrize(
    ("configured_model_name_or_path", "expected_model_name_or_path"),
    [
        (None, Qwen3SpeakerEmbedding.MODEL_NAME),
        ("/models/qwen3_voice_embedding", "/models/qwen3_voice_embedding"),
    ],
)
def test_speaker_encoder_uses_the_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
    configured_model_name_or_path: str | None,
    expected_model_name_or_path: str,
) -> None:
    fake_auto_model = FakeAutoModel()
    monkeypatch.setattr(speaker_encoder, "AutoModel", fake_auto_model)

    encoder = speaker_encoder.SpeakerEncoder(
        device="cpu",
        speaker_embedding_model_name_or_path=configured_model_name_or_path,
    )
    encoder.get_embedder()
    cache_key = encoder.cache_key(encoder.normalize_input((torch.zeros(1, 8), 16_000)))

    assert fake_auto_model.received_model_names_or_paths == [
        expected_model_name_or_path
    ]
    assert encoder.encoder_id == expected_model_name_or_path
    assert cache_key is not None
    assert cache_key.encoder_id == expected_model_name_or_path


@pytest.mark.parametrize("configured_model_name_or_path", ["", "   "])
def test_speaker_encoder_rejects_an_empty_model_source(
    configured_model_name_or_path: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="speaker_embedding_model_name_or_path must not be empty",
    ):
        speaker_encoder.SpeakerEncoder(
            device="cpu",
            speaker_embedding_model_name_or_path=configured_model_name_or_path,
        )


def test_factory_config_reaches_the_speaker_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_encoder_kwargs: dict[str, str | int | bool | None] = {}

    class Recorder:
        def __init__(
            self,
            *,
            device: str,
            cache_max_items: int,
            compile_forward: bool,
            speaker_embedding_model_name_or_path: str | None,
        ) -> None:
            speaker_encoder_kwargs.update(
                {
                    "device": device,
                    "cache_max_items": cache_max_items,
                    "compile_forward": compile_forward,
                    "speaker_embedding_model_name_or_path": (
                        speaker_embedding_model_name_or_path
                    ),
                }
            )

    monkeypatch.setattr(speaker_encoder, "SpeakerEncoder", Recorder)
    stage = StageConfig(
        name="speaker_encode",
        factory_path="sglang_omni.models.zonos2.stages.create_speaker_encode_executor",
        factory={
            "speaker_embedding_model_name_or_path": ("/models/qwen3_voice_embedding")
        },
    )

    create_speaker_encode_executor(
        "/models/zonos2",
        device="cpu",
        **resolve_stage_typed_kwargs(stage),
    )

    assert (
        speaker_encoder_kwargs["speaker_embedding_model_name_or_path"]
        == "/models/qwen3_voice_embedding"
    )
