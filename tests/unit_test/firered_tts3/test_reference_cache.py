# SPDX-License-Identifier: Apache-2.0
"""Reference encoding cache behavior."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import av
import numpy as np
import pytest
import torch

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import load_firered_tts3_state
from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceHook,
    ReferenceAudioInput,
    ReferenceConditioning,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.utils.audio import check_torchcodec_ready, load_audio
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer
from tests.unit_test.fixtures.firered_tts3_preprocessing import (
    make_preprocessing_payload,
    make_reference_encoder,
    preprocess_payload,
)


@pytest.mark.parametrize(
    "task_type,is_data_uri",
    [("Base", True), ("SemanticEdit", False), ("AcousticEdit", True)],
)
def test_compressed_reference_prompt_uses_encoded_audio_and_cache(
    tmp_path: Path,
    task_type: Literal["Base", "SemanticEdit", "AcousticEdit"],
    is_data_uri: bool,
) -> None:
    """AAC references must share the encoder's patch count without a second decode."""
    if not check_torchcodec_ready():
        pytest.skip("AAC decoding requires TorchCodec and FFmpeg")
    else:
        pass
    encoded_audio = io.BytesIO()
    with av.open(encoded_audio, mode="w", format="mp4") as container:
        audio_stream = container.add_stream("aac", rate=16000)
        audio_stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((2, 4097), dtype=np.float32), format="fltp", layout="stereo"
        )
        frame.sample_rate = 16000
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    audio_bytes = encoded_audio.getvalue()
    if is_data_uri:
        reference_source = (
            f"data:audio/mp4;base64,{base64.b64encode(audio_bytes).decode('ascii')}"
        )
    else:
        reference_path = tmp_path / "reference.m4a"
        reference_path.write_bytes(audio_bytes)
        reference_source = str(reference_path)
    config = FireRedTTS3Config(use_lm_head=True)
    tokenizer = make_speaker_tokenizer()
    with make_reference_encoder(config) as reference_encoder:
        with patch(
            "sglang_omni.models.firered_tts3.ref_encode.load_audio", wraps=load_audio
        ) as decode_reference:
            for _ in range(2):
                payload = make_preprocessing_payload(
                    inputs={
                        "text": "hello" if task_type == "Base" else "",
                        "task_type": task_type,
                        "instructions": "Replace 'birthday' with 'anniversary'.",
                        "references": [
                            {"audio_path": reference_source, "text": "source text"}
                        ],
                    }
                )
                preprocess_payload(payload, config=config, tokenizer=tokenizer)
                state = load_firered_tts3_state(
                    reference_encoder.encode_payload(payload)
                )
                if task_type == "Base":
                    reference_latents = state.prompt_latents
                    pad_token = "<|video_pad|>"
                else:
                    reference_latents = state.audio_in_latents
                    pad_token = "<|image_pad|>"
                assert reference_latents is not None
                assert state.input_ids.count(
                    tokenizer.convert_tokens_to_ids(pad_token)
                ) == (reference_latents.shape[1] // config.patch_size)
            assert decode_reference.call_count == 1


class ReferenceEncoderStub:
    """Distinct artifacts make cache hits observable without loading an audio model."""

    sample_rate: int = 24000
    pad_multiple: int = 3840
    redae_scale: float = 1.0
    spk_extractor: torch.nn.Module | None = None

    def __init__(self) -> None:
        self.encode_calls: int = 0

    def encode(
        self, path: str, max_seconds: float | None = None
    ) -> dict[str, torch.Tensor]:
        self.encode_calls += 1
        return {"prompt_latents": torch.full((1, 2, 4), float(self.encode_calls))}


def make_reference_service(
    reference_encoder: ReferenceEncoderStub, *, cache_enabled: bool = True
) -> ReferenceEncodeService[
    ReferenceAudioInput,
    ReferenceConditioning,
    ReferenceConditioning,
    ReferenceAudioInput,
]:
    return ReferenceEncodeService(
        FireRedReferenceHook(
            reference_encoder, model_id="stub", cache_enabled=cache_enabled
        ),
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_reuses_stored_artifact(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    first = service.get_or_encode(reference_audio)
    second = service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode(reference_audio)
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    uncapped_reference = ReferenceAudioInput(str(reference_audio_path), None)
    capped_reference = ReferenceAudioInput(str(reference_audio_path), 30.0)
    uncapped = service.get_or_encode(uncapped_reference)
    capped = service.get_or_encode(capped_reference)
    assert reference_encoder.encode_calls == 2
    assert service.stats()["entries"] == 2
    assert torch.equal(uncapped["prompt_latents"], torch.ones(1, 2, 4))
    assert torch.equal(capped["prompt_latents"], torch.full((1, 2, 4), 2.0))
    assert torch.equal(
        service.get_or_encode(uncapped_reference)["prompt_latents"],
        uncapped["prompt_latents"],
    )
    assert torch.equal(
        service.get_or_encode(capped_reference)["prompt_latents"],
        capped["prompt_latents"],
    )
    assert reference_encoder.encode_calls == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder, cache_enabled=False)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    service.get_or_encode(reference_audio)
    service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()
