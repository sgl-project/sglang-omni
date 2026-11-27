# SPDX-License-Identifier: Apache-2.0
"""Audio decoding, downmix, the Whisper sample rate, and the media policy."""

import asyncio
import io
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
import torch
from whisper.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE

from sglang_omni.models.ming3_audio.preprocessing import (
    AUDIO_PLACEHOLDER,
    Ming3AudioPreprocessor,
    separate_asr_language_prefix,
)
from sglang_omni.modules.ming_audio.audio_encoder import AudioEncoderConfig
from sglang_omni.modules.ming_audio.audio_media import AudioMediaIO
from sglang_omni.preprocessing import resource_connector
from sglang_omni.preprocessing.resource_connector import MediaPolicyError
from sglang_omni.proto.request import OmniRequest, StagePayload

STEREO_FRAME = np.array([[0.5, -0.25]], dtype=np.float32)
STEREO_FRAME_COUNT = 1600
MEAN_AMPLITUDE = 0.125
SOURCE_SAMPLE_RATE = 48000


def stereo_channel_wav_bytes(subtype: Literal["PCM_16", "FLOAT"]) -> bytes:
    waveform = np.tile(STEREO_FRAME, (STEREO_FRAME_COUNT, 1))
    encoded = io.BytesIO()
    soundfile.write(
        encoded, waveform, WHISPER_SAMPLE_RATE, format="WAV", subtype=subtype
    )
    return encoded.getvalue()


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
def test_wav_bytes_downmix_stereo_channels_to_mean(
    subtype: Literal["PCM_16", "FLOAT"],
) -> None:
    actual = AudioMediaIO().load_bytes(stereo_channel_wav_bytes(subtype))
    torch.testing.assert_close(
        actual, torch.full((STEREO_FRAME_COUNT,), MEAN_AMPLITUDE), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("transcript", "recognized_text"),
    [
        ("English\thello there", "hello there"),
        ("hello there", "hello there"),
        ("English\t", ""),
    ],
)
def test_asr_transcript_drops_language_prefix(
    transcript: str, recognized_text: str
) -> None:
    assert separate_asr_language_prefix(transcript) == recognized_text


def test_loader_resamples_to_the_whisper_rate() -> None:
    encoded = io.BytesIO()
    soundfile.write(
        encoded,
        np.zeros(SOURCE_SAMPLE_RATE, dtype=np.float32),
        SOURCE_SAMPLE_RATE,
        format="WAV",
        subtype="FLOAT",
    )
    actual = AudioMediaIO().load_bytes(encoded.getvalue())
    assert actual.shape == (WHISPER_SAMPLE_RATE,)


AUDIO_TOKEN_ID = 7


class PlaceholderTokenizer:
    def apply_chat_template(
        self, conversation: list[dict[str, object]], **template_options: object
    ) -> str:
        return AUDIO_PLACEHOLDER

    def encode(self, text: str) -> list[int]:
        return [AUDIO_TOKEN_ID] * text.count(AUDIO_PLACEHOLDER)


def test_audio_paths_follow_the_server_media_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chat audio path outside --allowed-local-media-path was read anyway."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside, outside = allowed / "inside.wav", tmp_path / "outside.wav"
    for path in (inside, outside):
        soundfile.write(
            path, np.zeros(STEREO_FRAME_COUNT, dtype=np.float32), WHISPER_SAMPLE_RATE
        )
    monkeypatch.setenv(resource_connector.ALLOWED_LOCAL_MEDIA_PATH_ENV, str(allowed))
    monkeypatch.setattr(resource_connector, "_global_connector", None)
    preprocessor = Ming3AudioPreprocessor.__new__(Ming3AudioPreprocessor)
    preprocessor.tokenizer = PlaceholderTokenizer()
    preprocessor.audio_config = AudioEncoderConfig(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3, 3, 3],
        downsampler_stride_list=[2, 2, 2],
    )
    preprocessor.hidden_size = 8
    preprocessor.audio_token_id = AUDIO_TOKEN_ID

    def preprocess(path: Path) -> StagePayload:
        inputs = {
            "messages": [{"role": "user", "content": "Transcribe this."}],
            "audios": [str(path)],
        }
        payload = StagePayload(
            "ming3-policy", request=OmniRequest(inputs=inputs), data={}
        )
        return asyncio.run(preprocessor(payload))

    assert len(preprocess(inside).data["features"]) == 1
    with pytest.raises(MediaPolicyError, match="not within allowed directory"):
        preprocess(outside)
