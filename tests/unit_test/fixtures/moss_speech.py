# SPDX-License-Identifier: Apache-2.0
"""Shared MOSS-Speech tokenizer, codec and client substitutes."""

import base64
import io
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile
import torch
import torchaudio
from transformers import EncoderDecoderCache, WhisperFeatureExtractor

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.models.moss_speech.codec import AudioEncoderOutput, MossSpeechCodec
from sglang_omni.models.moss_speech.fsm import AUDIO_PAD, MODALITY_PAD, SOSP
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechState,
    build_result,
    build_sglang_moss_request,
    preprocess_request,
)
from sglang_omni.proto.request import StagePayload


class TextTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        return bytes(token for token in tokens if token < 256).decode()


class ReferenceEncoder(torch.nn.Module):
    def forward(
        self,
        *,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        past_key_values: EncoderDecoderCache,
    ) -> AudioEncoderOutput:
        return SimpleNamespace(
            quantized_token_ids=torch.zeros(
                (1, input_features.shape[-1] // 8), dtype=torch.long
            )
        )


class OfflineDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feat_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=480, n_mels=80
        )
        self.campplus_session = Mock()
        self.campplus_session.get_inputs.return_value = [SimpleNamespace(name="speech")]
        self.campplus_session.run.return_value = [np.zeros((1, 192), dtype=np.float32)]

    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        *,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert finalize and token.shape == (1, 2)
        waveform = torch.rand(1, 1920) * 0.1 + prompt_feat.mean() / 1000
        return waveform, torch.zeros(1, 80, 4)


def make_codec() -> MossSpeechCodec:
    codec = object.__new__(MossSpeechCodec)
    codec.device = torch.device("cpu")
    codec.decoder = OfflineDecoder()
    codec.encoder = ReferenceEncoder()
    codec.feature_extractor = WhisperFeatureExtractor()
    codec.feature_frames_per_token = 8
    return codec


def audio_reference(frequency_hz: int = 220, sample_count: int = 24000) -> str:
    waveform = 0.1 * np.sin(2 * np.pi * frequency_hz * np.arange(sample_count) / 24000)
    encoded = io.BytesIO()
    soundfile.write(encoded, waveform, 24000, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(encoded.getvalue()).decode()


class SpeechClient(Client):
    def __init__(self, codec: MossSpeechCodec) -> None:
        self.codec = codec

    async def abort(self, request_id: str) -> bool:
        return True

    async def generate(
        self, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        request_id = request_id or "speech"
        prepared = preprocess_request(
            StagePayload(request_id, Client.build_omni_request(request), {}),
            TextTokenizer(),
            self.codec,
        )
        native = build_sglang_moss_request(prepared)
        if MossSpeechState.from_dict(prepared.data).output_modality == "text":
            native.output_rows = [(token, AUDIO_PAD) for token in b"four"]
        else:
            native.output_rows = [
                (SOSP, AUDIO_PAD),
                (MODALITY_PAD, 3),
                (MODALITY_PAD, 4),
            ][: native.max_new_tokens]
        native.finish_reason = "length"
        result = self.codec.decode(build_result(native), tokenizer=TextTokenizer())
        yield Client.default_result_builder(request_id, result.data)
