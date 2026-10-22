# SPDX-License-Identifier: Apache-2.0
"""Speech request encoding and speaker embedding loading for Voxtral TTS."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import torch
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from sglang_omni_mlx.voxtral_tts.config import VoxtralConfig


class VoxtralSpeechTokenizer:
    def __init__(self, checkpoint_directory: Path) -> None:
        tokenizer_path = checkpoint_directory / "tekken.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Voxtral tokenizer is missing: {tokenizer_path}")
        else:
            pass
        self.tokenizer = MistralTokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str, voice: str) -> list[int]:
        if not text.strip():
            raise ValueError("Voxtral TTS input text must not be empty")
        else:
            pass
        return list(
            self.tokenizer.encode_speech_request(
                SpeechRequest(input=text, voice=voice)
            ).tokens
        )


def load_voice_embeddings(config: VoxtralConfig) -> dict[str, mx.array]:
    voice_directory = config.checkpoint_directory / "voice_embedding"
    if not voice_directory.is_dir():
        raise FileNotFoundError(
            f"Voxtral voice embeddings are missing: {voice_directory}"
        )
    else:
        pass
    voice_embeddings: dict[str, mx.array] = {}
    for voice_path in sorted(voice_directory.glob("*.pt")):
        torch_embedding = torch.load(voice_path, map_location="cpu", weights_only=True)
        if not isinstance(torch_embedding, torch.Tensor):
            raise ValueError(f"Voxtral voice embedding is not a tensor: {voice_path}")
        else:
            pass
        voice_embeddings[voice_path.stem] = mx.array(torch_embedding.float().numpy())
    if not voice_embeddings:
        raise FileNotFoundError(
            f"Voxtral voice directory has no .pt files: {voice_directory}"
        )
    else:
        pass
    missing_voices = set(config.voices) - set(voice_embeddings)
    if missing_voices:
        raise ValueError(
            f"Voxtral voice embeddings are missing configured voices: {sorted(missing_voices)}"
        )
    else:
        pass
    return voice_embeddings
