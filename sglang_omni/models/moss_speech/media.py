# SPDX-License-Identifier: Apache-2.0
"""Decode bounded inline audio at its native sample rate."""

import torch

from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.serve.speech_limits import MAX_REFERENCE_AUDIO_BYTES
from sglang_omni.utils.audio import decode_audio_data_uri, load_audio

# The checkpoint encodes eight 10 ms Whisper frames per token.
AUDIO_TOKEN_MILLISECONDS = 80
MAX_REFERENCE_SECONDS = 30


def decode_chat_audio(source: str, *, reference: bool) -> tuple[torch.Tensor, int]:
    if len(source) - source.find(",") - 1 > 4 * ((MAX_REFERENCE_AUDIO_BYTES + 2) // 3):
        raise ValueError(f"Audio exceeds {MAX_REFERENCE_AUDIO_BYTES} bytes")
    else:
        pass
    encoded = decode_audio_data_uri(source)
    if encoded is None:
        raise ValueError(
            "Supply inline audio or enable media resolution on the chat server"
        )
    else:
        pass
    if len(encoded) > MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError(f"Audio exceeds {MAX_REFERENCE_AUDIO_BYTES} bytes")
    else:
        pass
    samples, sample_rate = load_audio(
        encoded, source_name="chat audio", mono=reference, preserve_sample_rate=True
    )
    waveform = (
        torch.from_numpy(samples).reshape(1, -1)
        if reference
        else torch.from_numpy(samples[:1])
    )
    sample_count = waveform.shape[-1]
    if not sample_count or not torch.isfinite(waveform).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    elif reference and sample_count * 1000 < sample_rate * AUDIO_TOKEN_MILLISECONDS:
        raise ValueError("Reference audio is too short for codec conditioning")
    elif reference and sample_count > sample_rate * MAX_REFERENCE_SECONDS:
        raise ValueError("Reference audio exceeds 30 seconds")
    elif (
        sample_count * 1000
        > sample_rate * AUDIO_TOKEN_MILLISECONDS * MAX_CONTEXT_TOKENS
    ):
        raise ValueError("Speech input exceeds the model context")
    else:
        pass
    return waveform, sample_rate
