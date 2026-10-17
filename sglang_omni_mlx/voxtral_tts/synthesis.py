# SPDX-License-Identifier: Apache-2.0
"""Autoregressive Voxtral semantic and acoustic code generation in MLX."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Literal

import mlx.core as mx

from sglang_omni_mlx.voxtral_tts.acoustic_head import (
    END_AUDIO_TOKEN_ID,
    SPECIAL_TOKEN_COUNT,
)
from sglang_omni_mlx.voxtral_tts.model import VoxtralMLXModel

AudioFinishReason = Literal["stop", "length"]


class SpeechGenerationCancelledError(RuntimeError):
    """The request was cancelled while Voxtral was generating speech."""


@dataclass(frozen=True, kw_only=True)
class GeneratedAudioCodes:
    audio_codes: mx.array
    finish_reason: AudioFinishReason


@dataclass(frozen=True, kw_only=True)
class AudioSynthesisResult:
    waveform: mx.array
    finish_reason: AudioFinishReason


class VoxtralSynthesizer:
    def __init__(
        self,
        model: VoxtralMLXModel,
        audio_token_id: int,
        max_audio_frames: int,
    ) -> None:
        if max_audio_frames < 1:
            raise ValueError(
                f"max_audio_frames must be positive, got {max_audio_frames}"
            )
        else:
            pass
        self.model = model
        self.audio_token_id = audio_token_id
        self.max_audio_frames = max_audio_frames

    def generate_codes(
        self,
        prompt_token_ids: list[int],
        voice_embedding: mx.array | None,
        cancellation_event: Event,
    ) -> GeneratedAudioCodes:
        if not prompt_token_ids:
            raise ValueError("Voxtral prompt must contain at least one token")
        else:
            pass
        input_ids = mx.array([prompt_token_ids], dtype=mx.int32)
        prompt_embeddings = self.model.text_model.embed_tokens(input_ids)
        placeholder_mask = input_ids[0] == self.audio_token_id
        if voice_embedding is not None:
            voice_embedding = voice_embedding.reshape(-1, voice_embedding.shape[-1])
            if voice_embedding.shape[0] == 0:
                raise ValueError(
                    "Voxtral voice embedding must contain at least one frame"
                )
            else:
                pass
            placeholder_count = int(mx.sum(placeholder_mask).item())
            voice_frame_count = voice_embedding.shape[0]
            if placeholder_count != voice_frame_count:
                raise ValueError(
                    "Voxtral voice embedding frame count does not match the "
                    f"prompt placeholder count: {voice_frame_count} != "
                    f"{placeholder_count}"
                )
            else:
                pass
            placeholder_offsets = mx.cumsum(placeholder_mask.astype(mx.int32)) - 1
            safe_voice_offsets = mx.clip(
                placeholder_offsets, 0, voice_embedding.shape[0] - 1
            )
            voice_prompt_embeddings = voice_embedding[safe_voice_offsets].astype(
                prompt_embeddings.dtype
            )
            prompt_embeddings = mx.where(
                placeholder_mask[None, :, None],
                voice_prompt_embeddings[None, :, :],
                prompt_embeddings,
            )
        else:
            pass

        caches = self.model.new_caches()
        language_hidden_states = self.model.text_model(prompt_embeddings, caches)
        generated_codes: list[mx.array] = []
        for _ in range(self.max_audio_frames):
            if cancellation_event.is_set():
                raise SpeechGenerationCancelledError(
                    "Voxtral speech generation was cancelled"
                )
            else:
                pass
            audio_codes = self.model.acoustic_head(language_hidden_states)
            semantic_code = int(audio_codes[0, 0].item())
            if semantic_code == END_AUDIO_TOKEN_ID:
                finish_reason: AudioFinishReason = "stop"
                break
            else:
                generated_codes.append(audio_codes[0])
                if len(generated_codes) < self.max_audio_frames:
                    feedback_embedding = self.model.embed_audio_codes(audio_codes)[
                        :, None, :
                    ]
                    language_hidden_states = self.model.text_model(
                        feedback_embedding, caches
                    )
                    mx.eval(language_hidden_states)
                else:
                    pass
        else:
            finish_reason = "length"
        if not generated_codes:
            audio_codes = mx.zeros(
                (0, self.model.acoustic_head.acoustic_codebook_count + 1),
                dtype=mx.int32,
            )
        else:
            audio_codes = mx.stack(generated_codes).astype(mx.int32)
        return GeneratedAudioCodes(
            audio_codes=audio_codes,
            finish_reason=finish_reason,
        )

    def synthesize(
        self,
        prompt_token_ids: list[int],
        voice_embedding: mx.array | None,
        cancellation_event: Event,
    ) -> AudioSynthesisResult:
        if self.model.audio_tokenizer is None:
            raise RuntimeError("Voxtral audio tokenizer was not loaded")
        else:
            pass
        generation_result = self.generate_codes(
            prompt_token_ids, voice_embedding, cancellation_event
        )
        if generation_result.audio_codes.shape[0] == 0:
            raise RuntimeError("Voxtral TTS generated no audio codes")
        else:
            pass
        audio_codes = generation_result.audio_codes - SPECIAL_TOKEN_COUNT
        warmup_codes = mx.repeat(audio_codes[:1], 2, axis=0)
        decoder_codes = mx.concatenate([warmup_codes, audio_codes], axis=0)
        waveform = self.model.audio_tokenizer(
            decoder_codes.transpose(1, 0)[None, :, :]
        )[0, 0]
        trim_samples = 2 * self.model.audio_tokenizer.downsample_factor
        waveform = waveform[trim_samples:]
        fade_samples = self.model.audio_tokenizer.sampling_rate // 100
        fade = mx.linspace(0.0, 1.0, fade_samples, dtype=waveform.dtype)
        fade_length = min(fade_samples, waveform.shape[0])
        waveform[:fade_length] = waveform[:fade_length] * fade[:fade_length]
        return AudioSynthesisResult(
            waveform=waveform,
            finish_reason=generation_result.finish_reason,
        )
