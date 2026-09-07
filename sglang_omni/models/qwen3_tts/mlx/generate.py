# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).
"""Single-request Qwen3-TTS generation on MLX, including Base voice cloning.

This is the model-level driver: it owns prompt construction, the autoregressive
frame loop, and vocoding for one request. Omni's scheduler-driven runner reuses
these pieces rather than replacing them, so the frame loop here is the same one
batched serving will call.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

from .config import ModelConfig, TokenizerConfig
from .sampling import SamplingParams, sample_codec_token, special_codec_token_ids
from .speaker_encoder import Qwen3TTSSpeakerEncoder
from .speech_tokenizer import Qwen3TTSSpeechTokenizer
from .talker import Qwen3TTSTalkerForConditionalGeneration
from .weights import align_conv_weights

SAMPLE_RATE = 24000


@dataclass
class CloneRequest:
    """One voice-cloning request."""

    text: str
    ref_audio: Any
    ref_text: str
    language: str = "auto"
    max_frames: int = 4096
    semantic: SamplingParams = field(
        default_factory=lambda: SamplingParams(
            temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05
        )
    )
    subtalker: SamplingParams = field(
        default_factory=lambda: SamplingParams(
            temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.0
        )
    )
    seed: int | None = None


@dataclass
class IclPrompt:
    """Everything the frame loop needs after prompt construction."""

    input_embeds: mx.array
    trailing_text_embeds: mx.array
    pad_embed: mx.array
    ref_codes: mx.array


def _load_audio_24k(source: Any) -> np.ndarray:
    """Mono 24 kHz float32 samples from a path, URL, bytes, or array."""
    if isinstance(source, mx.array):
        return np.array(source, dtype=np.float32).reshape(-1)
    if isinstance(source, np.ndarray):
        return source.astype(np.float32).reshape(-1)

    from sglang_omni.utils.audio import load_audio

    return load_audio(source, target_sample_rate=SAMPLE_RATE, mono=True).astype(
        np.float32
    )


class Qwen3TTSMlxGenerator:
    """Talker + speech tokenizer + speaker encoder, loaded from a checkpoint."""

    def __init__(
        self,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        speech_tokenizer: Qwen3TTSSpeechTokenizer,
        tokenizer: Any,
        config: ModelConfig,
        speaker_encoder: Qwen3TTSSpeakerEncoder | None = None,
    ) -> None:
        self.talker = talker
        self.speech_tokenizer = speech_tokenizer
        self.tokenizer = tokenizer
        self.config = config
        self.speaker_encoder = speaker_encoder
        self._reference_cache: dict[tuple, tuple[mx.array, mx.array]] = {}

    # -- loading ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> Qwen3TTSMlxGenerator:
        from transformers import AutoTokenizer

        model_path = Path(model_path)
        config = ModelConfig.from_dict(
            json.loads((model_path / "config.json").read_text())
        )

        weights = mx.load(str(model_path / "model.safetensors"))

        talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
        talker.load_weights(list(talker.sanitize(weights).items()), strict=True)
        talker.eval()

        speaker_encoder = None
        if config.speaker_encoder_config is not None:
            speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
            speaker_weights = align_conv_weights(
                speaker_encoder.sanitize(weights), speaker_encoder
            )
            speaker_encoder.load_weights(list(speaker_weights.items()), strict=True)
            speaker_encoder.eval()
        del weights

        tokenizer_dir = model_path / "speech_tokenizer"
        tokenizer_config = TokenizerConfig.from_dict(
            json.loads((tokenizer_dir / "config.json").read_text())
        )
        speech_tokenizer = Qwen3TTSSpeechTokenizer(tokenizer_config)
        tokenizer_weights = mx.load(str(tokenizer_dir / "model.safetensors"))
        speech_tokenizer.load_weights(
            list(
                align_conv_weights(
                    speech_tokenizer.sanitize(tokenizer_weights), speech_tokenizer
                ).items()
            ),
            strict=True,
        )
        speech_tokenizer.eval()
        del tokenizer_weights

        mx.eval(talker.parameters(), speech_tokenizer.parameters())
        if speaker_encoder is not None:
            mx.eval(speaker_encoder.parameters())

        return cls(
            talker=talker,
            speech_tokenizer=speech_tokenizer,
            tokenizer=AutoTokenizer.from_pretrained(str(model_path)),
            config=config,
            speaker_encoder=speaker_encoder,
        )

    # -- prompt pieces ---------------------------------------------------

    @property
    def _talker_config(self):
        return self.config.talker_config

    def _project_text(self, token_ids: mx.array) -> mx.array:
        return self.talker.text_projection(self.talker.get_text_embeddings()(token_ids))

    def _codec(self, token_ids: list[list[int]]) -> mx.array:
        return self.talker.get_input_embeddings()(mx.array(token_ids, dtype=mx.int32))

    def speaker_embedding(self, audio: np.ndarray) -> mx.array:
        """ECAPA-TDNN x-vector for the reference waveform."""
        if self.speaker_encoder is None:
            raise ValueError("This checkpoint has no speaker encoder")
        from .dsp import mel_spectrogram

        mels = mel_spectrogram(
            mx.array(audio),
            n_fft=1024,
            num_mels=self.config.speaker_encoder_config.mel_dim,
            sample_rate=SAMPLE_RATE,
            hop_size=256,
            win_size=1024,
            fmin=0.0,
            fmax=12000.0,
        )
        return self.speaker_encoder(mels)

    @property
    def cached_reference_count(self) -> int:
        """How many distinct references are memoised. See ``encode_reference``."""
        return len(self._reference_cache)

    def encode_reference(
        self, audio: np.ndarray, ref_text: str
    ) -> tuple[mx.array, mx.array]:
        """Reference codes and reference text ids, memoised per reference.

        Re-encoding a reference costs a full pass over its audio, which
        dominates time-to-first-audio when the same voice is reused.
        """
        key = (ref_text, audio.shape[0], float(audio.sum()))
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached

        codes = self.speech_tokenizer.encode(mx.array(audio)[None, None, :])
        chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        # Drop the 3 role tokens at the front and "<|im_end|>\n" at the back.
        ids = mx.array(self.tokenizer.encode(chat), dtype=mx.int32)[None, 3:-2]
        mx.eval(codes, ids)
        self._reference_cache[key] = (codes, ids)
        return codes, ids

    def build_icl_prompt(
        self,
        text: str,
        audio: np.ndarray,
        ref_text: str,
        language: str = "auto",
    ) -> IclPrompt:
        """Assemble the in-context voice-cloning prefill.

        Layout: role tokens, then a codec control prefix carrying the speaker
        embedding, then the text and reference-codec streams laid end to end.
        Both streams are summed with the other's padding embedding so every
        position carries one text and one codec contribution.
        """
        config = self._talker_config
        ref_codes, ref_text_ids = self.encode_reference(audio, ref_text)

        target_chat = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        target_ids = mx.array(self.tokenizer.encode(target_chat), dtype=mx.int32)[None]
        # Drop the 3 role tokens and the 5-token trailing template.
        text_ids = target_ids[:, 3:-5]

        specials = self._project_text(
            mx.array(
                [
                    [
                        self.config.tts_bos_token_id,
                        self.config.tts_eos_token_id,
                        self.config.tts_pad_token_id,
                    ]
                ],
                dtype=mx.int32,
            )
        )
        bos_embed, eos_embed, pad_embed = (
            specials[:, 0:1, :],
            specials[:, 1:2, :],
            specials[:, 2:3, :],
        )

        text_embed = self._project_text(
            mx.concatenate([ref_text_ids, text_ids], axis=1)
        )
        text_embed = mx.concatenate([text_embed, eos_embed], axis=1)

        # Reference codec stream: every codebook summed, group 0 through the
        # talker's table and the rest through the predictor's.
        codec_embed = self.talker.get_input_embeddings()(ref_codes[:, 0, :])
        for group in range(config.num_code_groups - 1):
            codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[
                group
            ](ref_codes[:, group + 1, :])
        codec_embed = mx.concatenate(
            [self._codec([[config.codec_bos_id]]), codec_embed], axis=1
        )

        codec_pad_embed = self._codec([[config.codec_pad_id]])
        icl_embed = mx.concatenate(
            [
                text_embed + codec_pad_embed,
                codec_embed + pad_embed,
            ],
            axis=1,
        )

        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            language_id = config.codec_language_id.get(language.lower())

        if language_id is None:
            control = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            control = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]
        prefix_embed = self._codec([control])

        if self.speaker_encoder is not None:
            speaker = self.speaker_embedding(audio)
            # The encoder runs in float32; without this cast the whole prompt
            # and the talker's KV cache get promoted along with it.
            speaker = speaker.astype(prefix_embed.dtype).reshape(1, 1, -1)
            prefix_embed = mx.concatenate([prefix_embed, speaker], axis=1)
        prefix_embed = mx.concatenate(
            [prefix_embed, self._codec([[config.codec_pad_id, config.codec_bos_id]])],
            axis=1,
        )

        # Text side of the control prefix: padding, then the TTS BOS, summed
        # with every control embedding except the final codec BOS.
        pad_run = mx.broadcast_to(
            pad_embed, (1, prefix_embed.shape[1] - 2, pad_embed.shape[-1])
        )
        control_embed = (
            mx.concatenate([pad_run, bos_embed], axis=1) + prefix_embed[:, :-1, :]
        )

        role_embed = self._project_text(target_ids[:, :3])
        return IclPrompt(
            input_embeds=mx.concatenate([role_embed, control_embed, icl_embed], axis=1),
            trailing_text_embeds=pad_embed,
            pad_embed=pad_embed,
            ref_codes=ref_codes,
        )

    # -- autoregressive loop ---------------------------------------------

    def generate_frames(
        self,
        prompt: IclPrompt,
        request: CloneRequest,
    ) -> Iterator[mx.array]:
        """Yield one ``[1, num_code_groups]`` codec frame at a time."""
        config = self._talker_config
        eos_id = config.codec_eos_token_id
        suppress_tokens = special_codec_token_ids(config.vocab_size, keep=eos_id)

        cache = self.talker.make_cache()
        code_cache = self.talker.code_predictor.make_cache()
        input_embeds = prompt.input_embeds
        emitted: list[int] = []
        trailing_index = 0

        for _ in range(request.max_frames):
            logits, hidden = self.talker(input_embeds, cache=cache)
            semantic = sample_codec_token(
                logits[:, -1, :],
                request.semantic,
                recent_tokens=emitted,
                suppress_tokens=suppress_tokens,
            )

            frame, codec_embed = self.talker.predict_codes(
                semantic,
                hidden,
                cache=code_cache,
                sampler=lambda group_logits, _index: sample_codec_token(
                    group_logits, request.subtalker
                ),
            )

            if trailing_index < prompt.trailing_text_embeds.shape[1]:
                text_embed = prompt.trailing_text_embeds[
                    :, trailing_index : trailing_index + 1, :
                ]
                trailing_index += 1
            else:
                text_embed = prompt.pad_embed
            next_embeds = text_embed + codec_embed

            # One sync per frame: the whole graph above resolves together.
            mx.eval(semantic, frame, next_embeds)
            token = int(semantic[0, 0])
            if token == eos_id:
                break

            emitted.append(token)
            input_embeds = next_embeds
            yield frame

    def decode_frames(self, frames: list[mx.array], ref_codes: mx.array) -> np.ndarray:
        """Vocode generated frames, using the reference codes as left context.

        The reference and generated codes are decoded as one sequence so the
        decoder starts from the reference's acoustic state, then the reference
        span is cut off the front of the waveform.
        """
        if not frames:
            return np.zeros((0,), dtype=np.float32)

        generated = mx.stack(frames, axis=1)
        reference = mx.transpose(ref_codes, (0, 2, 1))
        full = mx.concatenate([reference, generated], axis=1)

        waveform, lengths = self.speech_tokenizer.decode(full)
        audio = waveform[0]
        mx.eval(audio, lengths)

        valid = int(lengths[0])
        if 0 < valid < audio.shape[0]:
            audio = audio[:valid]
        cut = int(reference.shape[1] / max(full.shape[1], 1) * audio.shape[0])
        if 0 < cut < audio.shape[0]:
            audio = audio[cut:]
        return np.array(audio, dtype=np.float32)

    def clone(self, request: CloneRequest) -> np.ndarray:
        """Synthesise ``request.text`` in the reference voice, at 24 kHz."""
        if not self.speech_tokenizer.has_encoder:
            raise ValueError(
                "Voice cloning needs a Base checkpoint: this speech tokenizer "
                "has no encoder, so reference audio cannot be encoded"
            )
        if request.seed is not None:
            mx.random.seed(request.seed)

        audio = _load_audio_24k(request.ref_audio)
        prompt = self.build_icl_prompt(
            request.text, audio, request.ref_text, request.language
        )
        frames = list(self.generate_frames(prompt, request))
        return self.decode_frames(frames, prompt.ref_codes)


def frames_to_seconds(num_frames: int) -> float:
    """Codec frames at 12.5 Hz to seconds of audio."""
    return num_frames / 12.5


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write mono float32 samples as a 16-bit WAV."""
    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return math.ceil(len(audio) / sample_rate * 1000) / 1000
