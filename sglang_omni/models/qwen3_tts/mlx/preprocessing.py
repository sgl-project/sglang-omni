# SPDX-License-Identifier: Apache-2.0
"""MLX-native Qwen3-TTS request preprocessing.

The Torch preprocessing path reuses the engine's own model and wraps it in
``qwen_tts.Qwen3TTSModel`` to build prompts. Neither works on the MLX path: the
engine's model is the MLX talker, which has none of those builder methods, and
the wrapper additionally drags in a Torch speech tokenizer.

This assembles the same prompt tensors from the MLX talker instead, adding the
MLX speech-tokenizer encoder and speaker encoder only for Base voice cloning,
which is the only task that needs reference audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .prompt import PromptInputs, Qwen3TTSMlxPromptBuilder

logger = logging.getLogger(__name__)

TASK_BASE = "Base"
TASK_CUSTOM_VOICE = "CustomVoice"
TASK_VOICE_DESIGN = "VoiceDesign"
SAMPLE_RATE = 24000


class Qwen3TTSMlxPreprocessor:
    """Builds prompts for the MLX talker, with lazily loaded audio encoders."""

    def __init__(
        self,
        talker: Any,
        model_config: Any,
        tokenizer: Any,
        *,
        checkpoint_dir: str | Path,
    ) -> None:
        self.talker = talker
        self.model_config = model_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prompts = Qwen3TTSMlxPromptBuilder(talker, model_config, tokenizer)
        self._speech_tokenizer: Any | None = None
        self._speaker_encoder: Any | None = None
        self._generate_defaults: dict[str, Any] | None = None
        # Reference codes and text ids keyed by reference identity; re-encoding
        # a repeated voice dominates time-to-first-audio otherwise.
        self._reference_cache: dict[tuple, mx.array] = {}

    # -- lazily loaded audio front ends ----------------------------------

    @property
    def speech_tokenizer(self) -> Any:
        """The encoder half, loaded on first voice-clone request."""
        if self._speech_tokenizer is None:
            from .vocoder_loader import load_mlx_speech_tokenizer

            self._speech_tokenizer = load_mlx_speech_tokenizer(self.checkpoint_dir)
            if not self._speech_tokenizer.has_encoder:
                raise ValueError(
                    "Qwen3-TTS voice cloning needs a Base checkpoint: this "
                    "speech tokenizer has no encoder"
                )
        return self._speech_tokenizer

    @property
    def speaker_encoder(self) -> Any | None:
        """The x-vector encoder, or None when the checkpoint has none."""
        if self._speaker_encoder is None:
            config = self.model_config.speaker_encoder_config
            if config is None:
                return None
            from .speaker_encoder import Qwen3TTSSpeakerEncoder
            from .weights import align_conv_weights

            encoder = Qwen3TTSSpeakerEncoder(config)
            weights = mx.load(str(self.checkpoint_dir / "model.safetensors"))
            encoder.load_weights(
                list(align_conv_weights(encoder.sanitize(weights), encoder).items()),
                strict=True,
            )
            encoder.eval()
            mx.eval(encoder.parameters())
            self._speaker_encoder = encoder
        return self._speaker_encoder

    # -- reference audio --------------------------------------------------

    def _load_reference_audio(self, ref_audio: Any) -> np.ndarray:
        from sglang_omni.utils.audio import load_audio

        if isinstance(ref_audio, mx.array):
            return np.array(ref_audio, dtype=np.float32).reshape(-1)
        if isinstance(ref_audio, np.ndarray):
            return ref_audio.astype(np.float32).reshape(-1)
        return load_audio(ref_audio, target_sample_rate=SAMPLE_RATE, mono=True).astype(
            np.float32
        )

    def _encode_reference(self, audio: np.ndarray) -> mx.array:
        key = (audio.shape[0], float(audio.sum()))
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        codes = self.speech_tokenizer.encode(mx.array(audio)[None, None, :])
        mx.eval(codes)
        self._reference_cache[key] = codes
        return codes

    def _speaker_embedding(self, audio: np.ndarray) -> mx.array | None:
        encoder = self.speaker_encoder
        if encoder is None:
            return None
        from .dsp import mel_spectrogram

        mels = mel_spectrogram(
            mx.array(audio),
            num_mels=self.model_config.speaker_encoder_config.mel_dim,
            sample_rate=SAMPLE_RATE,
        )
        return encoder(mels)

    # -- entry point ------------------------------------------------------

    def prepare(self, state: Any) -> PromptInputs:
        """Build the prompt for one request from its pipeline state."""
        task = str(state.task_type or TASK_BASE)
        language = state.language or "auto"

        if task == TASK_CUSTOM_VOICE:
            from sglang_omni.models.qwen3_tts.request_builders import (
                QWEN3_TTS_DEFAULT_CUSTOM_VOICE,
            )

            return self.prompts.build_custom_voice(
                text=state.text,
                voice=state.voice or QWEN3_TTS_DEFAULT_CUSTOM_VOICE,
                language=language,
                non_streaming_mode=bool(state.non_streaming_mode),
                instructions=state.instructions,
            )

        if task == TASK_VOICE_DESIGN:
            return self.prompts.build_voice_design(
                text=state.text,
                instructions=state.instructions or "",
                language=language,
                non_streaming_mode=bool(state.non_streaming_mode),
            )

        if task != TASK_BASE:
            raise ValueError(f"Unhandled Qwen3-TTS task type: {task!r}")

        if state.ref_audio is None or not state.ref_text:
            raise ValueError(
                "Qwen3-TTS Base needs both ref_audio and ref_text to clone a voice"
            )
        audio = self._load_reference_audio(state.ref_audio)
        return self.prompts.build_voice_clone(
            text=state.text,
            ref_codes=self._encode_reference(audio),
            ref_text=state.ref_text,
            speaker_embed=self._speaker_embedding(audio),
            language=language,
            instructions=state.instructions,
        )

    # -- generation defaults ---------------------------------------------

    def merge_generate_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Request overrides on top of the checkpoint's generation_config."""
        merged = dict(self._generation_defaults())
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged

    def _generation_defaults(self) -> dict[str, Any]:
        if self._generate_defaults is None:
            import json

            path = self.checkpoint_dir / "generation_config.json"
            try:
                self._generate_defaults = json.loads(path.read_text())
            except (OSError, ValueError):
                logger.warning("No usable %s; using built-in defaults", path)
                self._generate_defaults = {}
        return self._generate_defaults


def build_mlx_prepared_request(
    preprocessor: Qwen3TTSMlxPreprocessor,
    state: Any,
    *,
    gen_kwargs: dict[str, Any],
) -> Any:
    """Assemble the scheduler's prepared-request record from an MLX prompt.

    Prompt tensors stay as MLX arrays: the MLX runner consumes them directly, so
    converting to Torch here and back at prefill would be pure overhead. Only the
    radix cache key is derived through Torch, because that keying is shared with
    every other backend.
    """
    import torch

    from sglang_omni.models.qwen3_tts.request_builders import (
        Qwen3TTSPreparedRequest,
        build_embedding_cache_key_ids,
    )

    prompt = preprocessor.prepare(state)
    mx.eval(prompt.input_embeds, prompt.trailing_text_embeds, prompt.pad_embed)

    # [1, T, hidden] -> [T, hidden], matching the Torch path's squeeze.
    prompt_embeds = prompt.input_embeds[0]
    trailing = prompt.trailing_text_embeds[0]

    key_source = torch.from_numpy(
        np.array(prompt_embeds.astype(mx.float32), dtype=np.float32)
    )
    input_ids_list = build_embedding_cache_key_ids(key_source)

    ref_codes = prompt.ref_codes
    if ref_codes is not None:
        mx.eval(ref_codes)
        state.ref_code_len = int(ref_codes.shape[-1])

    return Qwen3TTSPreparedRequest(
        state=state,
        input_ids_list=input_ids_list,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        attention_mask=torch.ones((1, int(prompt_embeds.shape[0])), dtype=torch.long),
        trailing_text_hidden=trailing,
        ref_code=ref_codes,
        prompt_input_embeds=prompt_embeds,
        tts_pad_embed=prompt.pad_embed,
        gen_kwargs=gen_kwargs,
    )
