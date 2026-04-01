# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for Ming-Omni: tokenize text, extract audio mel features."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.common import (
    load_ming_config,
    load_ming_tokenizer,
)
from sglang_omni.models.ming_omni.io import PipelineState, PromptInputs
from sglang_omni.models.ming_omni.pipeline.next_stage import AUDIO_STAGE
from sglang_omni.preprocessing.audio import load_audio_path
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

# Ming-Omni chat template tokens
ROLE_HUMAN = "<role>HUMAN</role>"
ROLE_ASSISTANT = "<role>ASSISTANT</role>"
ROLE_SYSTEM = "<role>SYSTEM</role>"
ROLE_END = "<|role_end|>"
DEFAULT_SYSTEM_PROMPT = "你是一个友好的AI助手。"

# Modality tokens
AUDIO_START = "<audio>"
AUDIO_END = "</audio>"
AUDIO_PATCH = "<audioPatch>"
END_OF_AUDIO = "<end_of_audio>"

# Whisper mel spectrogram parameters
WHISPER_N_MELS = 128
WHISPER_HOP_LENGTH = 160
WHISPER_SAMPLE_RATE = 16000


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    n_mels: int = WHISPER_N_MELS,
) -> np.ndarray:
    """Compute log-mel spectrogram features compatible with Whisper encoder.

    Args:
        waveform: Audio waveform as float32 numpy array, shape [num_samples].
        sample_rate: Sample rate of the waveform (must be 16kHz).
        n_mels: Number of mel filterbanks.

    Returns:
        Mel spectrogram as float32 numpy array, shape [num_frames, n_mels].
    """
    try:
        import whisper

        # Use whisper's built-in mel computation for exact compatibility
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        audio_tensor = torch.from_numpy(waveform)
        mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=n_mels)
        # mel shape: [n_mels, num_frames] -> transpose to [num_frames, n_mels]
        return mel.numpy().T
    except ImportError:
        raise ImportError(
            "whisper package is required for Ming-Omni audio preprocessing. "
            "Install with: pip install openai-whisper"
        )


def estimate_audio_feature_length(
    mel_frames: int,
    ds_kernel_size: int = 1,
    ds_stride: int = 1,
) -> int:
    """Estimate the number of audio tokens after Whisper encoder + Conv1d projection.

    Two downsampling stages (matching Ming's modeling_utils.py):
    1. Whisper encoder internal conv: kernel=3, stride=2, padding=1
    2. Projection Conv1d: kernel=ds_kernel_size, stride=ds_stride, padding=ds_kernel_size//2
    """
    # Whisper encoder internal conv: (L - 3 + 2*1) // 2 + 1
    whisper_out_len = (mel_frames - 3 + 2 * 1) // 2 + 1
    # Projection Conv1d: (L - k + 2*(k//2)) // s + 1
    padding = ds_kernel_size // 2
    proj_out_len = (whisper_out_len - ds_kernel_size + 2 * padding) // ds_stride + 1
    return proj_out_len


class MingPreprocessor:
    """Preprocessor for Ming-Omni model.

    Handles:
    - Chat template formatting with <role>HUMAN</role> / <role>ASSISTANT</role>
    - Audio input loading and mel-spectrogram extraction
    - Placeholder token insertion for audio segments
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._config = load_ming_config(model_path)
        self._tokenizer = load_ming_tokenizer(model_path)
        self._audio_config = self._config.audio_config

        # Resolve special token IDs
        self._audio_patch_id = self._tokenizer.convert_tokens_to_ids(AUDIO_PATCH)
        self._audio_start_id = self._tokenizer.convert_tokens_to_ids(AUDIO_START)
        self._audio_end_id = self._tokenizer.convert_tokens_to_ids(AUDIO_END)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        """Process a chat completion request into pipeline state."""
        request = payload.request
        raw_inputs = request.inputs if hasattr(request, "inputs") else {}
        if isinstance(raw_inputs, list):
            # OpenAI API passes messages list directly as inputs
            messages = raw_inputs
            audio_urls = []
        else:
            messages = raw_inputs.get("messages", [])
            audio_urls = raw_inputs.get("audios", [])

        # Load audio files concurrently (load_audio_path returns 16kHz np.ndarray)
        waveforms: list[np.ndarray] = []
        if audio_urls:
            tasks = [
                asyncio.to_thread(load_audio_path, url, target_sr=WHISPER_SAMPLE_RATE)
                for url in audio_urls
            ]
            waveforms = [
                a
                for a in await asyncio.gather(*tasks, return_exceptions=True)
                if isinstance(a, np.ndarray)
            ]

        # --- Compute mel features FIRST so we know exact placeholder counts ---
        mel_features_list: list[torch.Tensor] = []
        mel_lengths_list: list[int] = []
        audio_token_counts: list[int] = []

        for waveform in waveforms:
            mel = compute_mel_spectrogram(waveform)
            mel_features_list.append(torch.from_numpy(mel).float())
            mel_lengths_list.append(mel.shape[0])
            audio_token_counts.append(
                estimate_audio_feature_length(
                    mel.shape[0],
                    getattr(self._audio_config, "ds_kernel_size", 1),
                    getattr(self._audio_config, "ds_stride", 1),
                )
            )

        # --- Build prompt with correct placeholder counts, then tokenize ---
        prompt_text, audio_positions = self._build_prompt(
            messages, audio_token_counts=audio_token_counts
        )
        input_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        prompt: PromptInputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "prompt_text": prompt_text,
        }

        # --- Prepare audio encoder inputs ---
        # Always include audio_encoder key so that the aggregated input handler
        # (which waits for ALL configured sources) receives data from every source.
        encoder_inputs: dict[str, dict[str, Any]] = {
            AUDIO_STAGE: {"_skip": True, "_result": {}},
        }
        if mel_features_list:
            placeholder_loc_lens_list = []
            for i, num_tokens in enumerate(audio_token_counts):
                if i < len(audio_positions):
                    placeholder_loc_lens_list.append([audio_positions[i], num_tokens])

            concat_mel = torch.cat(mel_features_list, dim=0).unsqueeze(0)
            mel_lens = torch.tensor([mel_lengths_list], dtype=torch.long)
            placeholder_locs = torch.tensor(
                [placeholder_loc_lens_list], dtype=torch.long
            )

            encoder_inputs[AUDIO_STAGE] = {
                "audio_feats": concat_mel,
                "audio_feats_lengths": mel_lens,
                "audio_placeholder_loc_lens": placeholder_locs,
            }

        state = PipelineState(
            raw_inputs=raw_inputs,
            prompt=prompt,
            encoder_inputs=encoder_inputs,
        )

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        audio_token_counts: list[int] | None = None,
    ) -> tuple[str, list[int]]:
        """Build Ming-Omni chat format prompt with audio placeholders.

        Args:
            messages: Chat messages.
            audio_token_counts: Exact number of <audioPatch> tokens to insert
                per audio segment (computed from mel features). If None or
                shorter than the number of audio items, a fallback of 1 is used.

        Returns:
            prompt_text: The formatted prompt string.
            audio_positions: Token positions where <audioPatch> placeholders start.
        """
        counts = audio_token_counts or []
        parts: list[str] = []
        audio_idx = 0

        # System message: match the Jinja template behavior exactly.
        # Always include system prompt + "\n\ndetailed thinking off".
        # If no explicit system message, use the default system prompt.
        has_system = messages and messages[0].get("role") == "system"
        if has_system:
            system_content = messages[0].get("content", DEFAULT_SYSTEM_PROMPT)
            parts.append(
                f"{ROLE_SYSTEM}{system_content}\n\ndetailed thinking off{ROLE_END}"
            )
        else:
            # Official Jinja template always includes the default system prompt
            parts.append(
                f"{ROLE_SYSTEM}{DEFAULT_SYSTEM_PROMPT}\n\ndetailed thinking off{ROLE_END}"
            )

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            role_tag = ROLE_HUMAN if role == "user" else ROLE_ASSISTANT

            if isinstance(content, str):
                parts.append(f"{role_tag}{content}{ROLE_END}")
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text_parts.append(item.get("text", ""))
                        elif item_type in ("audio_url", "input_audio"):
                            n_tokens = (
                                counts[audio_idx] if audio_idx < len(counts) else 1
                            )
                            placeholder = (
                                f"{AUDIO_START}"
                                + AUDIO_PATCH * n_tokens
                                + f"{AUDIO_END}"
                            )
                            text_parts.append(placeholder)
                            audio_idx += 1
                    elif isinstance(item, str):
                        text_parts.append(item)
                parts.append(f"{role_tag}{''.join(text_parts)}{ROLE_END}")
            else:
                parts.append(f"{role_tag}{content}{ROLE_END}")

        # Add assistant prefix for generation
        parts.append(ROLE_ASSISTANT)

        prompt_text = "".join(parts)

        # Find audio placeholder positions in tokenized prompt
        audio_positions: list[int] = []
        tokens = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        for i, tok in enumerate(tokens):
            if tok == self._audio_start_id:
                # The audio patch tokens start after <audio>
                audio_positions.append(i + 1)

        return prompt_text, audio_positions
