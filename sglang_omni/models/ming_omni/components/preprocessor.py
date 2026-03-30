# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for Ming-Omni: tokenize text, extract audio mel features."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.common import load_ming_config
from sglang_omni.models.ming_omni.io import PipelineState, PromptInputs
from sglang_omni.models.ming_omni.pipeline.next_stage import AUDIO_STAGE
from sglang_omni.preprocessing.audio import load_audio
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

# Ming-Omni chat template tokens
ROLE_HUMAN = "<role>HUMAN</role>"
ROLE_ASSISTANT = "<role>ASSISTANT</role>"
ROLE_SYSTEM = "<role>SYSTEM</role>"
DEFAULT_SYSTEM_PROMPT = "你是一个友好的AI助手。\n"

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
    ds_kernel_size: int = 3,
    ds_stride: int = 2,
) -> int:
    """Estimate the number of audio tokens after Whisper encoder + Conv1d projection.

    The Whisper encoder preserves sequence length (conv layers are internal).
    The projection Conv1d downsamples by ds_stride.
    """
    # Whisper encoder internal conv: no length change in output
    whisper_out_len = mel_frames
    # Projection Conv1d: (L - kernel + 2*padding) // stride + 1
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
        from transformers import AutoTokenizer

        self._model_path = model_path
        self._config = load_ming_config(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._audio_config = self._config.audio_config

        # Resolve special token IDs
        self._audio_patch_id = self._tokenizer.convert_tokens_to_ids(AUDIO_PATCH)
        self._audio_start_id = self._tokenizer.convert_tokens_to_ids(AUDIO_START)
        self._audio_end_id = self._tokenizer.convert_tokens_to_ids(AUDIO_END)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        """Process a chat completion request into pipeline state."""
        request = payload.request
        raw_inputs = request.inputs if hasattr(request, "inputs") else {}
        messages = raw_inputs.get("messages", [])
        audio_urls = raw_inputs.get("audios", [])

        # Load audio files concurrently
        audio_data = []
        if audio_urls:
            tasks = [asyncio.to_thread(load_audio, url) for url in audio_urls]
            audio_data = await asyncio.gather(*tasks, return_exceptions=True)
            audio_data = [
                a for a in audio_data if not isinstance(a, Exception)
            ]

        # Build prompt text and tokenize
        prompt_text, audio_positions = self._build_prompt(messages, len(audio_data))
        input_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        prompt: PromptInputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "prompt_text": prompt_text,
        }

        # Prepare audio encoder inputs if audio is present
        encoder_inputs: dict[str, dict[str, Any]] = {}
        if audio_data:
            mel_features_list = []
            mel_lengths_list = []
            placeholder_loc_lens_list = []

            for i, (waveform, sr) in enumerate(audio_data):
                # Resample to 16kHz if needed
                if sr != WHISPER_SAMPLE_RATE:
                    import torchaudio
                    waveform_tensor = torch.from_numpy(waveform).float()
                    if waveform_tensor.dim() == 1:
                        waveform_tensor = waveform_tensor.unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, WHISPER_SAMPLE_RATE)
                    waveform_tensor = resampler(waveform_tensor)
                    waveform = waveform_tensor.squeeze(0).numpy()

                mel = compute_mel_spectrogram(waveform)
                mel_features_list.append(torch.from_numpy(mel).float())
                mel_lengths_list.append(mel.shape[0])

                # Compute placeholder location for this audio segment
                num_audio_tokens = estimate_audio_feature_length(
                    mel.shape[0],
                    self._audio_config.ds_kernel_size,
                    self._audio_config.ds_stride,
                )
                if i < len(audio_positions):
                    start_pos = audio_positions[i]
                    placeholder_loc_lens_list.append([start_pos, num_audio_tokens])

            # Pad mel features to same length and stack
            max_mel_len = max(m.shape[0] for m in mel_features_list)
            padded_mels = []
            for m in mel_features_list:
                if m.shape[0] < max_mel_len:
                    pad = torch.zeros(max_mel_len - m.shape[0], m.shape[1])
                    m = torch.cat([m, pad], dim=0)
                padded_mels.append(m)

            audio_feats = torch.stack(padded_mels)  # [num_audios, max_len, n_mels]
            audio_feats_lengths = torch.tensor(
                [[l] for l in mel_lengths_list], dtype=torch.long
            )  # [batch, num_audios_per_sample]

            # For single-batch: wrap into batch format
            # audio_feats: [1, total_mel_len, n_mels] (concatenated)
            # audio_feats_lengths: [1, num_audios]
            if len(mel_features_list) > 0:
                concat_mel = torch.cat(mel_features_list, dim=0).unsqueeze(0)
                mel_lens = torch.tensor(
                    [mel_lengths_list], dtype=torch.long
                )
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
        num_audios: int,
    ) -> tuple[str, list[int]]:
        """Build Ming-Omni chat format prompt with audio placeholders.

        Returns:
            prompt_text: The formatted prompt string
            audio_positions: List of token positions where <audioPatch> placeholders start
        """
        parts: list[str] = []
        audio_idx = 0
        audio_placeholder_texts: list[str] = []

        # System message
        parts.append(f"{ROLE_SYSTEM}{DEFAULT_SYSTEM_PROMPT}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Override system prompt
                parts[0] = f"{ROLE_SYSTEM}{content}\n"
                continue

            role_tag = ROLE_HUMAN if role == "user" else ROLE_ASSISTANT

            if isinstance(content, str):
                parts.append(f"{role_tag}{content}")
            elif isinstance(content, list):
                # Multimodal content list
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text_parts.append(item.get("text", ""))
                        elif item_type in ("audio_url", "input_audio"):
                            # Insert audio placeholder
                            num_placeholder_tokens = 100  # estimated, will be adjusted
                            placeholder = (
                                f"{AUDIO_START}"
                                + AUDIO_PATCH * num_placeholder_tokens
                                + f"{END_OF_AUDIO}{AUDIO_END}"
                            )
                            text_parts.append(placeholder)
                            audio_idx += 1
                    elif isinstance(item, str):
                        text_parts.append(item)
                parts.append(f"{role_tag}{''.join(text_parts)}")
            else:
                parts.append(f"{role_tag}{content}")

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
