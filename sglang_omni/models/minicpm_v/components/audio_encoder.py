# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 2.6 Whisper-based audio encoder.

This module provides the audio encoder component for MiniCPM-o 2.6,
which uses a Whisper-based architecture with the `apm.` weight prefix.

The audio encoder follows the same pattern as Qwen3OmniAudioEncoder but
is adapted for MiniCPM-o's Whisper-based audio processing model.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from sglang_omni.models.minicpm_v.components.common import (
    AUDIO_PREFIX,
    load_audio_config,
    load_minicpm_config,
)
from sglang_omni.models.weight_loader import load_module, resolve_dtype

logger = logging.getLogger(__name__)


def _get_whisper_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output sequence lengths after Whisper's convolution downsampling.

    Whisper uses two 1D convolution layers that each halve the sequence length:
    - Conv1: stride=1, no reduction
    - Conv2: stride=2, halves length

    The output length formula matches HF Whisper implementation.
    """
    # Whisper convolution downsampling: floor((input_length - 1) / 2) + 1
    # For the two conv layers: floor(floor((L - 1) / 1) - 1) / 2) + 1 = floor((L - 1) / 2)
    return (input_lengths - 1) // 2


def _build_audio_tower(
    model_path: str,
    *,
    audio_cfg: Any,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    """Build and load the Whisper audio encoder model.

    MiniCPM-o uses a standard Whisper encoder with projection layers
    to match the LLM hidden dimension.
    """
    try:
        # Try to load the MiniCPM-specific audio encoder if available
        from transformers import AutoModel

        # Load the full model and extract the audio encoder
        full_config = load_minicpm_config(model_path)
        if hasattr(full_config, "audio_config") or hasattr(full_config, "apm_config"):
            # MiniCPM-o has custom audio encoder, try to instantiate it
            from transformers import WhisperModel

            audio_tower = WhisperModel(audio_cfg).encoder
        else:
            # Fallback to standard Whisper encoder
            from transformers import WhisperModel

            audio_tower = WhisperModel(audio_cfg).encoder
    except Exception as e:
        logger.warning(f"Failed to build audio tower with custom config: {e}")
        # Fallback: use standard Whisper encoder
        from transformers import WhisperModel

        audio_tower = WhisperModel(audio_cfg).encoder

    # Load weights with prefix filtering
    return load_module(
        audio_tower,
        model_path,
        prefix=AUDIO_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=False,  # Allow missing keys for projection layers
    )


class MiniCPMOAudioEncoder(nn.Module):
    """Whisper-based audio encoder for MiniCPM-o 2.6.

    This encoder processes audio inputs through a Whisper encoder and
    produces embeddings compatible with the MiniCPM LLM backbone.

    The forward pass accepts mel-spectrogram features and returns
    audio embeddings with corresponding output lengths.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        self._device = torch.device(device)
        self._dtype = torch_dtype

        # Load audio config
        audio_cfg = load_audio_config(model_path)
        if audio_cfg is None:
            raise ValueError(
                f"No audio config found in {model_path}. "
                "This model may not support audio input (MiniCPM-V vs MiniCPM-o)."
            )

        self.audio_tower = _build_audio_tower(
            model_path,
            audio_cfg=audio_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Store config for output length computation
        self._audio_cfg = audio_cfg

        # Optional: projection layer to match LLM hidden dimension
        # This may be loaded from the checkpoint or created dynamically
        self.proj = None
        llm_hidden_size = getattr(audio_cfg, "output_dim", None)
        encoder_hidden_size = getattr(audio_cfg, "d_model", None)
        if llm_hidden_size and encoder_hidden_size and llm_hidden_size != encoder_hidden_size:
            self.proj = nn.Linear(encoder_hidden_size, llm_hidden_size, bias=False)
            if torch_dtype:
                self.proj = self.proj.to(dtype=torch_dtype)
            self.proj = self.proj.to(device=device)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        for p in self.audio_tower.parameters():
            return p.dtype
        return torch.float32

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process audio features through the Whisper encoder.

        Args:
            input_features: Mel-spectrogram tensor [batch, time, mel_bins] or [batch, mel_bins, time]
            feature_attention_mask: Optional attention mask [batch, time]
            audio_feature_lengths: Optional lengths tensor [batch]

        Returns:
            Dictionary containing:
            - audio_embeds: Encoded audio representations [batch, seq_len, hidden_dim]
            - audio_feature_lengths: Original input lengths
            - audio_output_lengths: Downsampled output lengths
        """
        # Compute lengths from attention mask if not provided
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            # Mask the input features if mask provided
            # Whisper expects [batch, mel_bins, time] format
            if input_features.dim() == 3:
                # Transpose if in [batch, time, mel_bins] format
                if input_features.shape[-1] != feature_attention_mask.shape[-1]:
                    input_features = input_features.transpose(1, 2)
                # Apply mask (zero out invalid positions)
                input_features = input_features * feature_attention_mask.unsqueeze(1).to(
                    input_features.dtype
                )

        if audio_feature_lengths is None:
            # Infer lengths from input shape
            if input_features.dim() == 3:
                audio_feature_lengths = torch.full(
                    (input_features.shape[0],),
                    input_features.shape[-1],
                    dtype=torch.long,
                    device=self._device,
                )
            else:
                raise ValueError(
                    "audio_feature_lengths or feature_attention_mask is required"
                )

        # Move to device
        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        input_features = input_features.to(device=self._device, dtype=self.dtype)

        # Ensure correct format for Whisper: [batch, mel_bins, time]
        if input_features.dim() == 3 and input_features.shape[1] > input_features.shape[2]:
            # Likely [batch, time, mel_bins], transpose to [batch, mel_bins, time]
            input_features = input_features.transpose(1, 2)

        # Forward through Whisper encoder
        outputs = self.audio_tower(
            input_features,
            attention_mask=None,  # Whisper handles padding internally
            output_hidden_states=False,
            return_dict=True,
        )

        audio_embeds = outputs.last_hidden_state

        # Apply projection if present
        if self.proj is not None:
            audio_embeds = self.proj(audio_embeds)

        # Compute output lengths after downsampling
        audio_output_lengths = _get_whisper_output_lengths(audio_feature_lengths)

        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
