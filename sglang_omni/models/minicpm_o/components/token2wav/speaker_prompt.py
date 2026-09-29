# SPDX-License-Identifier: Apache-2.0
"""Encoded speaker reference used to condition MiniCPM-o Code2Wav."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, frozen=True)
class SpeakerPrompt:
    """Prompt tokens, their lengths, the speaker embedding, and the prompt mel."""

    prompt_tokens: torch.Tensor
    prompt_token_lengths: torch.Tensor
    speaker_embedding: torch.Tensor
    prompt_mel: torch.Tensor
