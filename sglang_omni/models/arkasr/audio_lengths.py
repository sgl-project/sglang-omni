# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR audio-token-count helper.

Mirrors the checkpoint processor's ``calculate_audio_token_count``:
    downsampled = (mel_frames + 1) // 2      # WhisperEncoder conv2 stride-2
    tokens      = downsampled // merge_factor # MLP adapter frame merge (default 4)
at least 1.
"""

from __future__ import annotations


def arkasr_num_audio_tokens(mel_frames: int, merge_factor: int = 4) -> int:
    downsampled = (int(mel_frames) + 1) // 2
    merged = downsampled // max(int(merge_factor), 1)
    return max(int(merged), 1)


def arkasr_audio_token_lengths(mel_frame_lengths, merge_factor: int = 4):
    """Vectorized/list form used by the processor's feat-extract-output-lengths hook."""
    try:
        import torch

        if isinstance(mel_frame_lengths, torch.Tensor):
            downsampled = (mel_frame_lengths + 1) // 2
            merged = torch.div(
                downsampled, max(int(merge_factor), 1), rounding_mode="floor"
            )
            return merged.clamp(min=1)
    except Exception:
        pass
    return [arkasr_num_audio_tokens(m, merge_factor) for m in mel_frame_lengths]


__all__ = ["arkasr_num_audio_tokens", "arkasr_audio_token_lengths"]
