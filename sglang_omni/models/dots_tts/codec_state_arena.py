# SPDX-License-Identifier: Apache-2.0
"""Slot-indexed history for incremental dots.tts AudioVAE decoding."""

from __future__ import annotations

import torch


class DotsCodecStateArena:
    """The recent inputs that each decoder stage needs to continue a stream.

    Every tensor is indexed by vocoder slot first. A cold decode overwrites a
    slot's rows, so a released slot needs no reset before reuse.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        latent_channels: int,
        conv_pre_context: int,
        upsample_channels: list[int],
        stage_channels: list[int],
        stage_contexts: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        with torch.no_grad():
            # note (0xtoward): the last latent frames that conv_pre looks back on.
            self.conv_pre_history = torch.zeros(
                num_slots, latent_channels, conv_pre_context, device=device, dtype=dtype
            )
            # note (0xtoward): one input sample per causal transposed conv, enough
            # to finish the outputs that overlap the next input.
            self.upsample_histories = [
                torch.zeros(num_slots, channels, 1, device=device, dtype=dtype)
                for channels in upsample_channels
            ]
            # note (0xtoward): the left receptive field of each stage's resblocks.
            self.stage_histories = [
                torch.zeros(num_slots, channels, context, device=device, dtype=dtype)
                for channels, context in zip(stage_channels, stage_contexts)
            ]

    def tensors(self) -> list[torch.Tensor]:
        return [self.conv_pre_history, *self.upsample_histories, *self.stage_histories]


__all__ = ["DotsCodecStateArena"]
