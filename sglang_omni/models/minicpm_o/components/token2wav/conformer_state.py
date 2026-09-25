# SPDX-License-Identifier: Apache-2.0
"""Explicit Conformer histories and the packed vocoder cache boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, kw_only=True)
class ConvState:
    """Empty history enables streaming; no state disables caching."""

    history: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class AttentionState:
    """Keys and values concatenated along the last axis, in batch/head/time order."""

    history: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class ConformerState:
    lookahead: ConvState = field(default_factory=ConvState)
    upsample: ConvState = field(default_factory=ConvState)
    attention: tuple[AttentionState, ...] = ()
    up_attention: tuple[AttentionState, ...] = ()

    @classmethod
    def from_packed(
        cls,
        convolution: torch.Tensor | None,
        attention: torch.Tensor | None,
        num_blocks: int,
        stride: int,
    ) -> ConformerState:
        """Borrow read-only histories while retaining the packed tensor strides."""
        if attention is None:
            return cls()
        else:
            assert convolution is not None
            history_length = attention.shape[3] // stride
            return cls(
                lookahead=ConvState(history=convolution[:, :, :2]),
                upsample=ConvState(history=convolution[:, :, 2:]),
                attention=tuple(
                    AttentionState(history=layer[:, :, :history_length])
                    for layer in attention[:num_blocks]
                ),
                up_attention=tuple(
                    AttentionState(history=layer[:, :, : history_length * stride])
                    for layer in attention[num_blocks:]
                ),
            )

    def to_packed(self, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lookahead.history is not None
        assert self.upsample.history is not None
        convolution = torch.cat((self.lookahead.history, self.upsample.history), dim=2)
        histories: list[torch.Tensor] = []
        for layer in self.attention + self.up_attention:
            assert layer.history is not None
            histories.append(layer.history)
        # note (Junnan Li): The vocoder ABI repeats whole histories along time.
        first = torch.stack(histories[: len(self.attention)]).repeat(1, 1, 1, stride, 1)
        second = torch.stack(histories[len(self.attention) :])
        return convolution, torch.cat((first, second))
