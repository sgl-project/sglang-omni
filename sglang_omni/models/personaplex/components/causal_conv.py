# SPDX-License-Identifier: Apache-2.0
"""Causal convolutions that can run whole or one chunk at a time.

Mimi is a stack of causal convolutions and transposed convolutions. Run over a
whole recording they pad on the left; run chunk by chunk they must remember
the tail of the previous chunk instead, and a transposed convolution must hold
back the outputs that the next chunk still contributes to. The state lives in
a small object the caller owns, so one module can serve many sessions. Its
tensors keep a fixed size and are only updated in place, so a device graph
captured over a step keeps reading the same state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


def pad1d(x: torch.Tensor, left: int, right: int, mode: str) -> torch.Tensor:
    if left == 0 and right == 0:
        return x
    else:
        pass
    if mode == "reflect":
        # Note (wilsonzheng0327): Reflection needs more samples than it pads.
        max_pad = max(left, right)
        extra = 0
        if x.shape[-1] <= max_pad:
            extra = max_pad - x.shape[-1] + 1
            x = functional.pad(x, (0, extra))
        else:
            pass
        padded = functional.pad(x, (left, right), mode="reflect")
        return padded[..., : padded.shape[-1] - extra]
    else:
        pass
    return functional.pad(x, (left, right), mode=mode)


class StreamingModule(nn.Module):
    """A module that also runs chunk by chunk, over state the caller owns.

    Stateless modules inherit these defaults, so a stack of them needs no test
    for which of its members carry state.
    """

    def init_state(self, batch_size: int):
        return None

    def step(self, x: torch.Tensor, state):
        return self(x)


class ELU(StreamingModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.elu(x)


@dataclass
class ConvState:
    """The input tail the next chunk continues from, zeros before the first chunk.

    Replicate padding fills it from the first chunk instead; has_started marks that.
    """

    previous: torch.Tensor
    has_started: torch.Tensor

    def reset(self) -> None:
        self.previous.zero_()
        self.has_started.zero_()


class CausalConv1d(StreamingModule):
    """Conv1d with left padding of effective_kernel - stride samples.

    The whole-sequence path also pads on the right so the last window is
    full; with inputs that are multiples of the stride that padding is zero,
    which is what makes the chunked path land on the same samples.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.pad_mode = pad_mode

    @property
    def stride(self) -> int:
        return self.conv.stride[0]

    @property
    def effective_kernel_size(self) -> int:
        return (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1

    @property
    def padding_total(self) -> int:
        return self.effective_kernel_size - self.stride

    def extra_padding(self, length: int) -> int:
        kernel, stride, padding = (
            self.effective_kernel_size,
            self.stride,
            self.padding_total,
        )
        n_frames = (length - kernel + padding) / stride + 1
        ideal = (math.ceil(n_frames) - 1) * stride + (kernel - padding)
        return ideal - length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad1d(x, self.padding_total, self.extra_padding(x.shape[-1]), self.pad_mode)
        return self.conv(x)

    def init_state(self, batch_size: int) -> ConvState:
        weight = self.conv.weight
        return ConvState(
            previous=weight.new_zeros(
                batch_size, self.conv.in_channels, self.padding_total
            ),
            has_started=torch.zeros((), dtype=torch.bool, device=weight.device),
        )

    def step(self, x: torch.Tensor, state: ConvState) -> torch.Tensor:
        """Chunks must be whole strides, so the carried tail keeps one length."""
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)
        else:
            pass
        assert x.shape[-1] % self.stride == 0, (x.shape, self.stride)
        if self.pad_mode == "replicate":
            state.previous.copy_(
                torch.where(state.has_started, state.previous, x[..., :1])
            )
            state.has_started.fill_(True)
        else:
            assert self.pad_mode == "constant", self.pad_mode
        x = torch.cat([state.previous, x], dim=-1)
        state.previous.copy_(x[..., x.shape[-1] - self.padding_total :])
        return self.conv(x)


@dataclass
class ConvTransposeState:
    """The kernel - stride trailing outputs held back, bias removed, zeros at first."""

    partial: torch.Tensor

    def reset(self) -> None:
        self.partial.zero_()


class CausalConvTranspose1d(StreamingModule):
    """ConvTranspose1d whose kernel - stride trailing outputs are trimmed.

    Chunk by chunk those trailing outputs are not dropped but held back: the
    next chunk overlaps them and adds its own contribution before they leave.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )

    @property
    def padding_total(self) -> int:
        return self.convtr.kernel_size[0] - self.convtr.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        return y[..., : y.shape[-1] - self.padding_total]

    def init_state(self, batch_size: int) -> ConvTransposeState:
        return ConvTransposeState(
            partial=self.convtr.weight.new_zeros(
                batch_size, self.convtr.out_channels, self.padding_total
            )
        )

    def step(self, x: torch.Tensor, state: ConvTransposeState) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.convtr.out_channels, 0)
        else:
            pass
        out = self.convtr(x)
        out[..., : self.padding_total] += state.partial
        keep = out.shape[-1] - self.padding_total
        held_back = out[..., keep:]
        if self.convtr.bias is not None:
            # Note (wilsonzheng0327): Both renders added the bias; keep it once.
            held_back = held_back - self.convtr.bias[:, None]
        else:
            pass
        state.partial.copy_(held_back)
        return out[..., :keep]


__all__ = [
    "CausalConv1d",
    "CausalConvTranspose1d",
    "ConvState",
    "ConvTransposeState",
]
