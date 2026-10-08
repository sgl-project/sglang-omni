# SPDX-License-Identifier: Apache-2.0
"""Causal convolutions that can run whole or one chunk at a time.

Mimi is a stack of causal convolutions and transposed convolutions. Run over a
whole recording they pad on the left; run chunk by chunk they must remember
the tail of the previous chunk instead, and a transposed convolution must hold
back the outputs that the next chunk still contributes to. The state lives in
a small object the caller owns, so one module can serve many sessions.
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

    def init_state(self):
        return None

    def step(self, x: torch.Tensor, state):
        return self(x)


class ELU(StreamingModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.elu(x)


@dataclass
class ConvState:
    previous: torch.Tensor | None = None
    padded: bool = False


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

    def init_state(self) -> ConvState:
        return ConvState()

    def step(self, x: torch.Tensor, state: ConvState) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)
        else:
            pass
        if not state.padded:
            x = pad1d(x, self.padding_total, 0, self.pad_mode)
            state.padded = True
        else:
            pass
        if state.previous is not None:
            x = torch.cat([state.previous, x], dim=-1)
        else:
            pass
        kernel, stride = self.effective_kernel_size, self.stride
        num_frames = max(0, (x.shape[-1] - kernel) // stride + 1)
        consumed = num_frames * stride
        state.previous = x[..., consumed:]
        if num_frames == 0:
            return x.new_empty(x.shape[0], self.conv.out_channels, 0)
        else:
            pass
        return self.conv(x[..., : (num_frames - 1) * stride + kernel])


@dataclass
class ConvTransposeState:
    partial: torch.Tensor | None = None


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

    def init_state(self) -> ConvTransposeState:
        return ConvTransposeState()

    def step(self, x: torch.Tensor, state: ConvTransposeState) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x.new_empty(x.shape[0], self.convtr.out_channels, 0)
        else:
            pass
        out = self.convtr(x)
        partial = state.partial
        if partial is not None:
            width = partial.shape[-1]
            if self.convtr.bias is not None:
                # Note (wilsonzheng0327): Both renders added the bias; keep it once.
                out[..., :width] += partial - self.convtr.bias[:, None]
            else:
                out[..., :width] += partial
        else:
            pass
        keep = out.shape[-1] - self.padding_total
        state.partial = out[..., keep:]
        return out[..., :keep]


__all__ = [
    "CausalConv1d",
    "CausalConvTranspose1d",
    "ConvState",
    "ConvTransposeState",
]
