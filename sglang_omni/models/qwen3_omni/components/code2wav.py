# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni code2wav whose convolutions run channels last on CUDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
)


def causal_conv(
    module: Qwen3OmniMoeCausalConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal conv of a (B, L, C) activation, returned as (B, L, C_out)."""
    conv = module.conv
    assert conv.stride == (1,), "code2wav causal convs are stride 1"
    batch_size, length, channels = hidden_states.shape
    dilation = conv.dilation[0]
    if dilation == 1:
        padded = F.pad(hidden_states, (0, 0, module.padding, 0))
        output = F.conv2d(
            padded.transpose(1, 2).unsqueeze(2),
            conv.weight.unsqueeze(2),
            conv.bias,
            groups=conv.groups,
        )
        return output.squeeze(2).transpose(1, 2)
    else:
        # note (ratish): cuDNN has no fast channels-last engine for some dilated
        # shapes; viewed as (B, C, L / d, d) the conv is undilated along L / d, one
        # phase per column, without copying the phases apart.
        padded = F.pad(hidden_states, (0, 0, module.padding, (-length) % dilation))
        output = F.conv2d(
            padded.view(batch_size, -1, dilation, channels).permute(0, 3, 1, 2),
            conv.weight.unsqueeze(3),
            conv.bias,
            groups=conv.groups,
        )
        return (
            output.permute(0, 2, 3, 1)
            .reshape(batch_size, -1, output.shape[1])[:, :length]
            .contiguous()
        )


def causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal transposed conv of a (B, L, C) activation, returned as (B, L_out, C_out)."""
    conv = module.conv
    output = (
        F.conv_transpose2d(
            hidden_states.transpose(1, 2).unsqueeze(2),
            conv.weight.unsqueeze(2),
            conv.bias,
            stride=(1, conv.stride[0]),
        )
        .squeeze(2)
        .transpose(1, 2)
    )
    return output[:, module.left_pad : output.shape[1] - module.right_pad].contiguous()


def snake_activation(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run a (B, C, L) SnakeBeta on a (B, L, C) activation."""
    return module(hidden_states.transpose(1, 2)).transpose(1, 2)


class Qwen3OmniCode2Wav(Qwen3OmniMoeCode2Wav):
    """Code2Wav carrying (B, L, C) activations so cuDNN reads every conv channels last."""

    def lay_out_convs_channels_last(self) -> None:
        """Store every conv weight channels last so no call reformats it."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                module.weight.data = (
                    module.weight.data.transpose(1, 2).contiguous().transpose(1, 2)
                )
            else:
                pass

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, "
                f"got {codes.shape[1]}"
            )
        elif codes.device.type != "cuda":
            return super().forward(codes)
        else:
            pass
        hidden_states = self.code_embedding(codes + self.code_offset).mean(1)
        hidden_states = self.pre_transformer(
            inputs_embeds=hidden_states
        ).last_hidden_state
        for transconv, convnext in self.upsample:
            hidden_states = causal_transconv(transconv, hidden_states)
            residual = hidden_states
            # note (ratish): the depthwise conv stays on PyTorch's own NCL kernel, faster
            # than cuDNN's channels-last depthwise and free of its per-shape plan setup.
            hidden_states = convnext.dwconv(hidden_states.transpose(1, 2))
            hidden_states = convnext.norm(hidden_states.transpose(1, 2))
            hidden_states = convnext.pwconv2(
                convnext.act(convnext.pwconv1(hidden_states))
            )
            hidden_states = residual + convnext.gamma * hidden_states
        waveform = causal_conv(self.decoder[0], hidden_states)
        for decoder_block in self.decoder[1:-2]:
            waveform = snake_activation(decoder_block.block[0], waveform)
            waveform = causal_transconv(decoder_block.block[1], waveform)
            for residual_unit in decoder_block.block[2:]:
                residual = waveform
                waveform = causal_conv(
                    residual_unit.conv1,
                    snake_activation(residual_unit.act1, waveform),
                )
                waveform = causal_conv(
                    residual_unit.conv2,
                    snake_activation(residual_unit.act2, waveform),
                )
                waveform = waveform + residual
        waveform = causal_conv(
            self.decoder[-1], snake_activation(self.decoder[-2], waveform)
        )
        return waveform.transpose(1, 2).clamp(min=-1, max=1)
