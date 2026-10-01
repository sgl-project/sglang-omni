# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni code2wav whose convolutions run channels last."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavDecoderResidualUnit,
    Qwen3OmniMoeConvNeXtBlock,
    SnakeBeta,
)

from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
)
from sglang_omni.utils.snake_beta import FusedSnakeBeta


def causal_conv(
    module: Qwen3OmniMoeCausalConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal conv of a (B, L, C) activation, returned as (B, L, C_out)."""
    conv = module.conv
    assert conv.stride == (1,), "code2wav causal convs are stride 1"
    length = hidden_states.shape[1]
    padded = F.pad(hidden_states, (0, 0, module.padding, (-length) % conv.dilation[0]))
    return channels_last_conv1d(padded, conv, conv.weight, length)


def causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal transposed conv of a (B, L, C) activation, returned as (B, L_out, C_out)."""
    conv = module.conv
    output = channels_last_conv_transpose1d(hidden_states, conv, conv.weight, conv.bias)
    return output[:, module.left_pad : output.shape[1] - module.right_pad].contiguous()


def channels_last_block(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run one code2wav module on a (B, L, C) activation as its forward runs on (B, C, L)."""
    if isinstance(module, Qwen3OmniMoeCausalConvNet):
        return causal_conv(module, hidden_states)
    elif isinstance(module, Qwen3OmniMoeCausalTransConvNet):
        return causal_transconv(module, hidden_states)
    elif isinstance(module, (SnakeBeta, FusedSnakeBeta)):
        return module(hidden_states.transpose(1, 2)).transpose(1, 2)
    elif isinstance(module, Qwen3OmniMoeConvNeXtBlock):
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel,
        # faster than cuDNN's channels-last depthwise and free of its per-shape setup.
        normed = module.norm(
            module.dwconv(hidden_states.transpose(1, 2)).transpose(1, 2)
        )
        return hidden_states + module.gamma * module.pwconv2(
            module.act(module.pwconv1(normed))
        )
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderResidualUnit):
        output = hidden_states
        for block in (module.act1, module.conv1, module.act2, module.conv2):
            output = channels_last_block(block, output)
        return output + hidden_states
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderBlock):
        for block in module.block:
            hidden_states = channels_last_block(block, hidden_states)
        return hidden_states
    else:
        raise TypeError(
            f"no channels-last form for code2wav module {type(module).__name__}"
        )


class Qwen3OmniCode2Wav(Qwen3OmniMoeCode2Wav):
    """Code2Wav that carries (B, L, C) activations once its convs are channels last."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None:
        super().__init__(config)
        self.is_channels_last = False

    def use_channels_last(self) -> None:
        """Store every conv weight channels last and run the channels-last forward."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                module.weight.data = channels_last_weight(module)
            else:
                pass
        self.is_channels_last = True

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.is_channels_last:
            return super().forward(codes)
        elif codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, "
                f"got {codes.shape[1]}"
            )
        else:
            pass
        hidden_states = self.code_embedding(codes + self.code_offset).mean(1)
        hidden_states = self.pre_transformer(
            inputs_embeds=hidden_states
        ).last_hidden_state
        for blocks in self.upsample:
            for block in blocks:
                hidden_states = channels_last_block(block, hidden_states)
        for block in self.decoder:
            hidden_states = channels_last_block(block, hidden_states)
        return hidden_states.transpose(1, 2).clamp(min=-1, max=1)
