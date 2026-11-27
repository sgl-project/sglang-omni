# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CAM++ speaker encoder layers, adapted from 3D-Speaker."""

from typing import Literal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    nonlinear = nn.Sequential()
    for layer_name in config_str.split("-"):
        if layer_name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif layer_name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif layer_name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif layer_name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({layer_name}).")
    return nonlinear


def statistics_pooling(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True,
    eps: float = 1e-2,
) -> torch.Tensor:
    mean = x.mean(dim=dim)
    standard_deviation = x.std(dim=dim, unbiased=unbiased)
    statistics = torch.cat([mean, standard_deviation], dim=-1)
    if keepdim:
        statistics = statistics.unsqueeze(dim=dim)
    else:
        pass
    return statistics


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), f"Expect equal paddings, but got even kernel size ({kernel_size})"
            padding = (kernel_size - 1) // 2 * dilation
        else:
            pass
        self.linear: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear: nn.Sequential = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        self.linear_local: nn.Conv1d = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1: nn.Conv1d = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.linear2: nn.Conv1d = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        attention_weights = self.sigmoid(self.linear2(context))
        return y * attention_weights

    def seg_pooling(
        self,
        x: torch.Tensor,
        seg_len: int = 100,
        stype: Literal["avg", "max"] = "avg",
    ) -> torch.Tensor:
        segment_length = seg_len
        pooling_type = stype
        if pooling_type == "avg":
            segments = F.avg_pool1d(
                x, kernel_size=segment_length, stride=segment_length, ceil_mode=True
            )
        elif pooling_type == "max":
            segments = F.max_pool1d(
                x, kernel_size=segment_length, stride=segment_length, ceil_mode=True
            )
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = segments.shape
        segments = (
            segments.unsqueeze(-1)
            .expand(*shape, segment_length)
            .reshape(*shape[:-1], -1)
        )
        segments = segments[..., : x.shape[-1]]
        return segments


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        assert (
            kernel_size % 2 == 1
        ), f"Expect equal paddings, but got even kernel size ({kernel_size})"
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient: bool = memory_efficient
        self.nonlinear1: nn.Sequential = get_nonlinear(config_str, in_channels)
        self.linear1: nn.Conv1d = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2: nn.Sequential = get_nonlinear(config_str, bn_channels)
        self.cam_layer: CAMLayer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.nonlinear: nn.Sequential = get_nonlinear(config_str, in_channels)
        self.linear: nn.Conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.linear: nn.Conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear: nn.Sequential = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.conv2: nn.Conv2d = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)

        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = F.relu(self.bn1(self.conv1(x)))
        hidden_states = self.bn2(self.conv2(hidden_states))
        hidden_states += self.shortcut(x)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class FCM(nn.Module):
    def __init__(
        self,
        block: type[BasicResBlock] = BasicResBlock,
        num_blocks: tuple[int, int] = (2, 2),
        m_channels: int = 32,
        feat_dim: int = 80,
    ) -> None:
        super().__init__()
        self.in_planes: int = m_channels
        self.conv1: nn.Conv2d = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(m_channels)

        self.layer1: nn.Sequential = self.make_layer(
            block, m_channels, num_blocks[0], stride=2
        )
        self.layer2: nn.Sequential = self.make_layer(
            block, m_channels, num_blocks[1], stride=2
        )

        self.conv2: nn.Conv2d = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(m_channels)
        self.out_channels: int = m_channels * (feat_dim // 8)

    def make_layer(
        self, block: type[BasicResBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[BasicResBlock] = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        hidden_states = F.relu(self.bn1(self.conv1(x)))
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.layer2(hidden_states)
        hidden_states = F.relu(self.bn2(self.conv2(hidden_states)))

        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1] * shape[2], shape[3])
        return hidden_states
