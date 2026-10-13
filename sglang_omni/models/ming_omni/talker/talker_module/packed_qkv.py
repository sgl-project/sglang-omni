# SPDX-License-Identifier: Apache-2.0
"""Packed QKV projection and checkpoint shard loading for Ming talkers."""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.weight_loader import default_weight_loader

PACKED_QKV_SHARD_IDS: tuple[str, str, str] = ("q", "k", "v")


class PackedQKVLinear(nn.Linear):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, 3 * output_size)
        self.output_size = output_size
        self.weight.weight_loader = self.weight_loader
        self.bias.weight_loader = self.weight_loader

    def weight_loader(
        self,
        parameter: nn.Parameter,
        checkpoint_weight: torch.Tensor,
        shard_id: str | None = None,
    ) -> None:
        if shard_id is None:
            default_weight_loader(parameter, checkpoint_weight)
            return
        else:
            pass

        shard_index = PACKED_QKV_SHARD_IDS.index(shard_id)
        shard = parameter.data.narrow(
            0,
            shard_index * self.output_size,
            self.output_size,
        )
        default_weight_loader(shard, checkpoint_weight)


def load_packed_qkv_shard(
    checkpoint_name: str,
    checkpoint_weight: torch.Tensor,
    parameters: dict[str, nn.Parameter],
) -> tuple[str, str] | None:
    for shard_id in PACKED_QKV_SHARD_IDS:
        source_name = f".to_{shard_id}."
        if source_name not in checkpoint_name:
            continue
        else:
            pass
        target_name = checkpoint_name.replace(source_name, ".to_qkv.", 1)
        parameter = parameters.get(target_name)
        if parameter is None:
            continue
        else:
            pass
        parameter.weight_loader(parameter, checkpoint_weight, shard_id)
        return target_name, shard_id
    return None
