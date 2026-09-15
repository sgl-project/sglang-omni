# SPDX-License-Identifier: Apache-2.0
"""A fused parameter is not fully loaded merely because one shard was seen."""

import pytest
import torch
from torch import nn

from sglang_omni.models.voxcpm2.sglang_model import VoxCPM2SGLangModel


def model_with_projection(kind):
    model = VoxCPM2SGLangModel.__new__(VoxCPM2SGLangModel)
    nn.Module.__init__(model)
    model.num_base_layers = 1
    model.feat_encoder = nn.Linear(1, 1, bias=False)
    model.feat_decoder = nn.Linear(1, 1, bias=False)
    layer = nn.Module()
    parent = nn.Module()
    if kind == "qkv":
        shard_ids = ["q", "k", "v"]
        separate = ["q_proj", "k_proj", "v_proj"]
        container, fused = "self_attn", "qkv_proj"
    else:
        shard_ids = [0, 1]
        separate = ["gate_proj", "up_proj"]
        container, fused = "mlp", "gate_up_proj"
    projection = nn.Linear(2, 2 * len(shard_ids), bias=False)
    setattr(parent, fused, projection)
    setattr(layer, container, parent)
    model.layers = nn.ModuleList([layer])

    def load(parameter, value, shard_id):
        index = shard_ids.index(shard_id)
        with torch.no_grad():
            parameter[2 * index : 2 * index + 2].copy_(value)

    projection.weight.weight_loader = load
    weights = [
        (
            f"base_lm.layers.0.{container}.{name}.weight",
            torch.full((2, 2), float(i + 1)),
        )
        for i, name in enumerate(separate)
    ]
    other = [
        ("feat_encoder.weight", torch.ones(1, 1)),
        ("feat_decoder.weight", torch.ones(1, 1)),
    ]
    return model, projection, weights, other


@pytest.mark.parametrize("kind", ["qkv", "gate_up"])
def test_a_partially_loaded_fused_weight_is_rejected(kind):
    model, _, weights, other = model_with_projection(kind)
    with pytest.raises(ValueError, match="without shards"):
        model.load_weights(weights[:-1] + other)


@pytest.mark.parametrize("kind", ["qkv", "gate_up"])
def test_all_shards_land_in_their_own_fused_slices(kind):
    model, projection, weights, other = model_with_projection(kind)
    model.load_weights(list(reversed(weights)) + other)
    for index in range(len(weights)):
        torch.testing.assert_close(
            projection.weight[2 * index : 2 * index + 2],
            torch.full((2, 2), float(index + 1)),
            rtol=0,
            atol=0,
        )
