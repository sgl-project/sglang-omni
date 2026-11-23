# SPDX-License-Identifier: Apache-2.0
"""One qkv and one gate-up weight per decoder layer, for the codec decoders' fused transformers."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def merge_projection_weights(layers: Iterable[torch.nn.Module]) -> None:
    """Give each layer's attention a qkv_weight and its MLP a gate_up_weight."""
    # note (ratish): q, k, v and gate, up become views of one matrix each, so the
    # fused GEMMs read the loaded weights without a second copy.
    for layer in layers:
        attention, mlp = layer.self_attn, layer.mlp
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        attention.qkv_weight = torch.cat(
            [projection.weight.data for projection in projections]
        )
        start = 0
        for projection in projections:
            output_features = projection.weight.shape[0]
            projection.weight.data = attention.qkv_weight[
                start : start + output_features
            ]
            start += output_features
        mlp.gate_up_weight = torch.cat(
            (mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
        )
        mlp.gate_proj.weight.data = mlp.gate_up_weight[: mlp.intermediate_size]
        mlp.up_proj.weight.data = mlp.gate_up_weight[mlp.intermediate_size :]
