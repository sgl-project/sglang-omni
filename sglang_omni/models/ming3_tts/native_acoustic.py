# SPDX-License-Identifier: Apache-2.0
"""Own native-precision dispatch and buffers used by Tiny acoustic graphs."""

from __future__ import annotations

import torch

from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss
from sglang_omni.modules.ming_audio.flow.modules import Attention, RMSNorm
from sglang_omni.modules.ming_audio.flow.normalization import (
    HAS_TRITON,
    NormalizationDispatch,
    normalization_affine,
    normalize_native_rms,
    square_float32,
)
from sglang_omni.modules.ming_audio.flow.rotary import (
    SharedRotaryTransform,
    apply_shared_rotary,
)


class NativeAcousticCapture:
    """Retain graph inputs after restoring each module's eager implementation."""

    def __init__(self, flowloss: FlowLoss, aggregator: Aggregator) -> None:
        self.flowloss: FlowLoss = flowloss
        self.attention_layers: list[Attention] = [
            layer
            for head in (flowloss, aggregator)
            for layer in head.modules()
            if isinstance(layer, Attention)
        ]
        self.normalization_layers: list[RMSNorm] = [
            layer
            for head in (flowloss, aggregator)
            for layer in head.modules()
            if isinstance(layer, RMSNorm)
        ]
        self.normalization_dispatch: NormalizationDispatch = NormalizationDispatch(
            square_float32=torch.compile(square_float32, fullgraph=True, dynamic=True),
            affine=torch.compile(
                normalization_affine,
                fullgraph=True,
                dynamic=True,
                options={"emulate_precision_casts": True},
            ),
            normalize_contiguous=(
                normalize_native_rms
                if HAS_TRITON
                and torch.__version__.split("+")[0] == "2.14.1"
                and torch.version.hip is None
                else None
            ),
        )
        self.rotary_dispatch: SharedRotaryTransform = torch.compile(
            apply_shared_rotary,
            fullgraph=True,
            dynamic=True,
            options={"emulate_precision_casts": True},
        )
        self.projection_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=next(flowloss.parameters()).device
        )
        self.packed_projections: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.timestep_embeddings: dict[int, tuple[torch.Tensor, ...]] = {}
        self.active: bool = False

    def install(self) -> None:
        self.active = True
        for layer in self.attention_layers:
            layer.projection_stream = self.projection_stream
            layer.rotary_dispatch = self.rotary_dispatch
            if layer.to_qkv is None:
                packed_weight = torch.cat(
                    [layer.to_q.weight, layer.to_k.weight, layer.to_v.weight]
                )
                if layer.to_q.bias is None:
                    packed_bias = None
                else:
                    packed_bias = torch.cat(
                        [layer.to_q.bias, layer.to_k.bias, layer.to_v.bias]
                    )
                layer.packed_projection = (packed_weight, packed_bias)
                self.packed_projections.append(layer.packed_projection)
            else:
                pass
        for layer in self.normalization_layers:
            layer.normalization_dispatch = self.normalization_dispatch

    def prepare_batch(self, timesteps: torch.Tensor, batch_size: int) -> None:
        assert self.active
        with torch.autocast(device_type=timesteps.device.type, dtype=torch.bfloat16):
            embeddings = self.flowloss.cfm.build_timestep_embeddings(
                timesteps, batch_size
            )
        self.timestep_embeddings[batch_size] = embeddings
        self.flowloss.cfm.timestep_embeddings = embeddings

    def restore(self) -> None:
        self.active = False
        self.flowloss.cfm.timestep_embeddings = None
        for layer in self.attention_layers:
            layer.projection_stream = None
            layer.packed_projection = None
            layer.rotary_dispatch = None
        for layer in self.normalization_layers:
            layer.normalization_dispatch = None
