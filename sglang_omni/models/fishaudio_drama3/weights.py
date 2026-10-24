# SPDX-License-Identifier: Apache-2.0
"""Checkpoint key routing and AudioDecoder weight loading for Drama 3.

Backbone keys keep their Hugging Face Qwen3.5 MoE names; decoder keys are
renamed onto AudioDecoder parameters here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.fishaudio_drama3.audio_decoder import (
    AudioDecoder,
    DecoderSelfAttention,
)

CheckpointPart = Literal["backbone", "audio_decoder"]
ShardName = Literal["whole", "gate", "up"]

BACKBONE_PREFIXES = ("model.language_model.", "lm_head.")
AUDIO_DECODER_PREFIXES = (
    "audio_decoder.",
    "audio_decoder_mtp_shared.",
    "audio_decoder_mtp.",
)
MODULE_PREFIXES = (
    ("audio_decoder_mtp_shared.layers.", "shared_layers."),
    ("audio_decoder_mtp.layers.", "residual_layers."),
    ("audio_decoder_mtp.heads.", "codebook_heads."),
    ("audio_decoder.codebook_embeddings.", "codebook_embeddings."),
)
LAYER_RENAMES: tuple[tuple[str, str, ShardName], ...] = (
    ("semantic_embedding.", "token_embedding.", "whole"),
    ("codebook_embedding.", "token_embedding.", "whole"),
    ("transformer_layer.attention.wqkv.", "self_attn.qkv_proj.", "whole"),
    ("transformer_layer.attention.wo.", "self_attn.o_proj.", "whole"),
    ("transformer_layer.attention.q_norm.", "self_attn.q_norm.", "whole"),
    ("transformer_layer.attention.k_norm.", "self_attn.k_norm.", "whole"),
    ("transformer_layer.attention_norm.", "input_layernorm.", "whole"),
    ("transformer_layer.ffn_norm.", "post_attention_layernorm.", "whole"),
    ("transformer_layer.feed_forward.w1.", "mlp.gate_up_proj.", "gate"),
    ("transformer_layer.feed_forward.w3.", "mlp.gate_up_proj.", "up"),
    ("transformer_layer.feed_forward.w2.", "mlp.down_proj.", "whole"),
)
QKV_SUFFIX = ".qkv_proj.weight"
GATE_UP_SUFFIX = "mlp.gate_up_proj.weight"


@dataclass(frozen=True, kw_only=True)
class ParameterShard:
    """One AudioDecoder parameter, or the gate / up half of a gate_up_proj weight."""

    parameter_name: str
    shard_name: ShardName


def checkpoint_part(checkpoint_name: str) -> CheckpointPart:
    if checkpoint_name.startswith(BACKBONE_PREFIXES):
        return "backbone"
    elif checkpoint_name.startswith(AUDIO_DECODER_PREFIXES):
        return "audio_decoder"
    else:
        raise ValueError(
            f"checkpoint key {checkpoint_name!r} is outside the Drama 3 namespaces "
            f"{BACKBONE_PREFIXES + AUDIO_DECODER_PREFIXES}"
        )


def parameter_shard(checkpoint_name: str) -> ParameterShard:
    """The AudioDecoder parameter shard an audio_decoder* checkpoint tensor fills."""
    for checkpoint_prefix, module_prefix in MODULE_PREFIXES:
        if checkpoint_name.startswith(checkpoint_prefix):
            parameter_name = module_prefix + checkpoint_name[len(checkpoint_prefix) :]
            break
        else:
            pass
    else:
        raise ValueError(f"{checkpoint_name!r} is not an audio decoder weight")
    for checkpoint_fragment, module_fragment, shard_name in LAYER_RENAMES:
        if checkpoint_fragment in parameter_name:
            return ParameterShard(
                parameter_name=parameter_name.replace(
                    checkpoint_fragment, module_fragment
                ),
                shard_name=shard_name,
            )
        else:
            pass
    return ParameterShard(parameter_name=parameter_name, shard_name="whole")


def expected_parameter_shards(decoder: AudioDecoder) -> set[ParameterShard]:
    shards: set[ParameterShard] = set()
    for parameter_name, _ in decoder.named_parameters():
        if parameter_name.endswith(GATE_UP_SUFFIX):
            shards.add(ParameterShard(parameter_name=parameter_name, shard_name="gate"))
            shards.add(ParameterShard(parameter_name=parameter_name, shard_name="up"))
        else:
            shards.add(
                ParameterShard(parameter_name=parameter_name, shard_name="whole")
            )
    return shards


def interleave_query_and_gate(
    qkv_weight: torch.Tensor, head_count: int, head_dim: int
) -> torch.Tensor:
    """[query | gate | key | value] rows -> per-head [query, gate], then key, value."""
    query_width = head_count * head_dim
    query, gate, key_and_value = qkv_weight.split(
        [query_width, query_width, qkv_weight.shape[0] - 2 * query_width], dim=0
    )
    per_head = torch.stack(
        (
            query.reshape(head_count, head_dim, -1),
            gate.reshape(head_count, head_dim, -1),
        ),
        dim=1,
    )
    return torch.cat((per_head.reshape(2 * query_width, -1), key_and_value), dim=0)


def load_audio_decoder_weights(
    decoder: AudioDecoder, weights: Iterable[tuple[str, torch.Tensor]]
) -> None:
    """Copy audio_decoder* tensors into decoder; each shard must arrive once."""
    parameters = dict(decoder.named_parameters())
    expected = expected_parameter_shards(decoder)
    loaded: set[ParameterShard] = set()
    for checkpoint_name, tensor in weights:
        shard = parameter_shard(checkpoint_name)
        if shard not in expected:
            raise ValueError(
                f"{checkpoint_name!r} maps to {shard.parameter_name}"
                f"[{shard.shard_name}], which is not a decoder parameter shard"
            )
        elif shard in loaded:
            raise ValueError(f"{checkpoint_name!r} fills {shard} a second time")
        else:
            pass
        parameter = parameters[shard.parameter_name]
        half_rows = parameter.shape[0] // 2
        if shard.shard_name == "gate":
            target = parameter[:half_rows]
        elif shard.shard_name == "up":
            target = parameter[half_rows:]
        else:
            target = parameter
        if tensor.shape != target.shape:
            raise ValueError(
                f"{checkpoint_name!r} has shape {tuple(tensor.shape)}; {shard} needs "
                f"{tuple(target.shape)}"
            )
        else:
            pass
        if shard.parameter_name.endswith(QKV_SUFFIX):
            attention = decoder.get_submodule(
                shard.parameter_name.removesuffix(QKV_SUFFIX)
            )
            assert isinstance(attention, DecoderSelfAttention)
            value = interleave_query_and_gate(
                tensor, attention.head_count, attention.head_dim
            )
        else:
            value = tensor
        with torch.no_grad():
            target.copy_(value)
        loaded.add(shard)
    missing = expected - loaded
    if missing:
        missing_names = sorted(
            f"{shard.parameter_name}[{shard.shard_name}]" for shard in missing
        )
        raise ValueError(
            f"the checkpoint left {len(missing)} decoder parameter shards unfilled, "
            f"e.g. {missing_names[:8]}"
        )
    else:
        pass
