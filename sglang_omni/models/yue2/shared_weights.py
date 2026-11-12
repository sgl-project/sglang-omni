# SPDX-License-Identifier: Apache-2.0
"""AR weights the acoustic stage reads from the AR stage's GPU memory over CUDA IPC."""

import torch
from torch import nn

from sglang_omni.models.yue2.layers import Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.utils import ipc_weights

ACOUSTIC_ONLY_PREFIXES = ("llm2vae.", "vae2llm.", "time_embedder.", "latent_pos_embed.")
ATTACH_TIMEOUT_SECONDS = 60.0


def is_acoustic_only(name: str) -> bool:
    return ".nar_" in name or name.startswith(ACOUSTIC_ONLY_PREFIXES)


class SharedBackbone(nn.Module):
    """The AR tensors both stages read, under the names both sides agree on."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name, tensor in tensors.items():
            self.register_parameter(name, nn.Parameter(tensor, requires_grad=False))


def engine_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """The SGLang Qwen3 tensors, which keep q, k, v and gate, up stacked."""
    backbone = model.model
    tensors = {
        "embed_tokens": backbone.embed_tokens.weight,
        "norm": backbone.norm.weight,
    }
    for index, layer in enumerate(backbone.layers):
        tensors[f"layer{index}_input_layernorm"] = layer.input_layernorm.weight
        tensors[f"layer{index}_qkv"] = layer.self_attn.qkv_proj.weight
        tensors[f"layer{index}_q_norm"] = layer.self_attn.q_norm.weight
        tensors[f"layer{index}_k_norm"] = layer.self_attn.k_norm.weight
        tensors[f"layer{index}_o_proj"] = layer.self_attn.o_proj.weight
        tensors[f"layer{index}_post_attention_layernorm"] = (
            layer.post_attention_layernorm.weight
        )
        tensors[f"layer{index}_gate_up"] = layer.mlp.gate_up_proj.weight
        tensors[f"layer{index}_down_proj"] = layer.mlp.down_proj.weight
    return tensors


def placeholder_tensors(
    config: Yue2ModelConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    key_value = config.num_key_value_heads * config.head_dim
    shapes = {"embed_tokens": (config.vocab_size, hidden), "norm": (hidden,)}
    for index in range(config.num_hidden_layers):
        shapes[f"layer{index}_input_layernorm"] = (hidden,)
        shapes[f"layer{index}_qkv"] = (query + 2 * key_value, hidden)
        shapes[f"layer{index}_q_norm"] = (config.head_dim,)
        shapes[f"layer{index}_k_norm"] = (config.head_dim,)
        shapes[f"layer{index}_o_proj"] = (hidden, query)
        shapes[f"layer{index}_post_attention_layernorm"] = (hidden,)
        shapes[f"layer{index}_gate_up"] = (2 * config.intermediate_size, hidden)
        shapes[f"layer{index}_down_proj"] = (hidden, config.intermediate_size)
    # note (ruiling): attaching replaces each placeholder, so one element stands in for every shape.
    element = torch.empty((), dtype=torch.bfloat16, device=device)
    return {name: element.expand(shape) for name, shape in shapes.items()}


def drop_ar_weights(model: Yue2Model) -> None:
    """Unregister the AR tensors so loading and device moves skip them."""
    for name, _ in list(model.named_parameters()):
        if is_acoustic_only(name):
            pass
        else:
            module_path, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(module_path), leaf, None)


def export_ar_weights(model: nn.Module, directory: str, checkpoint: str) -> None:
    ipc_weights.leader_export(
        SharedBackbone(engine_tensors(model)), directory, model_path=checkpoint
    )


def bind_ar_weights(
    model: Yue2Model, tensors: dict[str, torch.Tensor], *, fused: bool
) -> None:
    """Point the acoustic model's AR tensors at the given storage."""

    def parameter(name: str) -> nn.Parameter:
        return nn.Parameter(tensors[name], requires_grad=False)

    backbone = model.model
    backbone.embed_tokens.weight = parameter("embed_tokens")
    backbone.norm.weight = parameter("norm")
    for index, layer in enumerate(backbone.layers):
        prefix = f"layer{index}_"
        layer.input_layernorm.weight = parameter(f"{prefix}input_layernorm")
        layer.self_attn.set_projections(tensors[f"{prefix}qkv"], fused=fused)
        layer.self_attn.q_norm.weight = parameter(f"{prefix}q_norm")
        layer.self_attn.k_norm.weight = parameter(f"{prefix}k_norm")
        layer.self_attn.o_proj.weight = parameter(f"{prefix}o_proj")
        layer.post_attention_layernorm.weight = parameter(
            f"{prefix}post_attention_layernorm"
        )
        layer.mlp.set_projections(tensors[f"{prefix}gate_up"], fused=fused)
        layer.mlp.down_proj.weight = parameter(f"{prefix}down_proj")


def attach_ar_weights(
    model: Yue2Model, directory: str, checkpoint: str, *, fused: bool
) -> ipc_weights.LeaderLivenessMonitor:
    """Read the AR tensors from the AR stage's storage instead of loading them."""
    device = model.llm2vae.weight.device
    shared = SharedBackbone(placeholder_tensors(model.config, device))
    _, monitor = ipc_weights.follower_attach(
        shared, directory, timeout_s=ATTACH_TIMEOUT_SECONDS, model_path=checkpoint
    )
    bind_ar_weights(
        model,
        {name: tensor.data for name, tensor in shared.named_parameters()},
        fused=fused,
    )
    return monitor
