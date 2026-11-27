# SPDX-License-Identifier: Apache-2.0
"""LFM3 depth decoder: one backbone hidden state to the logits of a speech frame's 12 codebooks.

Each codebook gets a slot of the projected hidden state plus its known code embedding or a learned mask embedding, and a bidirectional LFM3 trunk scores the masked codebooks. A frame is revealed over the passes of the MaskGIT schedule in a random codebook order (frame_sampling.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, kw_only=True)
class DepthTrunkConfig:
    """The depth trunk fields of the checkpoint's depth config section."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    norm_eps: float
    num_hidden_layers: int
    rope_theta: float

    @classmethod
    def from_checkpoint(cls, depth_config: Mapping[str, object]) -> DepthTrunkConfig:
        """Parse the config section, rejecting a trunk this module does not implement."""
        layer_types = depth_config["layer_types"]
        rope_parameters = depth_config["rope_parameters"]
        assert isinstance(layer_types, list) and isinstance(rope_parameters, Mapping)
        full_attention_rope = rope_parameters["full_attention"]
        assert isinstance(full_attention_rope, Mapping)
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError("The LFM3 depth trunk has attention layers only")
        elif depth_config.get("use_sandwich_norm"):
            raise ValueError("The LFM3 depth trunk has no sandwich norm")
        elif float(full_attention_rope.get("partial_rotary_factor", 1.0)) != 1.0:
            raise ValueError("The LFM3 depth trunk uses full RoPE")
        else:
            pass
        hidden_size = int(depth_config["hidden_size"])
        num_attention_heads = int(depth_config["num_attention_heads"])
        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(depth_config["intermediate_size"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(depth_config["num_key_value_heads"]),
            head_dim=int(
                depth_config.get("head_dim") or hidden_size // num_attention_heads
            ),
            norm_eps=float(depth_config["norm_eps"]),
            num_hidden_layers=len(layer_types),
            rope_theta=float(full_attention_rope["rope_theta"]),
        )


DEPTH_PREFIXES = (
    "depth_linear.",
    "mask_embedding",
    "depthformer.model.",
    "depth_embeddings.",
)


class DepthRMSNorm(nn.Module):
    """The trunk's norm: normalized in fp32, rounded to the input dtype, then scaled by the weight."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(x.dtype)


class HeadRMSNorm(DepthRMSNorm):
    """The codebook heads' norm: scaled by the weight in fp32 and rounded once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class DepthAttention(nn.Module):
    """Bidirectional GQA with Q/K norm, full RoPE and a per-head sigmoid gate."""

    def __init__(self, config: DepthTrunkConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.q_norm = DepthRMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = DepthRMSNorm(self.head_dim, config.norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_slots, _ = hidden_states.shape
        head_shape = (batch_size, num_slots, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        # note (ruilin): RoPE runs in fp32 on tables rounded to the activation dtype, as the checkpoint's code does.
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        q = ((q.float() * cos) + (rotate_half(q.float()) * sin)).to(hidden_states.dtype)
        k = ((k.float() * cos) + (rotate_half(k.float()) * sin)).to(hidden_states.dtype)
        group_size = self.num_heads // self.num_key_value_heads
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        attention_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attention_weights = F.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attention_output = torch.matmul(attention_weights, v).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        return self.o_proj((attention_output * gate).reshape(batch_size, num_slots, -1))


class DepthMLP(nn.Module):
    def __init__(self, config: DepthTrunkConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class DepthLayer(nn.Module):
    def __init__(self, config: DepthTrunkConfig) -> None:
        super().__init__()
        self.operator_norm = DepthRMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = DepthRMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = DepthAttention(config)
        self.feed_forward = DepthMLP(config)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.operator_norm(hidden_states), cos, sin
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class DepthTrunk(nn.Module):
    def __init__(self, config: DepthTrunkConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DepthLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.embedding_norm = DepthRMSNorm(config.hidden_size, config.norm_eps)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope_tables(
        self, num_slots: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(num_slots, device=device, dtype=torch.float)
        angles = positions[:, None] * self.inv_freq.to(device)[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype)[None], angles.sin().to(dtype)[None]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope_tables(
            hidden_states.shape[1], hidden_states.device, hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.embedding_norm(hidden_states)


class CodebookHead(nn.Module):
    """Input embedding and tied output head of one codebook."""

    def __init__(self, codebook_size: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, hidden_size)
        self.embedding_norm = HeadRMSNorm(hidden_size, eps)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embedding_norm(hidden_states), self.embedding.weight)


class Lfm3DepthDecoder(nn.Module):
    """Scores the masked codebooks of one frame from the backbone hidden state."""

    def __init__(
        self,
        depth_config: Mapping[str, object],
        text_hidden_size: int,
        num_codebooks: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        trunk_config = DepthTrunkConfig.from_checkpoint(depth_config)
        self.num_codebooks = num_codebooks
        self.hidden_size = trunk_config.hidden_size
        self.depth_linear = nn.Linear(
            text_hidden_size, num_codebooks * self.hidden_size
        )
        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size))
        self.model = DepthTrunk(trunk_config)
        self.depth_embeddings = nn.ModuleList(
            CodebookHead(codebook_size, self.hidden_size, trunk_config.norm_eps)
            for _ in range(num_codebooks)
        )

    def forward(
        self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        """hidden_states (N, H), codes and masked (N, 12) -> logits (N, 12, codebook_size)."""
        depth_inputs = self.depth_linear(hidden_states).unflatten(
            -1, (self.num_codebooks, self.hidden_size)
        )
        code_embeds = torch.stack(
            [
                head.embedding(codes[..., index])
                for index, head in enumerate(self.depth_embeddings)
            ],
            dim=-2,
        )
        depth_inputs = depth_inputs + torch.where(
            masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds
        )
        depth_outputs = self.model(depth_inputs)
        return torch.stack(
            [
                head.logits(depth_outputs[:, index])
                for index, head in enumerate(self.depth_embeddings)
            ],
            dim=1,
        )

    def load_weight(
        self, name: str, tensor: torch.Tensor, parameters: Mapping[str, nn.Parameter]
    ) -> bool:
        """Load one checkpoint weight (export names) into parameters (this module's named_parameters); False when it is not a depth weight."""
        if name.endswith(".to_logits.weight"):
            # note (ruilin): LFM3-Audio also stores each tied head as to_logits; its own code ignores the copy.
            return True
        else:
            pass
        if name.startswith("depthformer.model."):
            target = "model." + name[len("depthformer.model.") :]
        elif name.startswith(DEPTH_PREFIXES):
            target = name
        else:
            return False
        parameter = parameters[target]
        with torch.no_grad():
            parameter.copy_(tensor.to(parameter.dtype))
        return True
