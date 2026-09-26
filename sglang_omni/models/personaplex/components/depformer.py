# SPDX-License-Identifier: Apache-2.0
"""The depth transformer: one frame's 8 agent codebooks, one step each.

Every step has its own projection, gating and output head (weights_per_step
in the reference); the norms are shared. Attention runs over the steps of the
same frame only, so each layer's K/V cache is one buffer sized to the frame's
steps, written in place and re-made every frame.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    DEPFORMER,
    TEXT_CARD,
    DepformerSpec,
)


def rms_norm_f32(x: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """x * alpha / sqrt(eps + mean(x²)) computed in float32, as the checkpoint
    was trained (rms_norm_f32: eps inside the root, not the usual outside)."""
    x_f32 = x.float()
    variance = eps + x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * (alpha.float() * torch.rsqrt(variance))).to(x.dtype)


class DepformerLayer(nn.Module):
    def __init__(self, spec: DepformerSpec) -> None:
        super().__init__()
        self.spec = spec
        dim, steps, ffn = spec.dim, spec.steps, spec.ffn_hidden
        self.in_proj_weight = nn.Parameter(torch.empty(steps, 3 * dim, dim))
        self.out_proj_weight = nn.Parameter(torch.empty(steps, dim, dim))
        self.norm1_alpha = nn.Parameter(torch.ones(dim))
        self.norm2_alpha = nn.Parameter(torch.ones(dim))
        self.gate_in_weight = nn.Parameter(torch.empty(steps, 2 * ffn, dim))
        self.gate_out_weight = nn.Parameter(torch.empty(steps, dim, ffn))

    def step(
        self, x_BD: torch.Tensor, step: int, cache_2BHSD: torch.Tensor
    ) -> torch.Tensor:
        """One step; cache_2BHSD holds this frame's keys and values, slot per step."""
        spec = self.spec
        h = rms_norm_f32(x_BD, self.norm1_alpha, spec.rms_norm_eps)
        qkv = functional.linear(h, self.in_proj_weight[step])
        q, k, v = rearrange(qkv, "b (p h d) -> p b h d", p=3, h=spec.num_heads)
        cache_2BHSD[0, :, :, step] = k
        cache_2BHSD[1, :, :, step] = v
        attn = functional.scaled_dot_product_attention(
            q[:, :, None],
            cache_2BHSD[0, :, :, : step + 1],
            cache_2BHSD[1, :, :, : step + 1],
        )
        x_BD = x_BD + functional.linear(
            rearrange(attn, "b h 1 d -> b (h d)"), self.out_proj_weight[step]
        )

        h = rms_norm_f32(x_BD, self.norm2_alpha, spec.rms_norm_eps)
        gate = functional.linear(h, self.gate_in_weight[step])
        gate, up = gate.chunk(2, dim=-1)
        return x_BD + functional.linear(
            functional.silu(gate) * up, self.gate_out_weight[step]
        )


class Depformer(nn.Module):
    def __init__(self, spec: DepformerSpec = DEPFORMER) -> None:
        super().__init__()
        self.spec = spec
        self.depformer_in = nn.ModuleList(
            nn.Linear(spec.input_dim, spec.dim, bias=False) for _ in range(spec.steps)
        )
        self.depformer_text_emb = nn.Embedding(TEXT_CARD + 1, spec.dim)
        # Note (wilsonzheng0327): The last codebook is never an input.
        self.depformer_emb = nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, spec.dim) for _ in range(spec.steps - 1)
        )
        self.layers = nn.ModuleList(
            DepformerLayer(spec) for _ in range(spec.num_layers)
        )
        self.linears = nn.ModuleList(
            nn.Linear(spec.dim, AUDIO_CARD, bias=False) for _ in range(spec.steps)
        )

    def generate(
        self,
        text_token_B: torch.Tensor,
        transformer_out_BD: torch.Tensor,
        forced_BK: torch.Tensor,
        sample: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Spell out one frame of agent codes.

        Args:
            text_token_B: the frame's sampled text token.
            transformer_out_BD: the temporal transformer's normalised output.
            forced_BK: [B, steps] codes to keep instead of sampling, -1
                where the step is free. A forced code still conditions the
                steps after it, as teacher forcing does in the reference.
            sample: [B, card] float logits → [B] ids.
        """
        spec = self.spec
        caches = [
            transformer_out_BD.new_empty(
                2,
                transformer_out_BD.shape[0],
                spec.num_heads,
                spec.steps,
                spec.head_dim,
            )
            for _ in self.layers
        ]
        previous = text_token_B
        codes = []
        for step in range(spec.steps):
            token_emb = (
                self.depformer_text_emb(previous)
                if step == 0
                else self.depformer_emb[step - 1](previous)
            )
            x = self.depformer_in[step](transformer_out_BD) + token_emb
            for layer, cache in zip(self.layers, caches, strict=True):
                x = layer.step(x, step, cache)
            sampled = sample(self.linears[step](x).float())
            forced = forced_BK[:, step]
            previous = torch.where(forced >= 0, forced, sampled)
            codes.append(previous)
        return torch.stack(codes, dim=1)

    def load_reference_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load depformer* / linears.* tensors in the checkpoint's names.

        Per-step tensors are stacked along a leading step axis; a checkpoint
        with more steps than we run (16 vs 8) simply has its tail ignored.
        """
        spec = self.spec
        state: dict[str, torch.Tensor] = {}
        for step in range(spec.steps):
            state[f"depformer_in.{step}.weight"] = weights[
                f"depformer_in.{step}.weight"
            ]
            state[f"linears.{step}.weight"] = weights[f"linears.{step}.weight"]
        for step in range(spec.steps - 1):
            state[f"depformer_emb.{step}.weight"] = weights[
                f"depformer_emb.{step}.weight"
            ]
        state["depformer_text_emb.weight"] = weights["depformer_text_emb.weight"]
        for index in range(spec.num_layers):
            source = f"depformer.layers.{index}"
            target = f"layers.{index}"
            in_proj = weights[f"{source}.self_attn.in_proj_weight"]
            out_proj = weights[f"{source}.self_attn.out_proj.weight"]
            state[f"{target}.in_proj_weight"] = in_proj.view(
                -1, 3 * spec.dim, spec.dim
            )[: spec.steps]
            state[f"{target}.out_proj_weight"] = out_proj.view(-1, spec.dim, spec.dim)[
                : spec.steps
            ]
            state[f"{target}.norm1_alpha"] = weights[f"{source}.norm1.alpha"].reshape(
                -1
            )
            state[f"{target}.norm2_alpha"] = weights[f"{source}.norm2.alpha"].reshape(
                -1
            )
            state[f"{target}.gate_in_weight"] = torch.stack(
                [
                    weights[f"{source}.gating.{s}.linear_in.weight"]
                    for s in range(spec.steps)
                ]
            )
            state[f"{target}.gate_out_weight"] = torch.stack(
                [
                    weights[f"{source}.gating.{s}.linear_out.weight"]
                    for s in range(spec.steps)
                ]
            )
        self.load_state_dict(state, strict=True)


__all__ = ["Depformer", "DepformerLayer", "rms_norm_f32"]
