# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 RVQ depth decoder (c1..c7)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, bias=False
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_layernorm(x)
        bsz, seq, hidden = h.shape
        qkv = F.linear(h, self.self_attn.in_proj_weight)
        q, k, v = qkv.chunk(3, dim=-1)
        heads = self.self_attn.num_heads
        head_dim = hidden // heads
        q = q.reshape(bsz, seq, heads, head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq, heads, head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq, heads, head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).contiguous().reshape(bsz, seq, hidden)
        x = x + self.self_attn.out_proj(h)
        h = self.post_attention_layernorm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h


class RVQDepthDecoder(nn.Module):
    """Depth autoregressive decoder for the seven residual codebooks."""

    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.audio_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, audio_vocab_size, bias=False)
                for _ in range(num_codebooks - 1)
            ]
        )

    def forward(self, seq: Tensor) -> Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = seq + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def load_checkpoint_state(self, audio_state: dict[str, Tensor]) -> None:
        """Load the checkpoint's split q/k/v weights with fail-fast checks."""

        prefix = "model.audio_decoder."
        loaded: set[str] = set()
        missing: list[str] = []
        mismatches: list[str] = []

        def fetch(key: str) -> Tensor | None:
            source = audio_state.get(key)
            if source is None:
                missing.append(key)
            return source

        def assign(
            label: str, target: Tensor, source: Tensor, keys: tuple[str, ...]
        ) -> None:
            if tuple(source.shape) != tuple(target.shape):
                mismatches.append(
                    f"{label}: checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
                )
                return
            target.data.copy_(source.to(dtype=target.dtype))
            loaded.update(keys)

        def copy_tensor(key: str, target: Tensor) -> None:
            source = fetch(key)
            if source is not None:
                assign(key, target, source, (key,))

        copy_tensor(prefix + "projection.weight", self.projection.weight)
        copy_tensor(prefix + "pos_embedding.weight", self.pos_embedding.weight)
        copy_tensor(prefix + "norm.weight", self.norm.weight)
        for idx, head in enumerate(self.audio_heads):
            copy_tensor(prefix + f"audio_heads.{idx}.weight", head.weight)

        for idx, layer in enumerate(self.layers):
            lp = f"{prefix}layers.{idx}."
            copy_tensor(lp + "input_layernorm.weight", layer.input_layernorm.weight)
            copy_tensor(
                lp + "post_attention_layernorm.weight",
                layer.post_attention_layernorm.weight,
            )
            qkv_keys = tuple(
                lp + f"self_attn.{name}_proj.weight" for name in ("q", "k", "v")
            )
            parts = [fetch(key) for key in qkv_keys]
            if all(part is not None for part in parts):
                assign(
                    f"{lp}self_attn.qkv",
                    layer.self_attn.in_proj_weight,
                    torch.cat(parts, dim=0),
                    qkv_keys,
                )
            copy_tensor(lp + "self_attn.o_proj.weight", layer.self_attn.out_proj.weight)
            copy_tensor(lp + "mlp.gate_proj.weight", layer.gate_proj.weight)
            copy_tensor(lp + "mlp.up_proj.weight", layer.up_proj.weight)
            copy_tensor(lp + "mlp.down_proj.weight", layer.down_proj.weight)

        expected = {key for key in audio_state if key.startswith(prefix)}
        unexpected = sorted(expected - loaded)
        if missing or mismatches or unexpected:
            raise ValueError(
                "MiniMax Music 3 RVQ decoder strict load failed: "
                f"missing={missing[:8]}, mismatches={mismatches[:8]}, unexpected={unexpected[:8]}"
            )


_TOP_K = 50


def sample_topk(
    logits: Tensor,
    *,
    generator: torch.Generator,
) -> Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, _TOP_K, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def sample_topk_seeded(
    logits: Tensor,
    seeds: Tensor,
    positions: Tensor,
) -> Tensor:
    """Top-k sample every row in one kernel, each from its own RNG stream.

    Args:
        logits: [rows, vocab].
        seeds: [rows] per-request sampling seeds.
        positions: [rows] draw identifiers, unique per (frame, codebook).

    Returns:
        [rows] sampled column indices.
    """
    from sglang.srt.layers.sampler import multinomial_with_seed

    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, _TOP_K, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    return multinomial_with_seed(values, seeds, positions).squeeze(-1)


__all__ = ["DecoderBlock", "RVQDepthDecoder", "sample_topk", "sample_topk_seeded"]
