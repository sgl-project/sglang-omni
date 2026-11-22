# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio depth decoder and MaskGIT frame sampler.

The depth decoder turns one backbone hidden state into the 12 codes of a
speech frame: each codebook gets a slot of the projected hidden state plus its
known code embedding or a learned mask embedding, and a 3-layer bidirectional
LFM3 trunk scores the masked codebooks. A frame is revealed over the passes
of the MaskGIT schedule ([4, 4, 4]) in a random codebook order.

The sampler follows the checkpoint's ``sample_audio_frame`` call for call
(order permutation, then one draw per revealed codebook, row by row), on a
per-request generator, so a seeded request reproduces the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

DEPTH_PREFIXES = (
    "depth_linear.",
    "mask_embedding",
    "depthformer.model.",
    "depth_embeddings.",
)


class DepthRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class DepthAttention(nn.Module):
    """Bidirectional GQA with Q/K norm, full RoPE and a per-head sigmoid gate."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or hidden // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.q_norm = DepthRMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_norm = DepthRMSNorm(self.head_dim, cfg["norm_eps"])

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, slots, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, slots, -1, self.head_dim)).transpose(
            1, 2
        )
        k = self.k_norm(self.k_proj(x).view(batch, slots, -1, self.head_dim)).transpose(
            1, 2
        )
        v = self.v_proj(x).view(batch, slots, -1, self.head_dim).transpose(1, 2)
        # RoPE in float32 from tables rounded to the activation dtype, as the reference does.
        cos32 = cos.unsqueeze(1).float()
        sin32 = sin.unsqueeze(1).float()
        q = ((q.float() * cos32) + (rotate_half(q.float()) * sin32)).to(x.dtype)
        k = ((k.float() * cos32) + (rotate_half(k.float()) * sin32)).to(x.dtype)
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        out = (out * gate).reshape(batch, slots, -1)
        return self.o_proj(out)


class DepthMLP(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.w3 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.w2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DepthLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.operator_norm = DepthRMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.ffn_norm = DepthRMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.self_attn = DepthAttention(cfg)
        self.feed_forward = DepthMLP(cfg)
        if cfg.get("use_sandwich_norm"):
            raise ValueError("The LFM3-Audio depth decoder has no sandwich norm")
        else:
            pass

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.operator_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class DepthTrunk(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        if any(layer_type != "full_attention" for layer_type in cfg["layer_types"]):
            raise ValueError("The LFM3-Audio depth trunk has attention layers only")
        else:
            pass
        self.layers = nn.ModuleList(DepthLayer(cfg) for _ in cfg["layer_types"])
        self.embedding_norm = DepthRMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        rope = cfg["rope_parameters"]["full_attention"]
        if float(rope.get("partial_rotary_factor", 1.0)) != 1.0:
            raise ValueError("The LFM3-Audio depth trunk uses full RoPE")
        else:
            pass
        head_dim = (
            cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        )
        inv_freq = 1.0 / (
            float(rope["rope_theta"])
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope_tables(
        self, slots: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(slots, device=device, dtype=torch.float)
        freqs = positions[:, None] * self.inv_freq.to(device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope_tables(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.embedding_norm(x)


class CodebookHead(nn.Module):
    """Input embedding and tied output head of one codebook."""

    def __init__(self, codebook_size: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, hidden)
        self.embedding_norm = DepthRMSNorm(hidden, eps)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embedding_norm(x), self.embedding.weight)


class Lfm3DepthDecoder(nn.Module):
    """Scores the masked codebooks of one frame from the backbone hidden state."""

    def __init__(
        self,
        depth_config: dict,
        text_hidden_size: int,
        num_codebooks: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.hidden_size = depth_config["hidden_size"]
        self.depth_linear = nn.Linear(
            text_hidden_size, num_codebooks * self.hidden_size
        )
        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size))
        self.model = DepthTrunk(depth_config)
        self.depth_embeddings = nn.ModuleList(
            CodebookHead(codebook_size, self.hidden_size, depth_config["norm_eps"])
            for _ in range(num_codebooks)
        )

    def forward(
        self, hidden: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        """``hidden`` (N, H), ``codes`` and ``masked`` (N, 12) -> logits (N, 12, codebook_size)."""
        depth_in = self.depth_linear(hidden).unflatten(
            -1, (self.num_codebooks, self.hidden_size)
        )
        code_embeds = torch.stack(
            [
                head.embedding(codes[..., index])
                for index, head in enumerate(self.depth_embeddings)
            ],
            dim=-2,
        )
        depth_in = depth_in + torch.where(
            masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds
        )
        depth_out = self.model(depth_in)
        return torch.stack(
            [
                head.logits(depth_out[:, index])
                for index, head in enumerate(self.depth_embeddings)
            ],
            dim=1,
        )

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        """Load one checkpoint weight (export names); False when it is not a depth weight."""
        if name.endswith(".to_logits.weight"):
            # The export also stores each tied head as `to_logits`; the reference ignores the copy.
            return True
        else:
            pass
        if name.startswith("depthformer.model."):
            target = "model." + name[len("depthformer.model.") :]
        elif name.startswith(DEPTH_PREFIXES):
            target = name
        else:
            return False
        param = dict(self.named_parameters())[target]
        with torch.no_grad():
            param.copy_(tensor.to(param.dtype))
        return True


@dataclass
class FrameSampling:
    """Per-request speech sampling settings and state."""

    temperature: float | None = 1.0
    top_k: int | None = 128
    repetition_penalty: float | None = 1.1
    repetition_window: int = 4096
    generator: torch.Generator | None = None
    history: list[torch.Tensor] | None = None

    def warpers(self) -> list:
        if self.temperature == 0:
            return []
        else:
            pass
        warpers: list = []
        if self.temperature:
            warpers.append(TemperatureLogitsWarper(float(self.temperature)))
        else:
            pass
        if self.top_k and self.top_k > 0:
            warpers.append(TopKLogitsWarper(int(self.top_k)))
        else:
            pass
        return warpers

    def append(self, frame: torch.Tensor) -> None:
        if self.history is None:
            self.history = []
        else:
            pass
        self.history.append(frame)
        if self.repetition_window and len(self.history) > self.repetition_window:
            del self.history[: len(self.history) - self.repetition_window]
        else:
            pass


def sample_code(
    logits: torch.Tensor,
    sampling: FrameSampling,
    warpers: list,
    history: torch.Tensor | None,
) -> torch.Tensor:
    scores = logits.float()[None].clone()
    penalty = sampling.repetition_penalty
    if penalty is not None and penalty != 1.0 and history is not None:
        scores = RepetitionPenaltyLogitsProcessor(float(penalty))(
            history.reshape(1, -1).to(scores.device), scores
        )
    else:
        pass
    if not warpers:
        return scores.argmax(-1)
    else:
        pass
    for warper in warpers:
        scores = warper(None, scores)
    return torch.multinomial(scores.softmax(-1), 1, generator=sampling.generator)[0]


@torch.no_grad()
def sample_frames(
    decoder: Lfm3DepthDecoder,
    hidden: torch.Tensor,
    samplings: Sequence[FrameSampling],
    schedule: Sequence[int],
) -> torch.Tensor:
    """Sample one frame per row of ``hidden`` (N, H) -> codes (N, 12).

    Each pass of the depth decoder runs batched; the draws are per row and per
    codebook in the reference's order so a per-request generator stays in step.
    """
    batch = hidden.shape[0]
    num_codebooks = decoder.num_codebooks
    codes = torch.zeros((batch, num_codebooks), dtype=torch.long, device=hidden.device)
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    order = torch.stack(
        [
            torch.randperm(
                num_codebooks, device=hidden.device, generator=sampling.generator
            )
            for sampling in samplings
        ]
    )
    histories = [
        torch.stack(sampling.history) if sampling.history else None
        for sampling in samplings
    ]
    warpers = [sampling.warpers() for sampling in samplings]
    for pick in order.split(list(schedule), dim=1):
        logits = decoder(hidden, codes, ~revealed)
        for row in range(batch):
            for index in pick[row].tolist():
                history = None if histories[row] is None else histories[row][:, index]
                codes[row, index] = sample_code(
                    logits[row, index], samplings[row], warpers[row], history
                ).to(codes.device)
        revealed.scatter_(1, pick, True)
    return codes
