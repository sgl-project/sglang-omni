# SPDX-License-Identifier: Apache-2.0
"""LFM3 depth decoder and MaskGIT frame sampler.

The depth decoder turns one backbone hidden state into the 12 codes of a speech frame: each codebook gets a slot of the projected hidden state plus its known code embedding or a learned mask embedding, and a bidirectional LFM3 trunk scores the masked codebooks. A frame is revealed over the passes of the MaskGIT schedule in a random codebook order.

The sampler makes the checkpoint's random draws in the checkpoint's order (codebook permutation, then one draw per revealed codebook, row by row) on a per-request generator, so a seeded request reproduces the checkpoint's own generate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation.logits_process import (
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

from sglang_omni.models.moss_tts.sampling_kernels import sample_seeded_fused


class SpeechSampler(str, Enum):
    """How speech frames are drawn.

    exact: each request's generator draws in the checkpoint's order, so a seed reproduces the checkpoint's own generate; only the depth passes are graphed.
    seeded: hashed draws from the request seed and frame position, so one graph holds the whole frame; the same seed gives the same frames at any batch size, but not the checkpoint's.
    """

    EXACT = "exact"
    SEEDED = "seeded"


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

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        """Load one checkpoint weight (export names); False when it is not a depth weight."""
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
        parameter = dict(self.named_parameters())[target]
        with torch.no_grad():
            parameter.copy_(tensor.to(parameter.dtype))
        return True


@dataclass(kw_only=True)
class FrameSampling:
    """A request's speech sampling settings, generator and repetition history."""

    temperature: float
    top_k: int | None
    repetition_penalty: float
    repetition_window: int
    generator: torch.Generator
    # Row seed of the seeded sampler; the exact sampler draws from the generator instead.
    seed: int = 0
    # Frames sampled so far: the seeded sampler's draw positions.
    frames_sampled: int = 0
    # The last repetition_window frames, (frames, num_codebooks).
    history: torch.Tensor | None = None

    def warpers(self) -> list[LogitsProcessor]:
        """The logits warpers of these settings; none for greedy decoding."""
        if self.temperature == 0:
            return []
        else:
            pass
        warpers: list[LogitsProcessor] = [
            TemperatureLogitsWarper(float(self.temperature))
        ]
        if self.top_k is not None and self.top_k > 0:
            warpers.append(TopKLogitsWarper(int(self.top_k)))
        else:
            pass
        return warpers

    def append(self, frame: torch.Tensor) -> None:
        if self.history is None:
            self.history = frame[None]
        else:
            self.history = torch.cat([self.history, frame[None]])[
                -self.repetition_window :
            ]


class DepthLogits(Protocol):
    def __call__(
        self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor: ...


@torch.no_grad()
def sample_frames(
    depth_logits: DepthLogits,
    hidden_states: torch.Tensor,
    samplings: Sequence[FrameSampling],
    schedule: Sequence[int],
) -> torch.Tensor:
    """Sample one frame per row of hidden_states (N, H) -> codes (N, 12), drawing exactly as the checkpoint's generate.

    Each depth pass runs batched. A row's penalty and warpers run once over the codebooks it reveals in the pass, and its draws go codebook by codebook in the checkpoint's order on its own generator, so a seeded request reproduces the checkpoint's own generate.
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    num_codebooks = sum(schedule)
    codes = torch.zeros((batch_size, num_codebooks), dtype=torch.long, device=device)
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    reveal_order = torch.stack(
        [
            torch.randperm(num_codebooks, device=device, generator=sampling.generator)
            for sampling in samplings
        ]
    )
    warpers = [sampling.warpers() for sampling in samplings]
    for revealed_now in reveal_order.split(list(schedule), dim=1):
        logits = depth_logits(hidden_states, codes, ~revealed)
        for row, sampling in enumerate(samplings):
            codebooks = revealed_now[row]
            scores = logits[row].index_select(0, codebooks).float()
            history = sampling.history
            if sampling.repetition_penalty != 1.0 and history is not None:
                scores = RepetitionPenaltyLogitsProcessor(
                    float(sampling.repetition_penalty)
                )(history.index_select(1, codebooks).T, scores)
            else:
                pass
            if not warpers[row]:
                codes[row, codebooks] = scores.argmax(-1)
                continue
            else:
                pass
            for warper in warpers[row]:
                scores = warper(None, scores)
            probabilities = scores.softmax(-1)
            codes[row, codebooks] = torch.cat(
                [
                    torch.multinomial(
                        probabilities[index : index + 1],
                        1,
                        generator=sampling.generator,
                    )[0]
                    for index in range(probabilities.shape[0])
                ]
            )
        revealed.scatter_(1, revealed_now, True)
    return codes


def seeded_uniform_keys(
    seeds: torch.Tensor, positions: torch.Tensor, width: int
) -> torch.Tensor:
    """Per-row hashed integers in [0, 2**24) of (seed, position, column): (N,) x (N,) -> (N, width)."""
    columns = torch.arange(width, device=seeds.device, dtype=torch.int64)
    mixed = (
        (seeds[:, None] * 19349663)
        ^ (positions[:, None] * 73856093)
        ^ (columns * 83492791)
    ) & 0xFFFFFFFF
    for _ in range(2):
        mixed = ((mixed ^ (mixed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return (mixed ^ (mixed >> 16)) & 0xFFFFFF


@torch.no_grad()
def seeded_frame_codes(
    depth_logits: DepthLogits,
    hidden_states: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    repetition_penalty: torch.Tensor,
    history: torch.Tensor,
    schedule: Sequence[int],
) -> torch.Tensor:
    """One frame per row with hashed draws and no host sync, so one CUDA graph can hold the whole frame.

    A row's draws depend only on its seed and frame position: the same seed gives the same frames at any batch size. The draws differ from the checkpoint's generator stream.
    hidden_states (N, H); seeds, frame_positions, top_k (N,) int64; temperature, repetition_penalty (N,) float; history (N, window, 12) int64; a row without history passes penalty 1, which leaves its scores unchanged.
    """
    batch_size = hidden_states.shape[0]
    num_codebooks = sum(schedule)
    draws_per_frame = num_codebooks + 1
    position_base = frame_positions * draws_per_frame
    reveal_order = seeded_uniform_keys(
        seeds, position_base + num_codebooks, num_codebooks
    ).argsort(dim=1, stable=True)
    codes = torch.zeros(
        (batch_size, num_codebooks), dtype=torch.long, device=hidden_states.device
    )
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    draw_positions = (
        position_base[:, None]
        + torch.arange(num_codebooks, device=hidden_states.device)[None, :]
    ).reshape(-1)
    row_seeds = seeds.repeat_interleave(num_codebooks)
    row_temperature = temperature.repeat_interleave(num_codebooks)
    row_top_k = top_k.repeat_interleave(num_codebooks)
    row_top_p = torch.ones_like(row_temperature)
    penalty = repetition_penalty[:, None, None]
    pass_start = 0
    for count in schedule:
        revealed_now = reveal_order[:, pass_start : pass_start + count]
        pass_start += count
        logits = depth_logits(hidden_states, codes, ~revealed).float()
        vocab = logits.shape[-1]
        repeated = torch.zeros(
            (batch_size, num_codebooks, vocab), dtype=torch.bool, device=logits.device
        ).scatter_(2, history.transpose(1, 2), True)
        penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
        scores = torch.where(repeated, penalized, logits)
        drawn = sample_seeded_fused(
            scores.reshape(-1, vocab),
            temperature=row_temperature,
            top_p=row_top_p,
            top_k=row_top_k,
            seeds=row_seeds,
            positions=draw_positions,
        ).view(batch_size, num_codebooks)
        reveal_mask = torch.zeros_like(revealed).scatter_(1, revealed_now, True)
        codes = torch.where(reveal_mask, drawn, codes)
        revealed = revealed | reveal_mask
    return codes
