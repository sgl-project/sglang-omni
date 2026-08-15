# SPDX-License-Identifier: Apache-2.0
"""Inference-only MiniMax Music 3 flow-matching DIT.

The parameter names mirror flowmatching_vae.pth exactly so checkpoint
loading remains strict.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from types import MethodType

import torch
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, enable_cache
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import AR_HIDDEN_SIZE, DEFAULT_DIT_CFG_SCALE, DEFAULT_DIT_STEPS

logger = logging.getLogger(__name__)

_ATTENTION_BACKENDS: dict[str, AttentionBackendEnum | None] = {
    "auto": None,
    "torch_sdpa": AttentionBackendEnum.TORCH_SDPA,
    "fa": AttentionBackendEnum.FA,
    "sage_attn": AttentionBackendEnum.SAGE_ATTN,
}


def _resolve_attention_backend(value: str) -> AttentionBackendEnum | None:
    name = value.strip().lower()
    if name not in _ATTENTION_BACKENDS:
        raise ValueError(
            "MiniMax Music 3 attention_backend must be one of: "
            + ", ".join(sorted(_ATTENTION_BACKENDS))
        )
    return _ATTENTION_BACKENDS[name]


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features))

    def forward(self, value: Tensor) -> Tensor:
        f = 2.0 * math.pi * value @ self.weight.T
        return torch.cat((f.cos(), f.sin()), dim=-1)


class LayerNorm(nn.Module):
    """The x-transformers LayerNorm variant used by the checkpoint."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.gamma, self.beta)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward_from_seq_len(self, seq_len: int) -> tuple[Tensor, float]:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1), 1.0


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
    rot_dim = rope_cos.shape[-1]
    rotated = x[..., :rot_dim]
    rope_cos = rope_cos[-x.shape[-2] :].to(dtype=rotated.dtype, device=x.device)
    rope_sin = rope_sin[-x.shape[-2] :].to(dtype=rotated.dtype, device=x.device)
    rotated = rotated * rope_cos + _rotate_half(rotated) * rope_sin
    return torch.cat((rotated, x[..., rot_dim:]), dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_heads: int,
        compute_dtype: torch.dtype,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.num_heads = dim // dim_heads
        self.dim_heads = dim_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        resolved_backend = _resolve_attention_backend(attention_backend)
        selected_backend = resolved_backend or AttentionBackendEnum.FA
        supported_backends = None if resolved_backend is None else {resolved_backend}
        with component_attn_backend_context_manager(
            selected_backend, component_name="minimax_music3_dit"
        ):
            self.backend = LocalAttention(
                num_heads=self.num_heads,
                head_size=self.dim_heads,
                causal=False,
                supported_attention_backends=supported_backends,
                compute_dtype=compute_dtype,
            )
        if (
            resolved_backend is AttentionBackendEnum.SAGE_ATTN
            and self.backend.backend is not AttentionBackendEnum.SAGE_ATTN
        ):
            raise RuntimeError(
                "MiniMax Music 3 requested sage_attn, but SageAttention is unavailable; "
                "install the SageAttention backend or use attention_backend=torch_sdpa"
            )

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, seq, self.num_heads, self.dim_heads).transpose(1, 2)
        k = k.reshape(bsz, seq, self.num_heads, self.dim_heads).transpose(1, 2)
        v = v.reshape(bsz, seq, self.num_heads, self.dim_heads).transpose(1, 2)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        out = self.backend(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        out = out.contiguous().reshape(bsz, seq, dim)
        return self.to_out(out)


class GLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, inner_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * self.act(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            GLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_heads: int,
        inner_dim: int,
        compute_dtype: torch.dtype,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(
            dim,
            dim_heads=dim_heads,
            compute_dtype=compute_dtype,
            attention_backend=attention_backend,
        )
        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim, inner_dim)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.pre_norm(x), rope_cos, rope_sin)
        x = x + self.ff(self.ff_norm(x))
        return x


class ContinuousTransformer(nn.Module):
    def __init__(self, *, compute_dtype: torch.dtype, attention_backend: str) -> None:
        super().__init__()
        self.project_in = nn.Linear(2304, 2048, bias=False)
        self.project_out = nn.Linear(2048, 128, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(32)
        self._rotary_cache: dict[
            tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]
        ] = {}
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    2048,
                    dim_heads=64,
                    inner_dim=8192,
                    compute_dtype=compute_dtype,
                    attention_backend=attention_backend,
                )
                for _ in range(36)
            ]
        )

    def _rotary_cos_sin(
        self, seq_len: int, *, dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        key = (seq_len, dtype, device)
        cached = self._rotary_cache.get(key)
        if cached is None:
            freqs = self.rotary_pos_emb.forward_from_seq_len(seq_len)[0]
            freqs = freqs.to(dtype=dtype, device=device)
            cached = (freqs.cos(), freqs.sin())
            self._rotary_cache[key] = cached
        return cached

    def forward(self, x: Tensor, timestep_embed: Tensor) -> Tensor:
        x = self.project_in(x)
        x = torch.cat((timestep_embed.unsqueeze(1), x), dim=1)
        rope_cos, rope_sin = self._rotary_cos_sin(
            x.shape[1], dtype=x.dtype, device=x.device
        )
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)
        return self.project_out(x[:, 1:])


class DiffusionTransformer(nn.Module):
    """The checkpointed diffusion_transformer wrapper."""

    def __init__(self, *, compute_dtype: torch.dtype, attention_backend: str) -> None:
        super().__init__()
        self.transformer = ContinuousTransformer(
            compute_dtype=compute_dtype,
            attention_backend=attention_backend,
        )
        self.timestep_features = FourierFeatures(1, 256)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(256, 2048), nn.SiLU(), nn.Linear(2048, 2048)
        )
        self.preprocess_conv = nn.Conv1d(2304, 2304, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(128, 128, 1, bias=False)
        self._latent_zeros_cache: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], Tensor
        ] = {}

    def _latent_zeros(self, x: Tensor) -> Tensor:
        key = (tuple(x.shape), x.dtype, x.device)
        zeros = self._latent_zeros_cache.get(key)
        if zeros is None:
            zeros = torch.zeros_like(x)
            self._latent_zeros_cache[key] = zeros
        return zeros

    def _transformer(self, x: Tensor, t: Tensor, align_cond: Tensor) -> Tensor:
        zeros = self._latent_zeros(x)
        full = torch.cat((x, zeros, align_cond), dim=1)
        full = self.preprocess_conv(full) + full
        tfeat = self.timestep_features(t[:, None])
        temb = self.to_timestep_embed(tfeat)
        out = self.transformer(full.transpose(1, 2), temb)
        out = out.transpose(1, 2)
        return self.postprocess_conv(out) + out

    def forward(self, *, x: Tensor, t: Tensor, align_cond: Tensor) -> Tensor:
        """Keyword-only entry point used by the diffusion BCG runner."""
        return self._transformer(x, t, align_cond)


class MiniMaxMusic3DIT(nn.Module):
    """Checkpoint-backed DIT with a deterministic flow-matching wrapper."""

    def __init__(
        self,
        *,
        compute_dtype: torch.dtype = torch.float32,
        attention_backend: str = "torch_sdpa",
    ) -> None:
        super().__init__()
        self.latent_conditioners = nn.Sequential(
            nn.Conv1d(4096, 2048, kernel_size=3, padding=1)
        )
        self.diffusion_transformer = DiffusionTransformer(
            compute_dtype=compute_dtype,
            attention_backend=attention_backend,
        )
        self.cond_layer_logits = nn.Parameter(torch.empty(8))
        self.cond_layer_scale = nn.Parameter(torch.empty(1))
        self.sr_input = 24000
        self.sr_output = 44100
        self.hop_size_input = 960
        self.hop_size_output = 512
        self._bcg_runner: DiffusionBreakableCudaGraphRunner | None = None

    def enable_compiled_blocks(self, *, warmup_mel_length: int) -> None:
        """Fold each block's elementwise work into its matmul stream.s"""
        layers = self.diffusion_transformer.transformer.layers
        compiled = torch.compile(type(layers[0]).forward, dynamic=True)
        for layer in layers:
            layer.forward = MethodType(compiled, layer)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        shape = (2, 2048, warmup_mel_length)
        with torch.inference_mode(), set_forward_context(0, None):
            self.diffusion_transformer._transformer(
                torch.zeros((2, 128, warmup_mel_length), device=device, dtype=dtype),
                torch.zeros((2,), device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )

    def enable_cache_dit(
        self,
        *,
        num_steps: int,
        fn_compute_blocks: int,
        bn_compute_blocks: int,
        max_warmup_steps: int,
        residual_diff_threshold: float,
        max_continuous_cached_steps: int,
    ) -> None:
        adapter = BlockAdapter(
            transformer=self.diffusion_transformer.transformer,
            blocks=self.diffusion_transformer.transformer.layers,
            forward_pattern=ForwardPattern.Pattern_3,
            has_separate_cfg=False,
        )
        enable_cache(
            adapter,
            cache_config=DBCacheConfig(
                num_inference_steps=num_steps,
                Fn_compute_blocks=fn_compute_blocks,
                Bn_compute_blocks=bn_compute_blocks,
                max_warmup_steps=max_warmup_steps,
                residual_diff_threshold=residual_diff_threshold,
                max_continuous_cached_steps=max_continuous_cached_steps,
            ),
        )
        self._cache_dit_adapter = adapter

    def aligned_mel_length(self, frames: int) -> int:
        return max(
            1,
            int(
                frames
                * self.sr_output
                / self.sr_input
                * self.hop_size_input
                / self.hop_size_output
            ),
        )

    def enable_breakable_cuda_graph(
        self,
        *,
        mel_len: int,
        min_free_gb: float,
    ) -> bool:
        """Capture the fixed full-window DiT step, falling back on failure."""
        device = next(self.parameters()).device
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / 2**30
        if free_gb < min_free_gb:
            logger.warning(
                f"MiniMax Music 3 diffusion BCG skipped: free_gb={free_gb:.1f} below min_free_gb={min_free_gb:.1f}"
            )
            return False
        runner = DiffusionBreakableCudaGraphRunner(
            self.diffusion_transformer,
            device,
        )
        x = torch.zeros(
            (2, 128, mel_len),
            device=device,
            dtype=next(self.parameters()).dtype,
        )
        t = torch.zeros((2,), device=device, dtype=x.dtype)
        align_cond = torch.zeros((2, 2048, mel_len), device=device, dtype=x.dtype)
        with set_forward_context(0, None):
            captured = runner.capture(x=x, t=t, align_cond=align_cond)
        if not captured:
            runner.reset()
            return False
        self._bcg_runner = runner
        logger.info(
            f"MiniMax Music 3 diffusion BCG captured mel_len={mel_len} entries={len(runner.entries)} free_gb={free_gb:.1f}"
        )
        return True

    def condition(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != AR_HIDDEN_SIZE:
            raise ValueError(
                f"TTM DIT condition must be [B,T,{AR_HIDDEN_SIZE}], "
                f"got {tuple(hidden.shape)}"
            )
        hidden_cf = hidden.transpose(1, 2)
        bsz, _, frames = hidden_cf.shape
        hidden_cf = hidden_cf.reshape(bsz, 8, 4096, frames)
        weights = torch.softmax(self.cond_layer_logits, dim=0).to(hidden_cf.dtype)
        hidden_cf = torch.einsum("blht,l->bht", hidden_cf, weights)
        hidden_cf = self.cond_layer_scale.to(hidden_cf.dtype) * hidden_cf
        return self.latent_conditioners(hidden_cf)

    def aligned_condition(self, hidden: Tensor) -> Tensor:
        """Project AR hidden states and resample them to DAV latent time."""
        hidden = hidden.to(dtype=next(self.parameters()).dtype)
        align = self.condition(hidden)
        mel_len = self.aligned_mel_length(hidden.shape[1])
        return F.interpolate(align, size=mel_len, mode="nearest")

    @torch.inference_mode()
    def forward(
        self,
        align: Tensor,
        *,
        generator: torch.Generator,
        initial_latent: Tensor | None = None,
        initial_condition: Tensor | None = None,
        num_steps: int = DEFAULT_DIT_STEPS,
        cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
        should_abort: Callable[[], bool] | None = None,
    ) -> Tensor:
        """Solve a projected condition into a VAE latent [B,128,T_mel]."""
        if num_steps < 1:
            raise ValueError("MiniMax Music 3 DIT num_steps must be positive")
        mel_len = align.shape[-1]
        x = torch.randn(
            (align.shape[0], 128, mel_len),
            device=align.device,
            dtype=align.dtype,
            generator=generator,
        )
        left = 0
        latent_prompt: Tensor | None = None
        noise_prompt: Tensor | None = None
        if initial_latent is not None:
            left = min(initial_latent.shape[-1], mel_len)
            if left <= 0:
                left = 0
            else:
                latent_prompt = initial_latent[..., :left].to(x)
                noise_prompt = x[..., :left].clone()
                if initial_condition is not None:
                    align[..., :left] = initial_condition[..., :left].to(align)
        dt = 1.0 / num_steps
        cond_cfg = torch.zeros(
            (2, *align.shape[1:]), device=align.device, dtype=align.dtype
        )
        cond_cfg[0].copy_(align[0])
        for step in range(num_steps):
            if should_abort is not None and should_abort():
                raise InterruptedError("MiniMax Music 3 DIT generation aborted")
            t = torch.full(
                (x.shape[0],), step / num_steps, device=x.device, dtype=x.dtype
            )
            if left and latent_prompt is not None and noise_prompt is not None:
                x[..., :left] = (1.0 - (1.0 - 1e-6) * t[0]) * noise_prompt + t[
                    0
                ] * latent_prompt
            x_cfg = x.expand(2, -1, -1)
            t_cfg = t.expand(2)
            with set_forward_context(step, None):
                if self._bcg_runner is None:
                    d = self.diffusion_transformer._transformer(x_cfg, t_cfg, cond_cfg)
                else:
                    d = self._bcg_runner(
                        x=x_cfg,
                        t=t_cfg,
                        align_cond=cond_cfg,
                    )
            d = cfg_scale * d[:1] + (1.0 - cfg_scale) * d[1:2]
            x = x + dt * d
        if left and latent_prompt is not None:
            x[..., :left] = latent_prompt
        return x


__all__ = ["MiniMaxMusic3DIT", "_resolve_attention_backend"]
