# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR audio tower: Whisper encoder with RoPE self-attention + MLP frame-merge adapter.

Reimplemented natively (not via the checkpoint's remote code) because that code
targets the transformers-4 WhisperEncoderLayer API, which is incompatible with
transformers 5. Structure mirrors the checkpoint's modeling_audio.py /
modeling_arkasr.py: conv1/conv2 mel frontend -> RoPE-augmented Whisper encoder
layers -> LayerNorm -> (merge_factor frame merge) -> 2-layer MLP to LLM hidden.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix
from torch.nn.functional import scaled_dot_product_attention
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer


class ArkRotaryEmbedding(nn.Module):
    """RoPE cache generator (matches checkpoint modeling_audio.RotaryEmbedding)."""

    def __init__(self, dim: int, rope_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.rope_ratio = rope_ratio

    @torch.no_grad()
    def get_emb(self, seq_len, dtype, device, base: int = 10000):
        base = base * self.rope_ratio
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
                / self.dim
            )
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
        emb = torch.stack(
            [torch.cos(freqs), torch.sin(freqs)], dim=-1
        )  # [seq_len, dim/2, 2]
        if dtype in (torch.float16, torch.bfloat16):
            emb = emb.to(dtype)
        return emb


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """x: [b, nh, sq, hd]; rope_cache: [1, sq, rot_dim/2, 2]."""
    b, nh, sq, hd = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_shaped = x_rot.reshape(b, nh, sq, rot_dim // 2, 2)
    cos = rope_cache[..., 0].unsqueeze(1)  # [1, 1, sq, rot_dim/2]
    sin = rope_cache[..., 1].unsqueeze(1)
    x_out = torch.stack(
        [
            x_shaped[..., 0] * cos - x_shaped[..., 1] * sin,
            x_shaped[..., 1] * cos + x_shaped[..., 0] * sin,
        ],
        dim=-1,
    )
    x_out = x_out.flatten(3)
    return torch.cat([x_out, x_pass], dim=-1)


class WhisperRoPESdpaAttention(nn.Module):
    """Whisper self-attention with RoPE, SDPA backend (matches checkpoint)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        if quant_config is None:
            self.reset_parameters()
        self.is_causal = False

    def reset_parameters(self) -> None:
        """Match the former nn.Linear initialization for standalone use."""
        nn.init.kaiming_uniform_(self.qkv_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.embed_dim)
        if self.qkv_proj.bias is not None:
            nn.init.uniform_(self.qkv_proj.bias, -bound, bound)
            # The checkpoint has no K bias; keep the packed K slice equivalent.
            nn.init.zeros_(self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim])
        if self.out_proj.bias is not None:
            nn.init.uniform_(self.out_proj.bias, -bound, bound)

    def forward(
        self, hidden_states, attention_mask=None, rotary_pos_emb=None, **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = (
            q.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)
        attn = scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=self.is_causal
        )
        attn = attn.transpose(1, 2).contiguous().reshape(bsz, q_len, self.embed_dim)
        attn, _ = self.out_proj(attn)
        return attn, None, None


class WhisperSpecialEncoderLayer(WhisperEncoderLayer):
    """WhisperEncoderLayer with the self-attn swapped for the RoPE SDPA variant."""

    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.self_attn = WhisperRoPESdpaAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        rotary_pos_emb=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # Match the reference (modeling_audio.py): clamp after the final residual
        # so the fp16 path stays finite. fp16 max is ~65504 and deep-layer
        # activations can overflow to inf/NaN; bf16 has fp32-range exponents so
        # this is a no-op there and the bf16 path is unchanged.
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return (hidden_states, None, None)


class ArkAudioTower(nn.Module):
    """conv frontend + RoPE Whisper encoder layers + LayerNorm.

    Consumes mel features (B, num_mel_bins, T) and returns (B, T_down, d_model).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        wc = config.whisper_config
        embed_dim = wc.d_model
        wc._attn_implementation = "sdpa"
        self.conv1 = nn.Conv1d(wc.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(wc.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList(
            [
                WhisperSpecialEncoderLayer(
                    wc,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{index}", prefix),
                )
                for index in range(wc.encoder_layers)
            ]
        )
        self.use_rope = bool(getattr(config, "use_rope", True))
        if self.use_rope:
            head_dim = embed_dim // wc.encoder_attention_heads
            self.rotary_embedding = ArkRotaryEmbedding(head_dim // 2)
        # checkpoint disables the tower's own final LayerNorm (Identity) and
        # applies a separate LayerNorm in the adapter instead.
        self.layer_norm = nn.Identity()

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode mel features, optionally masking right-padded frames.

        ``attention_mask`` is a [B, T] right-padded mask for the input mel
        frames.  It clears invalid conv1 frames before conv2, then is
        downsampled for SDPA.  This gives a short sample in a mixed-length
        batch the same right-edge convolution context and encoder states it
        had when encoded alone.
        """
        input_frame_mask = None
        if attention_mask is not None:
            expected_shape = (input_features.shape[0], input_features.shape[-1])
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    "ARK-ASR attention_mask shape "
                    f"{tuple(attention_mask.shape)} does not match mel frames "
                    f"{tuple(expected_shape)}"
                )
            input_frame_mask = attention_mask.to(
                device=input_features.device, dtype=torch.bool
            )
            input_features = input_features * input_frame_mask.unsqueeze(1).to(
                dtype=input_features.dtype
            )

        inputs_embeds = F.gelu(self.conv1(input_features))
        if input_frame_mask is not None:
            inputs_embeds = inputs_embeds * input_frame_mask.unsqueeze(1).to(
                dtype=inputs_embeds.dtype
            )
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # [B, T_down, D]
        frame_mask = None
        sdpa_mask = None
        if input_frame_mask is not None:
            # note (nagisa-kunhah): Keep the downsampled mask contiguous for SDPA.
            frame_mask = input_frame_mask[:, ::2].contiguous()
            sdpa_mask = frame_mask[:, None, None, :]
            inputs_embeds = inputs_embeds * frame_mask.unsqueeze(-1).to(
                dtype=inputs_embeds.dtype
            )
        if self.use_rope:
            rope = self.rotary_embedding.get_emb(
                inputs_embeds.shape[1], inputs_embeds.dtype, inputs_embeds.device
            ).unsqueeze(0)
            hidden_states = inputs_embeds
        else:
            rope = None
            hidden_states = (
                inputs_embeds + self.embed_positions.weight[: inputs_embeds.shape[1]]
            )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=sdpa_mask, rotary_pos_emb=rope
            )[0]
            if frame_mask is not None:
                hidden_states = hidden_states * frame_mask.unsqueeze(-1).to(
                    dtype=hidden_states.dtype
                )
        return self.layer_norm(hidden_states)


class ArkAudioMLPAdapter(nn.Module):
    """Whisper tower + merge_factor frame-merge + 2-layer MLP to LLM hidden."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        wc = config.whisper_config
        self.merge_factor = int(config.merge_factor)
        self.whisper = ArkAudioTower(
            config,
            quant_config=quant_config,
            prefix=add_prefix("whisper", prefix),
        )
        self.layer_norm = nn.LayerNorm(wc.hidden_size)
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "selu": nn.SELU()}
        act = act_map.get(getattr(config, "mlp_adapter_act", "gelu"), nn.GELU())
        input_dim = wc.hidden_size * self.merge_factor
        output_dim = config.hidden_size
        self.adapting = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            act,
            nn.Linear(output_dim * 2, output_dim),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.whisper.dtype

    def forward(
        self,
        audios: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = audios.size(0)
        encoded = self.whisper(audios, attention_mask=attention_mask)  # (B, T, D)
        encoded = self.layer_norm(encoded)
        if attention_mask is not None:
            encoded = encoded * attention_mask[:, ::2].unsqueeze(-1).to(
                dtype=encoded.dtype
            )
        seq_len = encoded.size(1)
        if seq_len % self.merge_factor != 0:
            target_len = (seq_len // self.merge_factor) * self.merge_factor
            if target_len <= 0:
                target_len = self.merge_factor
                if seq_len < target_len:
                    pad = encoded.new_zeros(
                        (bsz, target_len - seq_len, encoded.size(-1))
                    )
                    encoded = torch.cat([encoded, pad], dim=1)
            else:
                encoded = encoded[:, :target_len, :]
        encoded = encoded.reshape(bsz, -1, encoded.size(-1) * self.merge_factor)
        return self.adapting(encoded)  # (B, T/merge, hidden)


__all__ = ["ArkAudioTower", "ArkAudioMLPAdapter"]
