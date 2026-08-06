from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang_omni.models.dots_tts.components.backbone.inference_utils import (
    project_attention,
)
from sglang_omni.models.dots_tts.components.backbone.layers import (
    Conv1d,
    Mlp,
    MultiHeadAttention,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=16,
        ffn_hidden_size=4096,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_drop=attn_dropout,
            norm_layer=norm_layer,
            **kwargs,
        )
        norm_cls = getattr(nn, norm_layer)
        self.attn_norm = norm_cls(hidden_size)
        self.ffn = Mlp(
            hidden_size, ffn_hidden_size, dropout=ffn_dropout, act_layer=nn.SiLU
        )
        self.ffn_norm = norm_cls(hidden_size)
        self.hidden_size = hidden_size

    def _build_causal_mask(self, T: int, device):
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def _build_padding_mask(self, x_lens, max_len: int, device):
        B = x_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        return positions < x_lens.unsqueeze(1)

    def _fuse_attn_mask(self, causal_mask, padding_mask):
        if causal_mask is None and padding_mask is None:
            return None
        if causal_mask is None:
            row = padding_mask.unsqueeze(2)
            col = padding_mask.unsqueeze(1)
            return row & col
        if padding_mask is None:
            return causal_mask.unsqueeze(0)

        _B, _T = padding_mask.shape
        causal = causal_mask.unsqueeze(0)
        row = padding_mask.unsqueeze(2)
        col = padding_mask.unsqueeze(1)
        pad_2d = row & col
        return causal & pad_2d

    def forward(
        self,
        x,
        x_lens=None,
        causal=True,
    ):
        _B, T, C = x.shape
        assert self.hidden_size == C
        device = x.device

        causal_mask = self._build_causal_mask(T, device) if causal else None
        if x_lens is not None:
            padding_mask = self._build_padding_mask(x_lens, T, device)
        else:
            padding_mask = None
        fused_mask = self._fuse_attn_mask(causal_mask, padding_mask)

        h = self.attn_norm(x)
        h = self.attn(
            q=h,
            mask=fused_mask,
        )
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h


class SuperviseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_heads=config.get("num_heads", 16),
                    ffn_hidden_size=config.get("ffn_hidden_size", 4096),
                    norm_layer=config.get("norm_layer", "LayerNorm"),
                )
                for _ in range(config.get("num_layers", 6))
            ]
        )
        self.causal = config.get("causal", False)

    def forward(self, x, x_lens=None):
        batch_size, seq_len, _ = x.shape
        if x_lens is None:
            x_lens = torch.full(
                (batch_size,), seq_len, device=x.device, dtype=torch.long
            )
        for layer in self.layers:
            x = layer(x, x_lens=x_lens, causal=self.causal)
        return x


class VAESemanticEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        self.expects_normalized_input = False
        in_ds_rate = 2
        self.patch_size = int(config.patch_size)
        self.in_ds_rate = in_ds_rate
        self.ds_proj = Conv1d(
            in_dim, in_dim, kernel_size=in_ds_rate, stride=in_ds_rate, causal=True
        )
        self.in_proj = nn.Linear(in_dim, config.PatchEncoder.hidden_size)
        self.encoder = SuperviseEncoder(config.PatchEncoder)
        self.out_ds_rate = self.patch_size // in_ds_rate
        self.out_proj = nn.Linear(
            config.PatchEncoder.hidden_size * self.out_ds_rate, out_dim
        )

    @property
    def input_dtype(self) -> torch.dtype:
        return self.in_proj.weight.dtype

    def forward(self, x, x_lens=None):
        x = self._downsample(x)
        x = self.in_proj(x)
        z = self.encoder(x, x_lens=x_lens)
        return self._project_embeddings(z)

    def _downsample(self, x):
        return self.ds_proj(x.transpose(1, 2)).transpose(1, 2)

    def _project_embeddings(self, z):
        if self.out_ds_rate > 1:
            z = rearrange(z, "b (s d) h -> b s (d h)", d=self.out_ds_rate)
        return self.out_proj(z)


class SemanticEncoderDecodeStep(nn.Module):
    """One cached patch step of :class:`VAESemanticEncoder`.

    The caller owns the K/V cache and the conv tail: it passes the per-row cache
    slices plus the mask/rotary for this step and writes the returned K/V back at
    whatever offset each row is at.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        if not isinstance(encoder, VAESemanticEncoder):
            raise TypeError(
                f"SemanticEncoderDecodeStep needs a VAESemanticEncoder, got {type(encoder)!r}"
            )
        self.encoder = encoder
        self.out_ds_rate = int(encoder.out_ds_rate)
        attn = encoder.encoder.layers[0].attn
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.num_layers = len(encoder.encoder.layers)

    @staticmethod
    def downsample(
        encoder: nn.Module,
        latent_patch: torch.Tensor,
        *,
        conv_tail: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal ``ds_proj`` over one patch, continued from ``conv_tail``."""
        raw = latent_patch.transpose(1, 2)
        ds_proj = encoder.ds_proj
        projected = F.conv1d(
            torch.cat([conv_tail, raw], dim=-1),
            ds_proj.weight,
            ds_proj.bias,
            stride=ds_proj.stride[0],
            padding=0,
            dilation=ds_proj.dilation[0],
            groups=ds_proj.groups,
        ).transpose(1, 2)
        return encoder.in_proj(projected), raw[..., -ds_proj.left_padding :]

    def forward(
        self,
        latent_patch: torch.Tensor,
        conv_tail: torch.Tensor,
        kv_buffer: tuple[torch.Tensor, torch.Tensor],
        kv_prefix_len: int,
        attn_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one patch; new K/V land in ``kv_buffer`` after ``kv_prefix_len``."""
        x, new_conv_tail = self.downsample(
            self.encoder, latent_patch, conv_tail=conv_tail
        )
        key_buffer, value_buffer = kv_buffer
        for layer_idx, layer in enumerate(self.encoder.encoder.layers):
            attn_out, _key, _value = project_attention(
                layer.attn,
                layer.attn_norm(x),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_cos=rotary_cos if layer.attn.rotary_bias else None,
                rotary_sin=rotary_sin if layer.attn.rotary_bias else None,
                kv_buffer=(key_buffer[layer_idx], value_buffer[layer_idx]),
                kv_prefix_len=kv_prefix_len,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))
        return self.encoder._project_embeddings(x), new_conv_tail
