import math
from typing import Any, Callable

import torch
import torch.nn as nn

from sglang_omni.models.dots_tts.components.backbone.inference_utils import (
    build_fused_qkv_projection,
    project_attention,
)
from sglang_omni.models.dots_tts.components.backbone.layers import (
    Mlp,
    MultiHeadAttention,
)


def modulate(x, shift, scale, **_kwargs):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, **_kwargs):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_size: int = 1024,
        modulation: bool = False,
        eps: float = 1e-5,
        **_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.attn = attention
        self.ffn = ffn
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, condition=None, mask=None, **kwargs):
        if condition is None:
            assert (
                not self.modulation
            ), "Without global condition, must set modulation to False"
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(condition).chunk(6, dim=1)
            )

        if condition is not None:
            pack_indices = kwargs.get("pack_indices")
            if pack_indices is not None:
                gate_attn = gate_attn[pack_indices]
                gate_ffn = gate_ffn[pack_indices]
            else:
                gate_attn = gate_attn.unsqueeze(1)
                gate_ffn = gate_ffn.unsqueeze(1)

        if condition is not None:
            x = x + gate_attn * self.attn(
                modulate(self.norm1(x), shift_attn, scale_attn, **kwargs),
                mask=mask,
                **kwargs,
            )
        else:
            x = x + self.attn(self.norm1(x), mask=mask, **kwargs)

        if condition is not None:
            x = x + gate_ffn * self.ffn(
                modulate(self.norm2(x), shift_ffn, scale_ffn, **kwargs)
            )
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        transformer_config,
        *,
        mode: str = "flow_matching",
    ):
        super().__init__()
        if mode not in {"flow_matching", "meanflow"}:
            raise ValueError(
                f"DiT mode must be 'flow_matching' or 'meanflow', got {mode!r}."
            )

        transformer_kwargs = transformer_config.to_dict()
        model_dim = transformer_config.hidden_size
        self.mode = mode
        self.num_layers = transformer_config.num_layers

        self.input_layer = nn.Linear(in_dim, model_dim)
        self.time_embedder = TimestepEmbedder(model_dim)
        if mode == "meanflow":
            self.duration_embedder = TimestepEmbedder(model_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            attn_block = MultiHeadAttention(**transformer_kwargs, name=f"layer_{i}")
            ffn_block = Mlp(
                act_layer=lambda: nn.GELU(approximate="tanh"), **transformer_kwargs
            )
            self.blocks.append(
                DiTBlock(attention=attn_block, ffn=ffn_block, **transformer_kwargs)
            )

        self.output_layer = FinalLayer(model_dim, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.linear.weight, 0)
        nn.init.constant_(self.output_layer.linear.bias, 0)

    def forward(
        self,
        x,
        timesteps,
        duration: torch.Tensor | None = None,
        mask=None,
        attn_mask=None,
        g_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        t = self.time_embedder(timesteps)
        c = t
        duration_embedder = getattr(self, "duration_embedder", None)
        if duration_embedder is not None and duration is not None:
            c = c + duration_embedder(duration)
        if g_cond is not None:
            c = c + g_cond

        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, mask=attn_mask, **kwargs)
        return self.output_layer(x, c, **kwargs)


class FusedQKVMultiHeadAttention(nn.Module):
    """``MultiHeadAttention`` with a fused Q/K/V projection. Self-attention only."""

    _INHERITED_ATTRS = (
        "num_heads",
        "head_dim",
        "rotary_bias",
        "q_norm",
        "k_norm",
        "attn_drop",
        "o_proj",
        "o_dropout",
    )

    def __init__(self, old_attn: nn.Module):
        super().__init__()
        for name in self._INHERITED_ATTRS:
            setattr(self, name, getattr(old_attn, name))
        self.qkv_proj = build_fused_qkv_projection(old_attn)
        if self.rotary_bias:
            self.rotary = old_attn.rotary

    def forward(
        self,
        q: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        rotary_cos = rotary_sin = None
        if self.rotary_bias:
            if pos_ids is None:
                pos_ids = torch.arange(q.size(1), device=q.device, dtype=torch.float32)
            rotary = self.rotary(pos_ids)
            rotary_cos, rotary_sin = rotary.cos(), rotary.sin()
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None, :, :]
        out, _k, _v = project_attention(
            self,
            q,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            attn_mask=mask,
        )
        return out


class FusedAdaLNDiT(nn.Module):
    """DiT rewired so all per-block adaLN linears fuse into a single Linear.

    Inference-only, and it requires every ``DiTBlock`` to have been trained with
    ``modulation=True``. Fusing lets a whole generation precompute its
    modulations once per ODE step instead of once per block per step.
    """

    def __init__(self, old_dit: nn.Module):
        super().__init__()
        if any(not block.modulation for block in old_dit.blocks):
            raise ValueError("FusedAdaLNDiT requires modulation=True for every block.")
        for block in old_dit.blocks:
            block.attn = FusedQKVMultiHeadAttention(block.attn)

        self.input_layer = old_dit.input_layer
        self.time_embedder = old_dit.time_embedder
        self.duration_embedder = getattr(old_dit, "duration_embedder", None)
        self.blocks = old_dit.blocks
        self.output_layer = old_dit.output_layer

        model_dim = int(self.input_layer.out_features)
        adaln_linears = [block.adaLN_modulation[-1] for block in self.blocks]
        adaln_linears.append(self.output_layer.adaLN_modulation[-1])
        first_linear = adaln_linears[0]

        self.fused_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                model_dim,
                model_dim * (6 * len(self.blocks) + 2),
                bias=first_linear.bias is not None,
            ),
        )
        self.fused_adaln.to(
            device=first_linear.weight.device, dtype=first_linear.weight.dtype
        )
        with torch.no_grad():
            self.fused_adaln[-1].weight.copy_(
                torch.cat([lin.weight.detach() for lin in adaln_linears], dim=0)
            )
            if self.fused_adaln[-1].bias is not None:
                self.fused_adaln[-1].bias.copy_(
                    torch.cat([lin.bias.detach() for lin in adaln_linears], dim=0)
                )

        for block in self.blocks:
            block.adaLN_modulation = nn.Identity()
        self.output_layer.adaLN_modulation = nn.Identity()

    def split_mods(
        self, all_mods: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        """Split the fused adaLN output into per-block mods + final (shift, scale)."""
        model_dim = int(self.input_layer.out_features)
        block_width = 6 * model_dim
        final_start = block_width * len(self.blocks)
        block_mods = all_mods[:, :final_start].split(block_width, dim=1)
        shift, scale = all_mods[:, final_start : final_start + 2 * model_dim].chunk(
            2, dim=1
        )
        return block_mods, (shift, scale)

    def run_modulated_blocks(
        self,
        *,
        x: torch.Tensor,
        all_mods: torch.Tensor,
        attention: Callable[[int, nn.Module, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run all blocks with a caller-supplied attention operator.

        The callback receives ``(layer_idx, block, attn_in)``; the cached decode
        path reads/writes its KV there while the plain path just forwards.
        """
        block_mods, final_mod = self.split_mods(all_mods)
        x = self.input_layer(x)
        for layer_idx, (block, block_mod) in enumerate(
            zip(self.blocks, block_mods, strict=True)
        ):
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                block_mod.chunk(6, dim=1)
            )
            attn_in = modulate(block.norm1(x), shift_attn, scale_attn)
            x = x + gate_attn.unsqueeze(1) * attention(layer_idx, block, attn_in)
            ffn_in = modulate(block.norm2(x), shift_ffn, scale_ffn)
            x = x + gate_ffn.unsqueeze(1) * block.ffn(ffn_in)
        return x, final_mod

    def apply_final_layer(
        self, x: torch.Tensor, final_mod: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        shift, scale = final_mod
        return self.output_layer.linear(
            modulate(self.output_layer.norm(x), shift, scale)
        )

    def build_mods(
        self,
        timesteps: torch.Tensor,
        *,
        duration: torch.Tensor | None = None,
        g_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused adaLN modulations for a (timestep, duration, speaker) condition."""
        c = self.time_embedder(timesteps)
        if self.duration_embedder is not None and duration is not None:
            c = c + self.duration_embedder(duration)
        if g_cond is not None:
            c = c + g_cond
        return self.fused_adaln(c)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        duration: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None,
        g_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Uncached full-sequence forward. Not on the serving path."""
        all_mods = self.build_mods(timesteps, duration=duration, g_cond=g_cond)
        x, final_mod = self.run_modulated_blocks(
            x=x,
            all_mods=all_mods,
            attention=lambda _i, block, attn_in: block.attn(
                attn_in, mask=attn_mask, pos_ids=pos_ids
            ),
        )
        return self.apply_final_layer(x, final_mod)


def fuse_dit_for_inference(core: Any) -> FusedAdaLNDiT:
    """Wrap ``core.velocity_field_predictor`` in ``FusedAdaLNDiT`` (idempotent)."""
    dit = core.velocity_field_predictor
    if not isinstance(dit, FusedAdaLNDiT):
        dit = FusedAdaLNDiT(dit)
        core.velocity_field_predictor = dit
    return dit
