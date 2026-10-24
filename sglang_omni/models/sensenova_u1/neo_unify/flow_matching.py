"""Flow matching."""

import math

import torch
import torch.nn as nn

# Modified for SGLang; see this directory's README.md for upstream source.


def modulate(
    x: torch.Tensor, shift: torch.Tensor | None, scale: torch.Tensor | None = None
) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale)
    else:
        pass
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size: int = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        else:
            pass
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 1.0) -> None:
        super().__init__()
        self.channels: int = channels
        self.intermediate_size: int = int(channels * mlp_ratio)

        self.in_ln: nn.LayerNorm = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )

        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class ConditionedFinalLayer(nn.Module):
    def __init__(self, hidden_feature_count: int, output_feature_count: int) -> None:
        super().__init__()
        self.norm_final: nn.LayerNorm = nn.LayerNorm(
            hidden_feature_count, elementwise_affine=False, eps=1e-6
        )
        self.linear: nn.Linear = nn.Linear(hidden_feature_count, output_feature_count)
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_feature_count, 2 * hidden_feature_count)
        )

    def forward(
        self, hidden_states: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep_embedding).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(hidden_states), shift, scale))


class TimestepConditionedMLP(nn.Module):
    """Deep flow head, separate from the patch-conditioned pixel MLP below."""

    def __init__(
        self,
        input_feature_count: int,
        output_feature_count: int,
        hidden_feature_count: int,
        layer_count: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.time_embed: TimestepEmbedder = TimestepEmbedder(hidden_feature_count)
        self.input_proj: nn.Linear = nn.Linear(
            input_feature_count, hidden_feature_count
        )
        self.res_blocks: nn.ModuleList = nn.ModuleList(
            [ResBlock(hidden_feature_count, mlp_ratio) for _ in range(layer_count)]
        )
        self.final_layer: ConditionedFinalLayer = ConditionedFinalLayer(
            hidden_feature_count, output_feature_count
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                else:
                    pass
            else:
                pass
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in (*self.res_blocks, self.final_layer):
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self, hidden_states: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.input_proj(hidden_states)
        timestep_embedding = self.time_embed(timesteps)
        for residual_block in self.res_blocks:
            hidden_states = residual_block(hidden_states, timestep_embedding)
        return self.final_layer(hidden_states, timestep_embedding)


class FlowMatchingHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        dim: int = 1536,
        layers: int = 12,
        mlp_ratio: float = 1.0,
    ) -> None:
        super(FlowMatchingHead, self).__init__()
        self.net: TimestepConditionedMLP = TimestepConditionedMLP(
            input_feature_count=input_dim,
            output_feature_count=out_dim,
            hidden_feature_count=dim,
            layer_count=layers,
            mlp_ratio=mlp_ratio,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.net.input_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.net.input_proj.weight.device

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.net(x, t)
        return x


class SimpleMLPAdaLN(nn.Module):
    """The MLP for Diffusion Loss."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels: int = in_channels
        self.model_channels: int = model_channels
        self.out_channels: int = out_channels
        self.num_res_blocks: int = num_res_blocks
        self.grad_checkpointing: bool = grad_checkpointing
        self.patch_size: int = patch_size

        self.cond_embed: nn.Linear = nn.Linear(
            z_channels, patch_size**2 * model_channels
        )

        self.input_proj: nn.Linear = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks: nn.ModuleList = nn.ModuleList(res_blocks)
        self.final_layer: FinalLayer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                else:
                    pass
            else:
                pass

        self.apply(_basic_init)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply the model to an input batch."""
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(-1, self.patch_size**2, self.model_channels)

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x)


class FinalLayer(nn.Module):
    """The final layer adopted from DiT."""

    def __init__(self, model_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final: nn.LayerNorm = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear: nn.Linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        x = self.linear(x)
        return x
