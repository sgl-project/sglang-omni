"""SenseNova vision embeddings and rotary positions."""

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)

# Modified for SGLang; see this directory's README.md for upstream source.


def precompute_rope_freqs_sincos(
    dim: int,
    max_position: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute 1D RoPE cosine and sine values."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute patch coordinates (x, y)"""
    device = grid_hw.device
    B = grid_hw.shape[0]

    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)

    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = (
        patch_id_within_image
        - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[
            patch_to_sample
        ]
    )

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D RoPE to part of the input tensor."""

    cos = cos_cached[positions]
    sin = sin_cached[positions]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D RoPE to input tensor x."""
    dim = x.shape[-1]
    dim_half = dim // 2

    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    rotated_part_1 = apply_rotary_emb_1d(
        x_part_1, cos_cached_x, sin_cached_x, abs_positions_x
    )
    rotated_part_2 = apply_rotary_emb_1d(
        x_part_2, cos_cached_y, sin_cached_y, abs_positions_y
    )

    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class NEOVisionEmbeddings(nn.Module):
    """Embedding Module for Vision."""

    def __init__(self, config: NEOVisionConfig) -> None:
        super().__init__()
        self.config: NEOVisionConfig = config
        self.embed_dim: int = config.hidden_size
        self.llm_embed_dim: int = config.llm_hidden_size[0]
        self.downsample_factor: int = int(1 / config.downsample_ratio[0])
        self.patch_size: int = config.patch_size

        self.patch_embedding: nn.Conv2d = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding: nn.Conv2d = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu: nn.GELU = nn.GELU()

        self.rope_dim_part: int = self.embed_dim // 2
        self.max_position_embeddings_vision: int = config.max_position_embeddings_vision
        self.rope_theta_vision: float = config.rope_theta_vision

        self.register_buffer("cos_cached_x", None, persistent=False)
        self.register_buffer("sin_cached_x", None, persistent=False)
        self.register_buffer("cos_cached_y", None, persistent=False)
        self.register_buffer("sin_cached_y", None, persistent=False)

    def ensure_rotary_embedding_cache(self, device: torch.device) -> None:
        if self.cos_cached_x is not None and self.cos_cached_x.device == device:
            return
        else:
            pass

        cos, sin = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            self.max_position_embeddings_vision,
            base=self.rope_theta_vision,
            device=device,
        )
        self.cos_cached_x: torch.Tensor = cos
        self.sin_cached_x: torch.Tensor = sin
        self.cos_cached_y: torch.Tensor = cos.clone()
        self.sin_cached_y: torch.Tensor = sin.clone()

    def apply_2d_rotary_position_embeddings(
        self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply 2D Rotary Position Embedding to the patch embeddings."""
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(
            grid_hw, device=patch_embeds.device
        )
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(
        self, pixel_values: torch.FloatTensor, grid_hw: torch.Tensor | None = None
    ) -> torch.Tensor:

        pixel_values = pixel_values.view(
            -1,
            3,
            self.patch_size,
            self.patch_size,
        )
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(
            -1, self.embed_dim
        )
        self.ensure_rotary_embedding_cache(patch_embeds.device)
        patch_embeds = self.apply_2d_rotary_position_embeddings(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = (
                patch_embeds[cur_position : cur_position + h * w]
                .view(h, w, -1)
                .unsqueeze(0)
            )
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)

        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(
            patch_embeds.shape[0] / self.downsample_factor**2
        )

        return embeddings


class NEOVisionModel(PreTrainedModel):
    main_input_name: str = "pixel_values"
    _supports_flash_attn_2: bool = True
    supports_gradient_checkpointing: bool = True
    config_class: type[NEOVisionConfig] = NEOVisionConfig
    _tp_plan: str = ""

    def __init__(self, config: NEOVisionConfig) -> None:
        super().__init__(config)
        self.config: NEOVisionConfig = config

        self.embeddings: NEOVisionEmbeddings = NEOVisionEmbeddings(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        pixel_embeds: torch.FloatTensor | None = None,
        grid_hw: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
        else:
            pass

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            assert (
                pixel_values.dim() == 2
            ), f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
            hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
        )
