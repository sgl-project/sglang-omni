"""Prefix prefill and image velocity prediction for native generation."""

import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.sampling import GENERATION_TIME_EPSILON


def t2i_prefix_forward(
    model: NEOChatModel,
    input_ids: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: dict[str, torch.Tensor],
) -> tuple[DynamicCache, torch.Tensor]:
    out = model.language_model.model(
        input_ids=input_ids,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return out.past_key_values, out.last_hidden_state


def it2i_prefix_forward(
    model: NEOChatModel,
    input_imbeds: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: dict[str, torch.Tensor],
) -> tuple[DynamicCache, torch.Tensor]:
    out = model.language_model.model(
        inputs_embeds=input_imbeds,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return out.past_key_values, out.last_hidden_state


def t2i_predict_v(
    model: NEOChatModel,
    input_embeds: torch.Tensor,
    indexes_image: torch.Tensor,
    attn_mask: dict[str, torch.Tensor | None] | None,
    past_key_values: DynamicCache,
    t: torch.Tensor,
    z: torch.Tensor,
    image_token_num: int,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    B, L = z.shape[0], z.shape[1]

    outputs = model.language_model.model(
        inputs_embeds=input_embeds,
        image_only=True,
        indexes=indexes_image,
        attention_mask=attn_mask,
        past_key_values=past_key_values,
        update_cache=False,
        use_cache=True,
    )

    if model.use_pixel_head:
        merge_size = int(1 / model.downsample_ratio)
        token_h = image_size[1] // (model.patch_size * merge_size)
        token_w = image_size[0] // (model.patch_size * merge_size)

        img_reshaped = outputs.last_hidden_state[:, -image_token_num:].view(
            B, token_h, token_w, -1
        )
        img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
        img_2d = img_2d.contiguous().view(B, -1, token_h, token_w)

        smoothed_img_2d = model.fm_modules["fm_head"](img_2d)

        smoothed_reshaped = smoothed_img_2d.view(
            B,
            3,
            token_h,
            model.patch_size * merge_size,
            token_w,
            model.patch_size * merge_size,
        )
        smoothed_reshaped = torch.einsum(
            "b c h p w q -> b h w p q c", smoothed_reshaped
        )
        out_1d = smoothed_reshaped.contiguous().view(
            B, L, model.patch_size * merge_size * model.patch_size * merge_size * 3
        )
        x_pred = out_1d
    else:
        if model.use_deep_fm_head:
            x_pred = model.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B * L, -1),
                t.repeat(B * L),
            ).view(B, L, -1)
        else:
            x_pred = model.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)
            ).view(B, L, -1)

    v_pred = (x_pred - z) / (1 - t).clamp_min(GENERATION_TIME_EPSILON)
    return v_pred
