"""Interleaved image denoise."""

import math
from typing import Literal

import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    clear_flash_kv_cache,
    prepare_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState

# Modified for SGLang; see this directory's README.md for upstream source.


def interleaved_image_denoise(
    cfg_interval: tuple[float, float],
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"],
    cfg_scale: float,
    device: torch.device,
    enable_timestep_shift: bool,
    generated_images: list[torch.Tensor],
    image_size: tuple[int, int] | list[tuple[int, int]],
    image_size_list: list[tuple[int, int]],
    img_cfg_scale: float,
    img_count: int,
    indexes_iu: torch.Tensor,
    merge_size: int,
    num_steps: int,
    outputs_cond: CausalLMOutputWithPast,
    past_key_values_cond: DynamicCache,
    past_key_values_iu: DynamicCache,
    past_key_values_tu: DynamicCache,
    self: GenerationState,
    t_index_cond: int,
    t_index_tu: int,
    timestep_shift: float,
    verbose: bool,
) -> torch.Tensor:
    cur_image_size = image_size_list[img_count]
    token_h = cur_image_size[1] // (self.patch_size * merge_size)
    token_w = cur_image_size[0] // (self.patch_size * merge_size)

    indexes_image_condition = self.build_t2i_image_indexes(
        token_h, token_w, t_index_cond + 1, device=device
    )
    indexes_image_text_uncondition = self.build_t2i_image_indexes(
        token_h, token_w, t_index_tu + 1, device=device
    )
    indexes_image_img_uncondition = self.build_t2i_image_indexes(
        token_h, token_w, indexes_iu[0].max() + 1, device=device
    )

    grid_h = cur_image_size[1] // self.patch_size
    grid_w = cur_image_size[0] // self.patch_size
    gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

    noise_scale = self.noise_scale
    if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(self.noise_scale_base_image_seq_len)
        noise_scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base) * float(
            self.noise_scale
        )
        if self.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
        else:
            pass
    else:
        pass
    noise_scale = min(noise_scale, self.noise_scale_max_value)
    image_prediction = noise_scale * torch.randn(
        (1, 3, cur_image_size[1], cur_image_size[0]),
        device=device,
        dtype=outputs_cond.logits.dtype,
    )

    past_key_values_cond_cfg = past_key_values_cond
    past_key_values_tu_cfg = past_key_values_tu
    past_key_values_iu_cfg = past_key_values_iu

    attention_mask_condition = {"full_attention": None}
    attention_mask_text_uncondition = {"full_attention": None}
    attention_mask_img_uncondition = {"full_attention": None}

    prepare_flash_kv_cache(
        past_key_values_cond_cfg,
        current_len=token_h * token_w,
        batch_size=1,
    )
    prepare_flash_kv_cache(
        past_key_values_tu_cfg,
        current_len=token_h * token_w,
        batch_size=1,
    )
    prepare_flash_kv_cache(
        past_key_values_iu_cfg,
        current_len=token_h * token_w,
        batch_size=1,
    )

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if enable_timestep_shift:
        timesteps = self.apply_time_schedule(
            timesteps, token_h * token_w, timestep_shift
        )
    else:
        pass

    step_iter = range(num_steps)
    if verbose:
        try:
            from tqdm import tqdm as _tqdm

            step_iter = _tqdm(
                step_iter,
                desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})",
                total=num_steps,
                leave=False,
            )
        except ImportError:
            pass
    else:
        pass
    for step_i in step_iter:
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]

        z = self.patchify(image_prediction, self.patch_size * merge_size)
        image_input = self.patchify(
            image_prediction, self.patch_size, channel_first=True
        )
        image_embeds = self.extract_feature(
            image_input.view(1 * grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=gen_grid_hw,
        ).view(1, token_h * token_w, -1)
        t_expanded = t.expand(token_h * token_w)
        timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(
            1, token_h * token_w, -1
        )
        if self.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(
                t_expanded, noise_scale / self.noise_scale_max_value
            )
            noise_embeddings = self.fm_modules["noise_scale_embedder"](
                noise_scale_tensor
            ).view(1, token_h * token_w, -1)
            timestep_embeddings += noise_embeddings
        else:
            pass
        image_embeds = image_embeds + timestep_embeddings

        use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
        out_cond = self.t2i_predict_v(
            image_embeds,
            indexes_image_condition,
            attention_mask_condition,
            past_key_values_cond_cfg,
            t,
            z,
            image_token_num=token_h * token_w,
            timestep_embeddings=timestep_embeddings,
            image_size=cur_image_size,
        )
        if not use_cfg:
            v_pred = out_cond
        elif cfg_scale == 1 and img_cfg_scale == 1:
            v_pred = out_cond
        elif img_cfg_scale == 1:
            out_img_cond = self.t2i_predict_v(
                image_embeds,
                indexes_image_text_uncondition,
                attention_mask_text_uncondition,
                past_key_values_tu_cfg,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=cur_image_size,
            )
            v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
        elif cfg_scale == img_cfg_scale:
            out_uncond = self.t2i_predict_v(
                image_embeds,
                indexes_image_img_uncondition,
                attention_mask_img_uncondition,
                past_key_values_iu_cfg,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=cur_image_size,
            )
            v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            out_img_cond = self.t2i_predict_v(
                image_embeds,
                indexes_image_text_uncondition,
                attention_mask_text_uncondition,
                past_key_values_tu_cfg,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=cur_image_size,
            )
            out_uncond = self.t2i_predict_v(
                image_embeds,
                indexes_image_img_uncondition,
                attention_mask_img_uncondition,
                past_key_values_iu_cfg,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=cur_image_size,
            )
            v_pred = (
                out_uncond
                + cfg_scale * (out_cond - out_img_cond)
                + img_cfg_scale * (out_img_cond - out_uncond)
            )
        if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
            if cfg_norm == "global":
                norm_v_condition = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                v_pred = v_pred * scale
            elif cfg_norm == "channel":
                norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                v_pred = v_pred * scale
            else:
                pass
        else:
            pass

        z = z + (t_next - t) * v_pred
        image_prediction = self.unpatchify(
            z,
            self.patch_size * merge_size,
            cur_image_size[1],
            cur_image_size[0],
        )

    generated_images.append(image_prediction)

    clear_flash_kv_cache(past_key_values_cond_cfg)
    clear_flash_kv_cache(past_key_values_tu_cfg)
    clear_flash_kv_cache(past_key_values_iu_cfg)

    return image_prediction
