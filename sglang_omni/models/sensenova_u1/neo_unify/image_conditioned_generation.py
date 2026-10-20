"""Image conditioned generation."""

from pathlib import Path
from typing import Literal

import torch.utils.checkpoint
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.conditioned_image_denoise import (
    conditioned_image_denoise,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.utils import (
    SYSTEM_MESSAGE_FOR_GEN,
    load_image_native,
)

# Modified for SGLang; see this directory's README.md for upstream source.


@torch.no_grad()
def it2i_generate(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    images: list[Image.Image | str | Path] | None,
    cfg_scale: float = 1,
    img_cfg_scale: float = 1,
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"] = "none",
    enable_timestep_shift: bool = True,
    timestep_shift: float = 1,
    image_size: tuple[int, int] = (256, 256),
    num_steps: int = 30,
    IMG_START_TOKEN: str = "<img>",
    IMG_END_TOKEN: str = "</img>",
    IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
    method: Literal["euler"] = "euler",
    cfg_interval: tuple[float, float] = (0, 1),
    batch_size: int = 1,
    t_eps: float = 0.02,
    think_mode: bool = False,
    seed: int = 0,
    max_think_tokens: int = 1024,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    if getattr(self, "sensenova_compact_mode", False):
        raise RuntimeError(
            "SenseNova compact mode currently supports text-to-image only"
        )
    else:
        pass
    assert cfg_norm in ["none", "global", "channel"]
    self.notify_layer_offload_phase("prefix")

    self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.config.t_eps = t_eps

    image_token_count = prompt.count("<image>")
    assert len(images) >= image_token_count
    if len(images) > image_token_count:
        if image_token_count == 0 and len(images) > 1:
            prompt = (
                "".join(f"Image-{i + 1}:<image>\n" for i in range(len(images))) + prompt
            )
        else:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt
    else:
        pass

    pixel_values = []
    grid_hw = []
    for image in images:
        cur_pixel_values, cur_grid_hw = load_image_native(
            image,
            self.patch_size,
            self.downsample_ratio,
            min_pixels=512 * 512,
            max_pixels=min(2048 * 2048, (4096 * 4096) // len(images)),
            upscale=False,
        )
        cur_grid_hw = cur_grid_hw.to(self.device)
        cur_pixel_values = cur_pixel_values.to(self.device).to(torch.bfloat16)
        pixel_values.append(cur_pixel_values)
        grid_hw.append(cur_grid_hw)
    pixel_values = torch.cat(pixel_values)
    grid_hw = torch.cat(grid_hw)

    merge_size = int(1 / self.downsample_ratio)
    question_condition = f"{prompt}"
    think_text = ""
    self.last_think_token_count = 0
    needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
    needs_img_condition = needs_cfg and (
        img_cfg_scale == 1 or cfg_scale != img_cfg_scale
    )
    needs_uncondition = needs_cfg and img_cfg_scale != 1

    think_content = (
        "<think>\n" if think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    )
    query_condition = self.build_t2i_query(
        question_condition,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text=think_content,
    )
    query_img_condition = (
        self.build_t2i_query("<image>" * len(images), append_text=IMG_START_TOKEN)
        if needs_img_condition
        else None
    )
    query_uncondition = (
        self.build_t2i_query("", append_text=IMG_START_TOKEN)
        if needs_uncondition
        else None
    )

    for i in range(grid_hw.shape[0]):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
        )
        query_condition = query_condition.replace("<image>", image_tokens, 1)
        if query_img_condition is not None:
            query_img_condition = query_img_condition.replace(
                "<image>", image_tokens, 1
            )
        else:
            pass

    input_embeds_condition, indexes_condition, attention_mask_condition_prefix = (
        self.build_it2i_inputs(tokenizer, query_condition, pixel_values, grid_hw)
    )
    if query_img_condition is not None:
        (
            input_embeds_img_condition,
            indexes_img_condition,
            attention_mask_img_condition_prefix,
        ) = self.build_it2i_inputs(
            tokenizer, query_img_condition, pixel_values, grid_hw
        )
    else:
        input_embeds_img_condition = indexes_img_condition = None
    if query_uncondition is not None:
        (
            input_embeds_uncondition,
            indexes_uncondition,
            attention_mask_uncondition_prefix,
        ) = self.build_it2i_inputs(tokenizer, query_uncondition)
    else:
        input_embeds_uncondition = indexes_uncondition = None

    token_h = image_size[1] // (self.patch_size * merge_size)
    token_w = image_size[0] // (self.patch_size * merge_size)

    indexes_image_condition = self.build_t2i_image_indexes(
        token_h,
        token_w,
        indexes_condition[0].max() + 1,
        device=input_embeds_condition.device,
    )
    indexes_image_img_condition = (
        self.build_t2i_image_indexes(
            token_h,
            token_w,
            indexes_img_condition[0].max() + 1,
            device=input_embeds_img_condition.device,
        )
        if indexes_img_condition is not None
        else None
    )
    indexes_image_uncondition = (
        self.build_t2i_image_indexes(
            token_h,
            token_w,
            indexes_uncondition[0].max() + 1,
            device=input_embeds_uncondition.device,
        )
        if indexes_uncondition is not None
        else None
    )

    if think_mode:
        outputs_condition = self.think_prefix_forward(
            inputs_embeds=input_embeds_condition,
            indexes=indexes_condition,
            attention_mask=attention_mask_condition_prefix,
        )
        past_key_values_condition = outputs_condition.past_key_values
        device = outputs_condition.logits.device
        dtype = outputs_condition.logits.dtype
        t_index_condition = indexes_condition[0].max().item()
        past_key_values_condition, t_index_condition, think_text = self.generate_think(
            tokenizer,
            outputs_condition,
            past_key_values_condition,
            t_index_condition,
            IMG_START_TOKEN,
            max_think_tokens=max_think_tokens,
        )
        indexes_image_condition = self.build_t2i_image_indexes(
            token_h,
            token_w,
            t_index_condition + 1,
            device=input_embeds_condition.device,
        )
        del outputs_condition
    else:
        past_key_values_condition, prefix_hidden_states = self.it2i_prefix_forward(
            input_embeds_condition,
            indexes_condition,
            attention_mask_condition_prefix,
        )
        device = prefix_hidden_states.device
        dtype = prefix_hidden_states.dtype
        del prefix_hidden_states
    past_key_values_img_condition = None
    if input_embeds_img_condition is not None:
        past_key_values_img_condition, _ = self.it2i_prefix_forward(
            input_embeds_img_condition,
            indexes_img_condition,
            attention_mask_img_condition_prefix,
        )
    else:
        pass
    past_key_values_uncondition = None
    if input_embeds_uncondition is not None:
        past_key_values_uncondition, _ = self.it2i_prefix_forward(
            input_embeds_uncondition,
            indexes_uncondition,
            attention_mask_uncondition_prefix,
        )
    else:
        pass

    del pixel_values, grid_hw
    del input_embeds_condition, indexes_condition, attention_mask_condition_prefix
    if input_embeds_img_condition is not None:
        del (
            input_embeds_img_condition,
            indexes_img_condition,
            attention_mask_img_condition_prefix,
        )
    else:
        pass
    if input_embeds_uncondition is not None:
        del (
            input_embeds_uncondition,
            indexes_uncondition,
            attention_mask_uncondition_prefix,
        )
    else:
        pass
    self.notify_layer_offload_phase("denoise")

    image_prediction = conditioned_image_denoise(
        batch_size,
        cfg_interval,
        cfg_norm,
        cfg_scale,
        device,
        dtype,
        enable_timestep_shift,
        image_size,
        img_cfg_scale,
        indexes_image_condition,
        indexes_image_img_condition,
        indexes_image_uncondition,
        merge_size,
        num_steps,
        past_key_values_condition,
        past_key_values_img_condition,
        past_key_values_uncondition,
        seed,
        self,
        timestep_shift,
        token_h,
        token_w,
    )

    self.last_think_content = think_text
    if think_mode:
        return image_prediction, think_text
    else:
        pass
    return image_prediction
