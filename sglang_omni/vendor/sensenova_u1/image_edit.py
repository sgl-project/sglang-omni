# SPDX-License-Identifier: Apache-2.0
"""Reference conditioning and Euler denoising for SenseNova image editing."""

import math
from dataclasses import dataclass

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import (
    NEOChatModel,
    _randn_with_seed,
    clear_flash_kv_cache,
    prepare_flash_kv_cache,
)
from sglang_omni.vendor.sensenova_u1.modeling_neo_vit import (
    build_abs_positions_from_grid_hw,
)
from sglang_omni.vendor.sensenova_u1.modeling_qwen3 import create_block_causal_mask
from sglang_omni.vendor.sensenova_u1.utils import SYSTEM_MESSAGE_FOR_GEN

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(kw_only=True)
class ImagePrefix:
    cache: DynamicCache
    image_indexes: torch.Tensor
    token_lengths: torch.Tensor


def load_image_native(
    image: Image.Image,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and pack a reference already resized by the input stage."""
    image_factor = int(patch_size / downsample_ratio)
    assert image.width % image_factor == 0 and image.height % image_factor == 0
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    pixel_values = transform(image).to(torch.float32)
    channel_count, height_pixels, width_pixels = pixel_values.shape
    grid_rows = height_pixels // patch_size
    grid_columns = width_pixels // patch_size
    flattened_pixels = (
        pixel_values.view(
            channel_count, grid_rows, patch_size, grid_columns, patch_size
        )
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_rows * grid_columns, channel_count * patch_size**2)
    )
    grid_shape = torch.tensor([[grid_rows, grid_columns]], device=pixel_values.device)
    return flattened_pixels, grid_shape


def build_it2i_queries(
    model: NEOChatModel,
    prompt: str,
    reference_sizes: list[tuple[int, int]],
    cfg_scale: float,
    img_cfg_scale: float,
) -> list[str]:
    image_placeholders = prompt.count("<image>")
    if len(reference_sizes) < image_placeholders:
        raise ValueError("Prompt has more image placeholders than reference images")
    else:
        pass
    if len(reference_sizes) > image_placeholders:
        if image_placeholders == 0 and len(reference_sizes) > 1:
            prompt = (
                "".join(
                    f"Image-{index + 1}:<image>\n"
                    for index in range(len(reference_sizes))
                )
                + prompt
            )
        else:
            prompt = "<image>\n" * (len(reference_sizes) - image_placeholders) + prompt
    else:
        pass
    queries = [
        model._build_t2i_query(
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
    ]
    if img_cfg_scale == 1 and cfg_scale == 1:
        pass
    else:
        if img_cfg_scale == 1 or cfg_scale != img_cfg_scale:
            queries.append(
                model._build_t2i_query(
                    "<image>" * len(reference_sizes), append_text="<img>"
                )
            )
        else:
            pass
        if img_cfg_scale != 1:
            queries.append(model._build_t2i_query("", append_text="<img>"))
        else:
            pass
    for width_pixels, height_pixels in reference_sizes:
        image_token_count = int(
            (height_pixels // model.patch_size)
            * (width_pixels // model.patch_size)
            * model.downsample_ratio**2
        )
        image_tokens = "<img>" + "<IMG_CONTEXT>" * image_token_count + "</img>"
        queries = [query.replace("<image>", image_tokens, 1) for query in queries]
    return queries


def build_it2i_inputs(
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    pixel_values: torch.Tensor | None = None,
    grid_hw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)

    if input_ids.shape[1] > min(
        tokenizer.model_max_length, model.config.llm_config.max_position_embeddings - 1
    ):
        raise ValueError("SenseNova editing prefix exceeds the model context length")
    else:
        pass
    indexes = get_thw_indexes(model, input_ids[0], grid_hw)

    attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}

    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    if pixel_values is not None:
        vit_embeds = model.extract_feature(
            pixel_values, grid_hw=grid_hw, gen_model=False
        )
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = input_ids == model.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)
    else:
        pass

    return input_embeds, indexes, attention_mask


def get_thw_indexes(
    model: NEOChatModel, input_ids: torch.Tensor, grid_hw: torch.Tensor | None = None
) -> torch.Tensor:
    img_start_shift = torch.cat(
        [
            torch.zeros(1, dtype=torch.long).to(input_ids.device),
            (input_ids == model.img_start_token_id).long(),
        ],
        dim=0,
    )[:-1]
    not_img_token = (input_ids != model.img_context_token_id).long()
    t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
    h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
    w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

    if grid_hw is not None:
        selected = input_ids == model.img_context_token_id
        if selected.long().sum() > 0:
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                grid_hw // int(1 / model.downsample_ratio), device=t_indexes.device
            )
            h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        else:
            pass
    else:
        pass
    return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


@torch.no_grad()
def it2i_generate(
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    images: list[Image.Image],
    *,
    image_size: tuple[int, int],
    num_steps: int,
    cfg_scale: float,
    img_cfg_scale: float,
    seed: int,
    timestep_shift: float,
    t_eps: float,
) -> torch.Tensor:
    model.config.t_eps = t_eps
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    reference_patches: list[torch.Tensor] = []
    reference_grids: list[torch.Tensor] = []
    for image in images:
        image_patches, image_grid = load_image_native(
            image, model.patch_size, model.downsample_ratio
        )
        reference_patches.append(
            image_patches.to(
                device=model.device, dtype=next(model.vision_model.parameters()).dtype
            )
        )
        reference_grids.append(image_grid.to(model.device))
    pixel_values = torch.cat(reference_patches)
    grid_shape = torch.cat(reference_grids)
    queries = build_it2i_queries(
        model, prompt, [image.size for image in images], cfg_scale, img_cfg_scale
    )
    image_patch_size = model.patch_size * int(1 / model.downsample_ratio)
    prefixes: list[ImagePrefix] = []
    for query in queries:
        has_references = "<IMG_CONTEXT>" in query
        input_embeddings, indexes, attention_mask = build_it2i_inputs(
            model,
            tokenizer,
            query,
            pixel_values if has_references else None,
            grid_shape if has_references else None,
        )
        prefix_output = model.language_model.model(
            inputs_embeds=input_embeddings,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        cache = prefix_output.past_key_values
        prefixes.append(
            ImagePrefix(
                cache=cache,
                image_indexes=model._build_t2i_image_indexes(
                    image_size[1] // image_patch_size,
                    image_size[0] // image_patch_size,
                    indexes[0].max() + 1,
                    device=input_embeddings.device,
                ),
                token_lengths=torch.tensor(
                    [cache.get_seq_length()], dtype=torch.long, device=model.device
                ),
            )
        )
    batch_size = 1
    device = prefixes[0].cache.layers[0].keys.device
    dtype = prefixes[0].cache.layers[0].keys.dtype
    merge_size = int(1 / model.downsample_ratio)
    image_patch_size = model.patch_size * merge_size
    image_token_rows = image_size[1] // image_patch_size
    image_token_columns = image_size[0] // image_patch_size
    image_token_count = image_token_rows * image_token_columns
    attention_masks: list[dict[str, torch.Tensor | None]] = []
    try:
        for prefix in prefixes:
            for layer in prefix.cache.layers:
                layer.keys = layer.keys.expand(batch_size, *layer.keys.shape[1:])
                layer.values = layer.values.expand(batch_size, *layer.values.shape[1:])
            prepare_flash_kv_cache(
                prefix.cache,
                current_len=image_token_count,
                batch_size=batch_size,
                prefix_lengths=prefix.token_lengths,
            )
            attention_masks.append({"full_attention": None})

        grid_rows = image_size[1] // model.patch_size
        grid_columns = image_size[0] // model.patch_size
        grid_shape = torch.tensor(
            [[grid_rows, grid_columns]] * batch_size, device=device
        )
        noise_scale = model.noise_scale
        if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            resolution_scale = math.sqrt(
                image_token_count / float(model.noise_scale_base_image_seq_len)
            )
            noise_scale *= resolution_scale
            if model.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
            else:
                pass
        else:
            pass
        noise_scale = min(noise_scale, model.noise_scale_max_value)
        image_prediction = noise_scale * _randn_with_seed(
            (batch_size, 3, image_size[1], image_size[0]),
            device=device,
            dtype=dtype,
            seed=[seed],
        )
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        noise_levels = 1 - timesteps
        timesteps = 1 - timestep_shift * noise_levels / (
            1 + (timestep_shift - 1) * noise_levels
        )
        # note (Wes): Every image token shares the step's timestep, so one embedding per step is broadcast.
        denoise_embeddings = model.fm_modules["timestep_embedder"](timesteps[:-1])
        if model.add_noise_scale_embedding:
            denoise_embeddings = denoise_embeddings + model.fm_modules[
                "noise_scale_embedder"
            ](timesteps.new_tensor([noise_scale / model.noise_scale_max_value]))
        else:
            pass

        for step_index in range(num_steps):
            timestep = timesteps[step_index]
            image_latents = model.patchify(image_prediction, image_patch_size)
            image_patches = model.patchify(
                image_prediction, model.patch_size, channel_first=True
            )
            image_embeddings = model.extract_feature(
                image_patches.view(batch_size * grid_rows * grid_columns, -1),
                gen_model=True,
                grid_hw=grid_shape,
            ).view(batch_size, image_token_count, -1)
            image_embeddings = image_embeddings + denoise_embeddings[step_index].view(
                1, 1, -1
            )
            predicted_velocities = [
                model._t2i_predict_v(
                    image_embeddings,
                    prefix.image_indexes,
                    attention_mask,
                    prefix.cache,
                    timestep,
                    image_latents,
                    image_token_num=image_token_count,
                    image_size=image_size,
                )
                for prefix, attention_mask in zip(prefixes, attention_masks)
            ]
            if len(predicted_velocities) == 1:
                predicted_velocity = predicted_velocities[0]
            elif len(predicted_velocities) == 2:
                conditioned, unconditioned = predicted_velocities
                predicted_velocity = unconditioned + cfg_scale * (
                    conditioned - unconditioned
                )
            else:
                conditioned, image_conditioned, unconditioned = predicted_velocities
                predicted_velocity = (
                    unconditioned
                    + cfg_scale * (conditioned - image_conditioned)
                    + img_cfg_scale * (image_conditioned - unconditioned)
                )
            image_latents = (
                image_latents
                + (timesteps[step_index + 1] - timestep) * predicted_velocity
            )
            image_prediction = model.unpatchify(
                image_latents, image_patch_size, image_size[1], image_size[0]
            )
        return image_prediction
    finally:
        for prefix in prefixes:
            clear_flash_kv_cache(prefix.cache)
