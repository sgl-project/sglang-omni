"""Shared Euler denoising for text generation and image editing."""

import logging
import math
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.cancellation import generation_checkpoint
from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    npu_fia_available,
)
from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    clear_flash_kv_cache,
    prepare_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_forward import t2i_predict_v
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    patchify,
    unpatchify,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import randn_with_seed
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.sampling import (
    GENERATION_TIMESTEP_SHIFT,
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.models.sensenova_u1.tensor_parallel import synchronize_tp_noise

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ImagePrefix:
    cache: DynamicCache
    image_indexes: torch.Tensor
    token_lengths: torch.Tensor


def denoise_images(
    model: NEOChatModel,
    *,
    sampling: SenseNovaU1Sampling | SenseNovaU1ImageEditSampling,
    prefixes: list[ImagePrefix],
    seeds: list[int],
) -> torch.Tensor:
    batch_size = len(seeds)
    device = prefixes[0].cache.layers[0].keys.device
    dtype = prefixes[0].cache.layers[0].keys.dtype
    merge_size = int(1 / model.downsample_ratio)
    image_patch_size = model.patch_size * merge_size
    image_token_rows = sampling.height // image_patch_size
    image_token_columns = sampling.width // image_patch_size
    image_token_count = image_token_rows * image_token_columns
    image_size = (sampling.width, sampling.height)
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
            if device.type == "npu" and not npu_fia_available():
                prefix_width = prefix.cache.layers[0].keys.shape[2]
                valid_prefix = (
                    torch.arange(prefix_width, device=device)[None, :]
                    < prefix.token_lengths[:, None]
                )
                valid_keys = torch.cat(
                    (
                        valid_prefix.expand(batch_size, -1),
                        torch.ones(
                            (batch_size, image_token_count),
                            dtype=torch.bool,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
                attention_masks.append(
                    {
                        "full_attention": valid_keys[:, None, None, :]
                        .expand(-1, -1, image_token_count, -1)
                        .contiguous()
                    }
                )
            else:
                attention_masks.append({"full_attention": None})
        logger.info(
            f"SenseNova-U1 attention caches: layouts={[prefix.cache.layers[0].flash_cache_layout for prefix in prefixes]}"
        )

        grid_rows = sampling.height // model.patch_size
        grid_columns = sampling.width // model.patch_size
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
        image_prediction = noise_scale * randn_with_seed(
            (batch_size, 3, sampling.height, sampling.width),
            device=device,
            dtype=dtype,
            seed=seeds,
        )
        synchronize_tp_noise(model, image_prediction)
        timesteps = torch.linspace(
            0.0, 1.0, sampling.num_inference_steps + 1, device=device
        )
        noise_levels = 1 - timesteps
        timesteps = 1 - GENERATION_TIMESTEP_SHIFT * noise_levels / (
            1 + (GENERATION_TIMESTEP_SHIFT - 1) * noise_levels
        )
        # note (Wes): Every image token shares the step's timestep, so one embedding per step is broadcast.
        denoise_embeddings = model.fm_modules["timestep_embedder"](timesteps[:-1])
        if model.add_noise_scale_embedding:
            denoise_embeddings = denoise_embeddings + model.fm_modules[
                "noise_scale_embedder"
            ](timesteps.new_tensor([noise_scale / model.noise_scale_max_value]))
        else:
            pass

        for step_index in range(sampling.num_inference_steps):
            generation_checkpoint()
            timestep = timesteps[step_index]
            image_latents = patchify(image_prediction, image_patch_size)
            image_patches = patchify(
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
                t2i_predict_v(
                    model,
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
                predicted_velocity = unconditioned + sampling.guidance_scale * (
                    conditioned - unconditioned
                )
            else:
                assert isinstance(sampling, SenseNovaU1ImageEditSampling)
                conditioned, image_conditioned, unconditioned = predicted_velocities
                predicted_velocity = (
                    unconditioned
                    + sampling.guidance_scale * (conditioned - image_conditioned)
                    + sampling.img_cfg_scale * (image_conditioned - unconditioned)
                )
            image_latents = (
                image_latents
                + (timesteps[step_index + 1] - timestep) * predicted_velocity
            )
            image_prediction = unpatchify(
                image_latents, image_patch_size, sampling.height, sampling.width
            )
        return image_prediction
    finally:
        for prefix in prefixes:
            clear_flash_kv_cache(prefix.cache)
