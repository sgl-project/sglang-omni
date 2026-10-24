"""Native image-edit prefill using prepared reference images."""

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.generation_forward import (
    it2i_prefix_forward,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    build_it2i_inputs,
    build_t2i_image_indexes,
    build_t2i_query,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_denoise import (
    ImagePrefix,
    denoise_images,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.utils import (
    SYSTEM_MESSAGE_FOR_GEN,
    load_image_native,
)
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling


def build_it2i_queries(
    model: NEOChatModel,
    prompt: str,
    reference_sizes: list[tuple[int, int]],
    sampling: SenseNovaU1ImageEditSampling,
) -> list[str]:
    image_placeholders = prompt.count("<image>")
    assert len(reference_sizes) >= image_placeholders
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
        build_t2i_query(
            model,
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
    ]
    if sampling.img_cfg_scale == 1 and sampling.guidance_scale == 1:
        pass
    else:
        if (
            sampling.img_cfg_scale == 1
            or sampling.guidance_scale != sampling.img_cfg_scale
        ):
            queries.append(
                build_t2i_query(
                    model, "<image>" * len(reference_sizes), append_text="<img>"
                )
            )
        else:
            pass
        if sampling.img_cfg_scale != 1:
            queries.append(build_t2i_query(model, "", append_text="<img>"))
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


@torch.no_grad()
def it2i_generate(
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    images: list[Image.Image],
    *,
    sampling: SenseNovaU1ImageEditSampling,
) -> torch.Tensor:
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
        model, prompt, [image.size for image in images], sampling
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
        cache, _ = it2i_prefix_forward(model, input_embeddings, indexes, attention_mask)
        prefixes.append(
            ImagePrefix(
                cache=cache,
                image_indexes=build_t2i_image_indexes(
                    sampling.height // image_patch_size,
                    sampling.width // image_patch_size,
                    indexes[0].max() + 1,
                    device=input_embeddings.device,
                ),
                token_lengths=torch.tensor(
                    [cache.get_seq_length()], dtype=torch.long, device=model.device
                ),
            )
        )
    return denoise_images(
        model, sampling=sampling, prefixes=prefixes, seeds=[sampling.seed]
    )
