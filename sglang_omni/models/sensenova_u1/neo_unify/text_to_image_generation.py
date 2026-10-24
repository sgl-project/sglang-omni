"""Native text-to-image prefill and multi-output generation."""

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.generation_forward import (
    t2i_prefix_forward,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    build_t2i_image_indexes,
    build_t2i_queries,
    build_t2i_text_inputs,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_denoise import (
    ImagePrefix,
    denoise_images,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling


@torch.no_grad()
def t2i_generate(
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[str],
    *,
    sampling: SenseNovaU1Sampling,
    seeds: list[int] | None = None,
) -> torch.Tensor:
    assert model.concat_time_token_num == 0
    prompts = prompt if isinstance(prompt, list) else [prompt]
    if not prompts:
        raise ValueError("prompt batch must not be empty")
    elif seeds is None:
        seeds = [sampling.seed + index for index in range(sampling.n)]
    else:
        pass
    if isinstance(prompt, list) and len(prompts) != len(seeds):
        raise ValueError("Each batched prompt requires one generation seed")
    else:
        pass
    queries = build_t2i_queries(model, prompts, sampling.guidance_scale)
    image_patch_size = model.patch_size * int(1 / model.downsample_ratio)
    prefixes: list[ImagePrefix] = []
    for branch_queries in queries:
        token_ids, indexes, attention_mask, _, token_lengths = build_t2i_text_inputs(
            model, tokenizer, branch_queries
        )
        cache, _ = t2i_prefix_forward(model, token_ids, indexes, attention_mask)
        prefixes.append(
            ImagePrefix(
                cache=cache,
                image_indexes=build_t2i_image_indexes(
                    sampling.height // image_patch_size,
                    sampling.width // image_patch_size,
                    token_lengths,
                    device=token_ids.device,
                ),
                token_lengths=token_lengths,
            )
        )
    return denoise_images(model, sampling=sampling, prefixes=prefixes, seeds=seeds)
