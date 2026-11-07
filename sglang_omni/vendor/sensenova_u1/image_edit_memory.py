# SPDX-License-Identifier: Apache-2.0
"""Estimate SenseNova editing attention and cache memory."""

import torch

from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import NEOChatModel


def estimate_it2i_prefix_memory_bytes(
    model: NEOChatModel,
    token_lengths: list[int],
    output_size_pixels: tuple[int, int],
    reference_sizes_pixels: list[tuple[int, int]],
) -> int:
    """Estimate dominant buffers; the caller must budget for remaining workspaces."""
    embedding = model.language_model.get_input_embeddings().weight
    element_bytes = embedding.element_size()
    # note (Wes): The FP32 mask promotes scores, which coexist with softmax output and its dtype conversion.
    attention_element_bytes = (
        max(element_bytes, torch.float32.itemsize)
        + torch.float32.itemsize
        + (element_bytes if embedding.dtype != torch.float32 else 0)
    )
    attention_heads = model.config.llm_config.num_attention_heads
    cache_bytes_per_token = sum(
        (layer.self_attn.k_proj.out_features + layer.self_attn.v_proj.out_features)
        * element_bytes
        for layer in model.language_model.model.layers
    )
    image_patch_size_pixels = model.patch_size * int(1 / model.downsample_ratio)
    reference_pixels = sum(
        width_pixels * height_pixels
        for width_pixels, height_pixels in reference_sizes_pixels
    )
    reference_tokens = reference_pixels // image_patch_size_pixels**2
    vision_embeddings = model.vision_model.embeddings
    # note (Wes): Reference patches, their concatenation and embeddings remain live through all prefix forwards.
    reference_memory_bytes = (
        2
        * 3
        * reference_pixels
        * vision_embeddings.patch_embedding.weight.element_size()
        + reference_tokens
        * vision_embeddings.dense_embedding.out_channels
        * vision_embeddings.dense_embedding.weight.element_size()
    )
    prefix_cache_bytes = 0
    peak_prefix_bytes = 0
    for token_count in token_lengths:
        prefix_cache_bytes += token_count * cache_bytes_per_token
        attention_bytes = token_count**2 * (
            attention_heads * attention_element_bytes + torch.float32.itemsize
        )
        peak_prefix_bytes = max(
            peak_prefix_bytes,
            reference_memory_bytes + prefix_cache_bytes + attention_bytes,
        )
    output_tokens = (output_size_pixels[0] // image_patch_size_pixels) * (
        output_size_pixels[1] // image_patch_size_pixels
    )
    # note (Wes): Expanded flash caches coexist with the original prefix caches.
    expanded_cache_bytes = prefix_cache_bytes + (
        len(token_lengths) * output_tokens * cache_bytes_per_token
    )
    return max(peak_prefix_bytes, prefix_cache_bytes + expanded_cache_bytes)
