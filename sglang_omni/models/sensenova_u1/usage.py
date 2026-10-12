# SPDX-License-Identifier: Apache-2.0
"""SenseNova input-token accounting for image generation and editing."""

from transformers import PreTrainedTokenizerBase

from sglang_omni.client.types import ImageUsageDict


def build_image_usage(
    prompt: str, tokenizer: PreTrainedTokenizerBase, image_tokens: int = 0
) -> ImageUsageDict:
    text_tokens = (
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if callable(tokenizer)
        else None
    )
    usage: ImageUsageDict = {
        "input_tokens_details": {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
        }
    }
    if text_tokens is not None:
        usage.update(input_tokens=text_tokens, total_tokens=text_tokens)
    else:
        pass
    return usage
