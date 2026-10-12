# SPDX-License-Identifier: Apache-2.0
from unittest.mock import create_autospec

import pytest
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.usage import build_image_usage


@pytest.mark.parametrize("image_tokens", [0, 1024])
def test_input_usage_excludes_reference_image_tokens_from_total(
    image_tokens: int,
) -> None:
    tokenizer = create_autospec(
        PreTrainedTokenizerBase, instance=True, return_value={"input_ids": [1, 2, 3]}
    )
    usage = build_image_usage("a red mug", tokenizer, image_tokens)
    tokenizer.assert_called_once_with("a red mug", add_special_tokens=False)
    assert usage == {
        "input_tokens": 3,
        "total_tokens": 3,
        "input_tokens_details": {"text_tokens": 3, "image_tokens": image_tokens},
    }


def test_tokenizer_failure_is_not_hidden() -> None:
    tokenizer = create_autospec(
        PreTrainedTokenizerBase,
        instance=True,
        side_effect=RuntimeError("Tokenization failed"),
    )
    with pytest.raises(RuntimeError, match="Tokenization failed"):
        build_image_usage("a mug", tokenizer, 1024)


def test_separate_requests_have_independent_token_counts() -> None:
    tokenizer = create_autospec(
        PreTrainedTokenizerBase,
        instance=True,
        side_effect=[{"input_ids": [1]}, {"input_ids": [1, 2]}],
    )
    assert build_image_usage("first", tokenizer)["input_tokens"] == 1
    assert build_image_usage("second", tokenizer)["input_tokens"] == 2
