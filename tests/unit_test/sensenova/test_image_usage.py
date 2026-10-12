# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest

from sglang_omni.models.sensenova_u1.usage import build_image_usage


@pytest.mark.parametrize("image_tokens", [0, 1024])
def test_input_usage_excludes_reference_image_tokens_from_total(image_tokens):
    tokenizer = Mock(return_value={"input_ids": [1, 2, 3]})
    usage = build_image_usage("a red mug", tokenizer, image_tokens)
    tokenizer.assert_called_once_with("a red mug", add_special_tokens=False)
    assert usage == {
        "input_tokens": 3,
        "total_tokens": 3,
        "input_tokens_details": {"text_tokens": 3, "image_tokens": image_tokens},
    }


def test_unavailable_text_counts_are_not_fabricated():
    assert build_image_usage("a mug", None, 1024) == {
        "input_tokens_details": {"text_tokens": None, "image_tokens": 1024}
    }


def test_separate_requests_have_independent_token_counts():
    tokenizer = Mock(side_effect=[{"input_ids": [1]}, {"input_ids": [1, 2]}])
    assert build_image_usage("first", tokenizer)["input_tokens"] == 1
    assert build_image_usage("second", tokenizer)["input_tokens"] == 2
