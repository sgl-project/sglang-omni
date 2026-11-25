# SPDX-License-Identifier: Apache-2.0
"""Check the Mandarin policy against the Thinker's greedy output and duplex controls."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from sglang_omni.models.qwen3_omni_d2.mandarin import MandarinOpening


@pytest.fixture
def opening() -> MandarinOpening:
    pieces = [
        "<pad>",
        "<response>",
        "<interrupt>",
        "2",
        "，",
        "Sure",
        "二",
        " 可以",
        "特殊",
        "中A",
        "�",
    ]
    tokenizer = Tokenizer(WordLevel({piece: i for i, piece in enumerate(pieces)}))
    return MandarinOpening(tokenizer, frozenset({0, 1, 2, 8}), 1, torch.device("cpu"))


def test_mandarin_first_word_after_response(opening: MandarinOpening) -> None:
    """An English opener and then a one-character substitution both produced spoken English."""
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 9.0, 5.0, 8.0, 7.0, 0.0]])
    for history in ([1], [1, 3, 4, 10]):
        assert opening.mask(logits, history).argmax().item() == 7
    for history in ([], [1, 7], [1, 5], [1, 0], [1, 2]):
        assert opening.mask(logits, history) is logits
    assert opening.mask(logits, [1, 7, 2, 1]).argmax().item() == 7
    logits[0, 7] = -torch.inf
    assert opening.mask(logits, [1]) is logits


@pytest.mark.parametrize("native_token", [0, 1, 2, 3, 4, 6, 7, 9, 10])
def test_native_control_number_and_non_latin_winners_pass_through(
    opening: MandarinOpening, native_token: int
) -> None:
    logits = torch.zeros((1, 11))
    logits[0, native_token] = 1
    assert opening.mask(logits, [1]) is logits
