# SPDX-License-Identifier: Apache-2.0
"""History unit scorers: KeyDiff keeps the units whose keys stand out from the mean key."""

import torch
import torch.nn.functional as F

from sglang_omni.scheduling.sglang_backend.unit_scorers import KeyDiffUnitScorer

KV_HEAD_COUNT = 2
HEAD_DIM = 8
NOISE = 0.05


def clustered_keys(
    direction: torch.Tensor, token_count: int, generator: torch.Generator
) -> torch.Tensor:
    noise = torch.randn((token_count, KV_HEAD_COUNT, HEAD_DIM), generator=generator)
    return direction + NOISE * noise


def test_keydiff_keeps_the_unit_that_stands_out_from_the_mean_key() -> None:
    """Three units share one key direction, like silence; the fourth points elsewhere, like speech."""
    generator = torch.Generator().manual_seed(0)
    common, rare = (
        F.normalize(torch.randn((KV_HEAD_COUNT, HEAD_DIM), generator=generator), dim=-1)
        for _ in range(2)
    )
    unit_lengths = [6, 6, 4, 6]
    keys = torch.cat(
        [
            clustered_keys(rare if index == 2 else common, length, generator)
            for index, length in enumerate(unit_lengths)
        ]
    )

    importance = KeyDiffUnitScorer().score(keys, unit_lengths)

    assert importance.shape == (len(unit_lengths),)
    assert int(importance.argmax()) == 2, "the unit far from the mean key is kept"


def test_keydiff_scores_a_unit_by_the_mean_over_its_tokens_and_heads() -> None:
    keys = torch.randn(
        (10, KV_HEAD_COUNT, HEAD_DIM), generator=torch.Generator().manual_seed(1)
    )
    similarity = F.cosine_similarity(keys, keys.mean(dim=0, keepdim=True), dim=-1).mean(
        dim=-1
    )

    importance = KeyDiffUnitScorer().score(keys, [3, 7])

    assert torch.allclose(
        importance, -torch.stack([similarity[:3].mean(), similarity[3:].mean()])
    )
