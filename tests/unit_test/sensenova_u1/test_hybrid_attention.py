# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.flow_matching import compare_pil_images
from sglang_omni.models.sensenova_u1.hybrid_attention import (
    build_image_spans,
    build_image_token_tag_from_t_indexes,
    build_m_block_summary,
    build_u1_hybrid_allowed_matrix,
    create_u1_hybrid_mask,
    u1_hybrid_attention_forward,
)


def _naive_allowed(t_indexes: torch.Tensor, image_tag: torch.Tensor) -> torch.Tensor:
    length = t_indexes.numel()
    allowed = torch.zeros((length, length), dtype=torch.bool)
    for row in range(length):
        for col in range(length):
            causal = col <= row
            same_image_span = (
                bool(image_tag[row])
                and bool(image_tag[col])
                and int(t_indexes[row]) == int(t_indexes[col])
            )
            allowed[row, col] = causal or same_image_span
    return allowed


def test_u1_hybrid_allowed_matrix_matches_naive_dense_policy() -> None:
    t_indexes = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4, 5])
    image_tag = torch.tensor([False, False, True, True, True, False, True, True, False])

    allowed = build_u1_hybrid_allowed_matrix(t_indexes, image_token_tag=image_tag)

    assert torch.equal(allowed.cpu(), _naive_allowed(t_indexes, image_tag))
    assert allowed[2, 4]
    assert allowed[4, 2]
    assert not allowed[1, 2]
    assert not allowed[6, 8]


def test_u1_hybrid_mask_and_span_summaries() -> None:
    indexes = torch.tensor(
        [
            [0, 1, 2, 2, 2, 3, 4, 4, 5],
            [0, 0, 0, 1, 2, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    inferred_tag = build_image_token_tag_from_t_indexes(indexes)

    assert inferred_tag.tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert [span.as_dict() for span in build_image_spans(indexes, inferred_tag)] == [
        {"start": 2, "end": 5, "length": 3, "t_index": 2},
        {"start": 6, "end": 8, "length": 2, "t_index": 4},
    ]
    assert build_m_block_summary(inferred_tag, block_m=4) == [
        {"start": 0, "end": 4, "has_image": True, "image_rows": [2, 3]},
        {"start": 4, "end": 8, "has_image": True, "image_rows": [4, 6, 7]},
        {"start": 8, "end": 9, "has_image": False, "image_rows": []},
    ]

    mask = create_u1_hybrid_mask(indexes, image_token_tag=inferred_tag, dtype=torch.float32)
    assert mask.shape == (1, 1, 9, 9)
    assert mask[0, 0, 2, 4].item() == 0.0
    assert mask[0, 0, 1, 2].item() == float("-inf")


def test_u1_hybrid_attention_forward_matches_manual_softmax() -> None:
    query = torch.tensor([[[[0.1, 0.2], [0.3, -0.1], [0.0, 0.5], [0.7, 0.2]]]])
    key = torch.tensor([[[[0.2, 0.0], [0.1, 0.4], [0.6, -0.2], [0.3, 0.3]]]])
    value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [2.0, -1.0]]]])
    t_indexes = torch.tensor([0, 1, 1, 2])
    image_tag = torch.tensor([False, True, True, False])

    out, weights = u1_hybrid_attention_forward(
        query,
        key,
        value,
        t_indexes,
        image_token_tag=image_tag,
    )

    scores = torch.matmul(query, key.transpose(2, 3)) * (query.shape[-1] ** -0.5)
    scores = scores + create_u1_hybrid_mask(
        t_indexes,
        image_token_tag=image_tag,
        dtype=scores.dtype,
    )
    expected_weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    expected_out = torch.matmul(expected_weights, value).transpose(1, 2).contiguous()

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(out, expected_out)


def test_compare_pil_images_reports_exact_and_shifted_metrics() -> None:
    black = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
    almost_black = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8), mode="RGB")

    exact = compare_pil_images(black, black)
    shifted = compare_pil_images(black, almost_black)

    assert exact["pixel_max_abs_diff_uint8"] == 0
    assert math.isinf(exact["psnr_db"])
    assert exact["ssim_global_rgb"] == 1.0
    assert shifted["pixel_max_abs_diff_uint8"] == 1
    assert shifted["psnr_db"] < 50.0
    assert shifted["ssim_global_rgb"] < 1.0
