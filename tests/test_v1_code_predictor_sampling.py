# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker


def test_sample_code_predictor_token_picks_argmax() -> None:
    # logits[:, -1, :] is the slice the function uses; choose unambiguous
    # winners (token 2 for the first row, token 0 for the second). Input is
    # 3D so argmax yields a 1D tensor and the function unsqueezes to (B, 1).
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0]],
            [[2.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    result = Qwen3OmniTalker._sample_code_predictor_token(logits)

    assert result.shape == (2, 1)
    assert result.dtype == torch.long
    assert result[:, 0].tolist() == [2, 0]


def test_sample_code_predictor_token_skips_unsqueeze_when_already_2d() -> None:
    # With a 4D input, logits[:, -1, :] is 3D and argmax returns a 2D tensor;
    # the function must leave it untouched rather than adding a third axis.
    logits = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    result = Qwen3OmniTalker._sample_code_predictor_token(logits)

    assert result.shape == (1, 2)
    assert result.tolist() == [[2, 0]]
