# SPDX-License-Identifier: Apache-2.0
"""Released D2 tensors reach the merged BF16 Qwen3-Omni modules with the release arithmetic."""

import pytest
import torch
from torch import nn

from sglang_omni.models.qwen3_omni_d2.release import (
    DUPLEX_ROW_NAMES,
    merge_release_tensors,
    normalize_release_names,
)


def bf16_linear() -> nn.Linear:
    return nn.Linear(4, 4, dtype=torch.bfloat16).requires_grad_(False)


def tiny_qwen() -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "thinker": nn.ModuleDict(
                {
                    "audio_tower": nn.ModuleDict(
                        {"fc1": bf16_linear(), "q_proj": bf16_linear()}
                    )
                }
            ),
            "talker": nn.ModuleDict(
                {"linear_fc1": bf16_linear(), "shared_expert": bf16_linear()}
            ),
        }
    )


def factors(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, generator=generator), torch.randn(
        4, 2, generator=generator
    )


def merged(base: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    delta = up.bfloat16().float() @ down.bfloat16().float() * 2.0
    return base + delta.bfloat16()


def duplex_rows() -> dict[str, torch.Tensor]:
    return {name: torch.zeros(2, 4) for name in DUPLEX_ROW_NAMES}


def test_d2_file_wins_and_each_adapter_merges_once_from_bf16_factors() -> None:
    model = tiny_qwen()
    fc1 = model.get_parameter("thinker.audio_tower.fc1.weight").clone()
    shared = model.get_parameter("talker.shared_expert.weight").clone()
    stale_down, stale_up = factors(0)
    down, up = factors(1)
    fresh_down, fresh_up = factors(2)
    distilled = torch.randn(4, 4)
    bias = torch.randn(4)
    qwen_tensors, rows = normalize_release_names(
        {
            "audio_tower.fc1.lora_a.weight": stale_down,
            "audio_tower.fc1.lora_b.weight": stale_up,
            "audio_tower.q_proj.weight": distilled,
        },
        {
            "qwen.thinker.audio_tower.audio_tower.fc1.lora_a.weight": down,
            "qwen.thinker.audio_tower.audio_tower.fc1.lora_b.weight": up,
            "qwen.talker.linear_fc1.base.bias": bias,
            "qwen.talker.shared_expert.fresh_lora_a.weight": fresh_down,
            "qwen.talker.shared_expert.fresh_lora_b.weight": fresh_up,
            **duplex_rows(),
        },
    )
    merge_release_tensors(model, qwen_tensors)
    assert set(rows) == DUPLEX_ROW_NAMES
    assert torch.equal(
        model.get_parameter("thinker.audio_tower.fc1.weight"), merged(fc1, down, up)
    )
    assert torch.equal(
        model.get_parameter("talker.shared_expert.weight"),
        merged(shared, fresh_down, fresh_up),
    )
    assert torch.equal(
        model.get_parameter("thinker.audio_tower.q_proj.weight"), distilled.bfloat16()
    )
    assert torch.equal(model.get_parameter("talker.linear_fc1.bias"), bias.bfloat16())


@pytest.mark.parametrize(
    "d2_tensors",
    [
        {"qwen.talker.linear_fc1.lora_a.weight": torch.zeros(2, 4), **duplex_rows()},
        {"qwen.talker.linear_fc1.weight": torch.zeros(4, 2), **duplex_rows()},
        {"thinker.lm_head.weight": torch.zeros(4, 4), **duplex_rows()},
        {name: torch.zeros(2, 4) for name in sorted(DUPLEX_ROW_NAMES)[1:]},
    ],
    ids=[
        "lone-adapter-factor",
        "wrong-shape",
        "unknown-top-level",
        "missing-duplex-row",
    ],
)
def test_incomplete_release_fails_before_serving(
    d2_tensors: dict[str, torch.Tensor]
) -> None:
    with pytest.raises(ValueError):
        qwen_tensors, _ = normalize_release_names({}, d2_tensors)
        merge_release_tensors(tiny_qwen(), qwen_tensors)
