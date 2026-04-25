# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni visual encoder batch budgeting."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime.encoder import EncoderRequestData
from sglang_omni.engines.omni.types import SchedulerRequest
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder

# V0 guardrail for image-encoder admission. This is deliberately explicit,
# not derived from transient free GPU memory during multiprocess startup.
QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES = 10 * 1024**3
# Covers temporary visual-forward activations beyond input/output tensors.
QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5


def create_qwen3_visual_request_cost_fn(model: Qwen3OmniImageEncoder):
    merge = int(model.spatial_merge_size) ** 2
    hidden = int(model.out_hidden_size)
    output_layers = 1 + int(model.deepstack_layers)
    dtype_bytes = int(model.visual_dtype_bytes)

    def _cost(request: SchedulerRequest) -> int:
        assert isinstance(request.data, EncoderRequestData)
        input_dict = request.data.input_dict
        if input_dict is None or input_dict.get("_skip"):
            return 0

        raw_bytes = _tensor_bytes(input_dict.get("pixel_values"))
        raw_bytes += _tensor_bytes(input_dict.get("pixel_values_videos"))

        visual_tokens = _grid_visual_tokens(input_dict.get("image_grid_thw"), merge)
        visual_tokens += _grid_visual_tokens(input_dict.get("video_grid_thw"), merge)
        output_bytes = visual_tokens * hidden * dtype_bytes * output_layers
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER

    return _cost


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    assert grid.device.type == "cpu", "visual batch cost must run before GPU staging"
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())
