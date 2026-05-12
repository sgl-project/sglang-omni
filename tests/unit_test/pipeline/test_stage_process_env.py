# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni_v1.pipeline.stage_process import (
    StageProcessSpec,
    get_stage_process_env,
)


def _tp_spec(*, gpu_id: int) -> StageProcessSpec:
    return StageProcessSpec(
        stage_name="thinker",
        role="leader",
        tp_rank=0,
        tp_size=2,
        gpu_id=gpu_id,
    )


def test_tp_process_env_maps_logical_gpu_through_visible_devices() -> None:
    env = get_stage_process_env(_tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "3,4"})

    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_tp_process_env_rejects_single_visible_device_for_second_gpu() -> None:
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES only exposes"):
        get_stage_process_env(_tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "0"})
