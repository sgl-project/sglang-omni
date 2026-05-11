# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from sglang_omni_v1.config.schema import PipelineConfig


def dummy_factory(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def runtime_factory(
    *,
    model_path: str,
    gpu_id: int,
    thinker_max_seq_len: int | None = None,
    video_fps: float | None = None,
    server_args_overrides: dict[str, Any] | None = None,
    encoder_mem_reserve: float | None = None,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "gpu_id": gpu_id,
        "thinker_max_seq_len": thinker_max_seq_len,
        "video_fps": video_fps,
        "server_args_overrides": server_args_overrides,
        "encoder_mem_reserve": encoder_mem_reserve,
    }


class RejectThinkerPlacementPolicy:
    def validate(self, config: PipelineConfig, plan) -> None:
        if "thinker" in plan.stages:
            raise ValueError("policy rejected thinker")
