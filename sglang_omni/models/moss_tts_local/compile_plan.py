# SPDX-License-Identifier: Apache-2.0
"""Compile plan for MOSS-TTS Local."""

from __future__ import annotations

import os
from typing import Any

from sglang_omni.compilation import (
    CompilePhase,
    CompilePlan,
    CompileTarget,
    tensor_dim_bucket,
)
from sglang_omni.models.moss_tts_local.local_transformer import sample_seeded_branchless


def create_moss_tts_local_compile_plan(model: Any) -> CompilePlan:
    def install(compiled: Any) -> None:
        model._compiled_frame_sampler = compiled
        model._sample_seeded_branchless = compiled

    return CompilePlan(
        name="moss_tts_local.frame_sampler",
        targets=(
            CompileTarget(
                name="moss_tts_local.frame_sampler",
                eager=sample_seeded_branchless,
                install=install,
                phase=CompilePhase.AFTER_PRIMARY_CUDA_GRAPH,
                compile_kwargs={
                    "mode": os.environ.get(
                        "SGLANG_TORCH_COMPILE_MODE",
                        "max-autotune-no-cudagraphs",
                    )
                },
                bucket_fn=tensor_dim_bucket("logits"),
            ),
        ),
    )
