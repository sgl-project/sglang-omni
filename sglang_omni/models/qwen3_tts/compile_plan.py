# SPDX-License-Identifier: Apache-2.0
"""Compile plan for the Qwen3-TTS decode backbone."""

from __future__ import annotations

from typing import Any

from sglang_omni.compilation import (
    CompilePlan,
    build_module_list_compile_plan,
    tensor_dim_bucket,
)


def create_qwen3_tts_compile_plan(model: Any) -> CompilePlan:
    text_model = model.model
    return build_module_list_compile_plan(
        "qwen3_tts.decode_backbone",
        text_model.layers,
        install=lambda layers: setattr(
            text_model,
            "_compiled_decode_layers",
            layers,
        ),
        bucket_fn=tensor_dim_bucket("hidden_states"),
    )
