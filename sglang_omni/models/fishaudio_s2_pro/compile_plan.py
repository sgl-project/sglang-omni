# SPDX-License-Identifier: Apache-2.0
"""Compile plan for FishAudio S2-Pro."""

from __future__ import annotations

from typing import Any

from sglang_omni.compilation import (
    CompilePlan,
    build_module_list_compile_plan,
    tensor_dim_bucket,
)


def create_fish_s2pro_compile_plan(model: Any) -> CompilePlan:
    audio_decoder = model._audio_decoder
    return build_module_list_compile_plan(
        "fishaudio_s2_pro.codebook_decoder",
        [layer.forward_kvcached for layer in audio_decoder.layers],
        install=lambda layers: audio_decoder.set_compiled_forward_kvcached_layers(
            layers,
            max_batch_size=audio_decoder.kv_cache_max_batch_size,
        ),
        bucket_fn=tensor_dim_bucket("x"),
    )
