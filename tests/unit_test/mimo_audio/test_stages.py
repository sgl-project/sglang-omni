# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from sglang_omni.models.mimo_audio.stages import create_thinker_executor


def test_mimo_thinker_initializes_a_zero_sized_multimodal_cache_by_default() -> None:
    signature = inspect.signature(create_thinker_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0
