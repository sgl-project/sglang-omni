# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.client.client import Client


def test_result_builder_passes_images_through() -> None:
    chunk = Client._default_result_builder(
        "req-1",
        {"text": "", "modality": "image", "images": ["aGk="], "finish_reason": "stop"},
    )
    assert chunk.images == ["aGk="]
    assert chunk.modality == "image"
