# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.components.t2i_generator import extract_prompt_text
from sglang_omni.proto import OmniRequest, StagePayload


def test_extract_prompt_text_string_content() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a cat"},
        {"role": "assistant", "content": "ok"},
    ]
    assert extract_prompt_text(messages) == "a cat"
    assert extract_prompt_text({"messages": messages}) == "a cat"


def test_extract_prompt_text_content_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "cat"},
            ],
        }
    ]
    assert extract_prompt_text(messages) == "a cat"
    assert extract_prompt_text([]) == ""


def test_image_decoder_executor_payload_contract(tmp_path, monkeypatch) -> None:
    from PIL import Image

    import sglang_omni.models.llada2_uni.components.decoder as decoder_pkg

    monkeypatch.setattr(
        decoder_pkg,
        "decode_vq_tokens",
        lambda *args, **kwargs: Image.new("RGB", (4, 4)),
    )

    executor = stages.create_image_decoder_executor(str(tmp_path), device="cpu")
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[]),
        data={"vq_token_ids": [1, 2, 3], "grid_h": 1, "grid_w": 3},
    )
    result = executor._fn(payload)

    assert result.data["modality"] == "image"
    assert result.data["usage"]["completion_tokens"] == 3
    png = base64.b64decode(result.data["images"][0])
    assert png.startswith(b"\x89PNG")
