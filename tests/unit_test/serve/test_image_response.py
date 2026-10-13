# SPDX-License-Identifier: Apache-2.0
import base64
import io

import pytest
from PIL import Image

from sglang_omni.serve.image_response import build_image_response


def encode_image(width: int = 64, height: int = 32, output_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format=output_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("image_tokens", [0, 1024])
def test_image_response_preserves_token_accounting(image_tokens: int) -> None:
    response = build_image_response(
        [encode_image()],
        {
            "input_tokens": 3,
            "total_tokens": 3,
            "input_tokens_details": {"text_tokens": 3, "image_tokens": image_tokens},
        },
    ).model_dump()
    usage = response["usage"]
    assert usage["input_tokens"] == usage["total_tokens"] == 3
    assert usage["input_tokens_details"] == {
        "text_tokens": 3,
        "image_tokens": image_tokens,
    }
    assert (usage["image_count"], usage["width"], usage["height"]) == (1, 64, 32)
    assert response["data"][0]["output_format"] == "png"


def test_image_response_without_token_producer() -> None:
    usage = build_image_response([encode_image()]).usage
    assert usage.input_tokens is None
    assert usage.total_tokens is None
    assert usage.image_count == 1


def test_multiple_outputs_have_actual_formats_and_no_false_common_size() -> None:
    response = build_image_response([encode_image(), encode_image(32, 64, "JPEG")])
    assert [image.output_format for image in response.data] == ["png", "jpeg"]
    assert response.usage.image_count == 2
    assert response.usage.width is None
    assert response.usage.height is None
