# SPDX-License-Identifier: Apache-2.0
"""Decoded image upload formats, byte limits and pixel boundaries."""

import io
from typing import Literal

import pytest
from PIL import Image

from sglang_omni.preprocessing.image import validate_upload_image


@pytest.mark.parametrize(
    "image_format,media_type", [("JPEG", "image/jpeg"), ("PNG", "image/png")]
)
def test_upload_image_accepts_supported_formats_at_the_byte_limit(
    image_format: Literal["JPEG", "PNG"],
    media_type: Literal["image/jpeg", "image/png"],
) -> None:
    """Valid JPEG uploads failed before decoding because they have no frame count."""
    encoded_image = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(encoded_image, format=image_format)
    image_bytes = encoded_image.getvalue()
    assert validate_upload_image(image_bytes, len(image_bytes)) == media_type
    with pytest.raises(ValueError, match="max_image_bytes"):
        validate_upload_image(image_bytes, len(image_bytes) - 1)


@pytest.mark.parametrize("width_pixels", [2048, 2049])
def test_upload_image_enforces_the_decoded_pixel_limit(width_pixels: int) -> None:
    encoded_image = io.BytesIO()
    Image.new("RGB", (width_pixels, 2048), "blue").save(encoded_image, format="PNG")
    if width_pixels == 2048:
        assert validate_upload_image(encoded_image.getvalue()) == "image/png"
    else:
        with pytest.raises(ValueError, match="max_image_pixels"):
            validate_upload_image(encoded_image.getvalue())


def test_upload_image_rejects_animated_png_frames() -> None:
    encoded_image = io.BytesIO()
    first_frame = Image.new("RGB", (16, 16), "red")
    second_frame = Image.new("RGB", (16, 16), "blue")
    first_frame.save(
        encoded_image, format="PNG", save_all=True, append_images=[second_frame]
    )
    with pytest.raises(ValueError, match="animated"):
        validate_upload_image(encoded_image.getvalue())
