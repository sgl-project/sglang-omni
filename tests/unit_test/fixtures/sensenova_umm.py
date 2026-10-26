# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for SenseNova controller and serving contracts."""

import base64
import io

import pytest
from PIL import Image

from sglang_omni.models.sensenova_u1.umm import SenseNovaUMMAdapter
from sglang_omni.pipeline.umm import UMMController, UMMLimits


def png_with_bad_idat_crc() -> bytes:
    """A PNG that opens normally but fails verification on its image-data checksum."""
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32), color="red").save(image_buffer, format="PNG")
    png_bytes = bytearray(image_buffer.getvalue())
    chunk_type_offset = png_bytes.index(b"IDAT")
    chunk_data_length = int.from_bytes(
        png_bytes[chunk_type_offset - 4 : chunk_type_offset], "big"
    )
    png_bytes[chunk_type_offset + 4 + chunk_data_length] ^= 0xFF
    return bytes(png_bytes)


@pytest.fixture
def encoded_images() -> list[str]:
    images = []
    for color in ("red", "green", "blue"):
        image_buffer = io.BytesIO()
        Image.new("RGB", (64, 32), color=color).save(image_buffer, format="PNG")
        images.append(base64.b64encode(image_buffer.getvalue()).decode("ascii"))
    return images


@pytest.fixture
def encoded_image(encoded_images: list[str]) -> str:
    return encoded_images[0]


@pytest.fixture
def controller() -> UMMController:
    return UMMController(
        SenseNovaUMMAdapter("unused"),
        limits=UMMLimits(
            max_dispatches=1,
            max_outputs=11,
            max_active=2,
            max_retained_bytes=256 * 1024 * 1024,
            timeout_s=60,
        ),
        phases={"generate": "generate"},
        output_stage="output",
    )
