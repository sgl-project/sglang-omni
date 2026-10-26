# SPDX-License-Identifier: Apache-2.0
"""Image payloads and routes for model-independent serving tests."""

import io

from fastapi import FastAPI
from PIL import Image


def register_test_images(app: FastAPI) -> None:
    model_name = app.state.model_name

    @app.post("/v1/images/generations")
    async def generate() -> dict[str, str]:
        return {"model": model_name}


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
