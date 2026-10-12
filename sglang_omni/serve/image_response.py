# SPDX-License-Identifier: Apache-2.0
"""Image response metadata shared by generation and editing endpoints."""

import base64
import io
import time

from PIL import Image
from pydantic import BaseModel

from sglang_omni.client.types import ImageUsageDict


class ImageInputTokensDetails(BaseModel):
    text_tokens: int | None = None
    image_tokens: int | None = None


class ImagePromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class ImageUsage(BaseModel):
    input_tokens: int | None = None
    input_tokens_details: ImageInputTokensDetails | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tokens_details: ImagePromptTokensDetails | None = None
    reasoning_tokens: int | None = 0
    image_count: int | None = None
    width: int | None = None
    height: int | None = None


class ImageGenerationData(BaseModel):
    b64_json: str
    output_format: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageGenerationData]
    usage: ImageUsage


def build_image_response(
    images_b64: list[str], image_usage: ImageUsageDict | None = None
) -> ImageGenerationResponse:
    """Return actual output format, common size, and producer token statistics."""
    images = []
    sizes = []
    for encoded_image in images_b64:
        with Image.open(
            io.BytesIO(base64.b64decode(encoded_image, validate=True))
        ) as image:
            output_format = image.format.lower() if image.format is not None else None
            sizes.append(image.size)
        images.append(
            ImageGenerationData(b64_json=encoded_image, output_format=output_format)
        )
    usage = ImageUsage(**(image_usage or {}), image_count=len(images))
    if sizes and all(size == sizes[0] for size in sizes):
        usage.width, usage.height = sizes[0]
    else:
        pass
    return ImageGenerationResponse(created=int(time.time()), data=images, usage=usage)
