# SPDX-License-Identifier: Apache-2.0
"""Decode, resize, and encode SenseNova image requests and results."""

from __future__ import annotations

import base64
import binascii
import io
import math

import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling
from sglang_omni.proto.request import StagePayload

DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512
IMAGE_RESIZE_FACTOR = 32


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
) -> tuple[int, int]:
    """Resize to the target pixel budget and round to the model patch grid."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    else:
        pass
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        resize_scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(
            factor, math.floor(height / resize_scale / factor) * factor
        )
        resized_width = max(factor, math.floor(width / resize_scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        resize_scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * resize_scale / factor) * factor
        resized_width = math.ceil(width * resize_scale / factor) * factor
    else:
        pass
    return resized_height, resized_width


def reference_image_size(
    width: int, height: int, reference_count: int, options: SenseNovaU1ImageEditSampling
) -> tuple[int, int]:
    if not options.do_resize:
        return width, height
    elif options.input_max_pixels is not None:
        input_max_pixels = options.input_max_pixels
    elif reference_count <= 2:
        input_max_pixels = DEFAULT_INPUT_MAX_PIXELS
    else:
        input_max_pixels = max(
            MIN_INPUT_MAX_PIXELS, 2 * DEFAULT_INPUT_MAX_PIXELS // reference_count
        )
    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=IMAGE_RESIZE_FACTOR,
        min_pixels=input_max_pixels,
        max_pixels=input_max_pixels,
    )
    return resized_width, resized_height


def decode_reference_images(value: str | list[str] | None) -> list[Image.Image]:
    values = value if isinstance(value, list) else [value]
    if not values or any(
        not isinstance(encoded_image, str) or not encoded_image
        for encoded_image in values
    ):
        raise ValueError("SenseNova-U1 requires at least one reference image")
    else:
        pass
    images: list[Image.Image] = []
    try:
        for encoded_image in values:
            image_bytes = base64.b64decode(encoded_image, validate=True)
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
            image.load()
    except (
        binascii.Error,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        for image in images:
            image.close()
        raise ValueError("SenseNova-U1 received an invalid reference image") from exc
    return images


def prepare_reference_images(
    value: str | list[str] | None, options: SenseNovaU1ImageEditSampling
) -> list[Image.Image]:
    images = decode_reference_images(value)
    prepared = []
    for image in images:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")
        width, height = reference_image_size(
            image.width, image.height, len(images), options
        )
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            pass
        prepared.append(image)
    return prepared


def resolve_edit_output_size(
    reference_size: tuple[int, int], options: SenseNovaU1ImageEditSampling
) -> tuple[int, int]:
    if options.size_explicit:
        return options.width, options.height
    else:
        pass
    target_pixels = options.width * options.height
    height, width = smart_resize(
        height=reference_size[1],
        width=reference_size[0],
        factor=IMAGE_RESIZE_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return width, height


def resolve_reference_output_size(
    encoded_references: str | list[str] | None, options: SenseNovaU1ImageEditSampling
) -> tuple[int, int]:
    """Validate reference pixels and resolve dimensions without resizing images."""
    references = decode_reference_images(encoded_references)
    try:
        reference_size_pixels = reference_image_size(
            references[0].width, references[0].height, len(references), options
        )
        return resolve_edit_output_size(reference_size_pixels, options)
    finally:
        for image in references:
            image.close()


def encode_image(
    payload: StagePayload, images: torch.Tensor, width: int, height: int
) -> StagePayload:
    if not isinstance(images, torch.Tensor) or images.shape != (1, 3, height, width):
        raise ValueError("SenseNova-U1 returned an invalid image tensor")
    else:
        pass
    image = ((images[0].float() + 1.0) * 127.5).clamp(0, 255)
    image = image.permute(1, 2, 0).byte().cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    payload.data = {
        "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload
