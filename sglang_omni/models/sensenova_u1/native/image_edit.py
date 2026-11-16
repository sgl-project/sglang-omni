# SPDX-License-Identifier: Apache-2.0
"""Prepare reference images for native SenseNova editing."""

import math

from PIL import Image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.vision import load_image

from sglang_omni.models.sensenova_u1.native.options import RESOLUTION_ALIGNMENT

MIN_REFERENCE_PIXELS = 512 * 512
DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
TOTAL_REFERENCE_MAX_PIXELS = 4096 * 4096


def smart_resize(
    height_pixels: int,
    width_pixels: int,
    *,
    alignment_pixels: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Fit the reference pixel budget on the model patch grid."""
    if max(height_pixels, width_pixels) / min(height_pixels, width_pixels) > 200:
        raise ValueError("Reference image aspect ratio must not exceed 200")
    else:
        pass
    resized_height_pixels = max(
        alignment_pixels, round(height_pixels / alignment_pixels) * alignment_pixels
    )
    resized_width_pixels = max(
        alignment_pixels, round(width_pixels / alignment_pixels) * alignment_pixels
    )
    if resized_height_pixels * resized_width_pixels > max_pixels:
        resize_scale = math.sqrt((height_pixels * width_pixels) / max_pixels)
        resized_height_pixels = max(
            alignment_pixels,
            math.floor(height_pixels / resize_scale / alignment_pixels)
            * alignment_pixels,
        )
        resized_width_pixels = max(
            alignment_pixels,
            math.floor(width_pixels / resize_scale / alignment_pixels)
            * alignment_pixels,
        )
    elif resized_height_pixels * resized_width_pixels < min_pixels:
        resize_scale = math.sqrt(min_pixels / (height_pixels * width_pixels))
        resized_height_pixels = (
            math.ceil(height_pixels * resize_scale / alignment_pixels)
            * alignment_pixels
        )
        resized_width_pixels = (
            math.ceil(width_pixels * resize_scale / alignment_pixels) * alignment_pixels
        )
    else:
        pass
    return resized_height_pixels, resized_width_pixels


def flatten_reference_image(image: Image.Image) -> Image.Image:
    if "A" in image.getbands() or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    else:
        return image.convert("RGB")


class SenseNovaU1InputStage(InputValidationStage):
    def load_condition_image(self, image: str) -> Image.Image:
        return load_image(image, convert_method=flatten_reference_image)

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width: int,
        condition_image_height: int,
    ) -> None:
        """Override native VAE preprocessing with reference patch preparation."""
        images = batch.condition_image
        if not isinstance(images, list):
            images = [images]
        else:
            pass
        input_max_pixels = min(
            DEFAULT_INPUT_MAX_PIXELS, TOTAL_REFERENCE_MAX_PIXELS // len(images)
        )
        reference_sizes_pixels: list[tuple[int, int]] = []
        total_reference_pixels = 0
        for image in images:
            height_pixels, width_pixels = smart_resize(
                image.height,
                image.width,
                alignment_pixels=RESOLUTION_ALIGNMENT,
                min_pixels=MIN_REFERENCE_PIXELS,
                max_pixels=input_max_pixels,
            )
            reference_sizes_pixels.append((width_pixels, height_pixels))
            total_reference_pixels += width_pixels * height_pixels
        if total_reference_pixels > TOTAL_REFERENCE_MAX_PIXELS:
            raise ValueError(
                f"Prepared reference images contain {total_reference_pixels} pixels, "
                f"exceeding the total limit of {TOTAL_REFERENCE_MAX_PIXELS} pixels"
            )
        else:
            pass
        batch.condition_image = [
            image.resize(reference_size_pixels)
            for image, reference_size_pixels in zip(images, reference_sizes_pixels)
        ]
