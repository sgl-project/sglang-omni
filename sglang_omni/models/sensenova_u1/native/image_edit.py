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

from sglang_omni.models.sensenova_u1.native.options import (
    RESOLUTION_ALIGNMENT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)

DEFAULT_INPUT_MAX_PIXELS = 512 * 512


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


class SenseNovaInputValidationStage(InputValidationStage):
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
        options = batch.extra[SENSENOVA_U1_REQUEST_EXTRA_KEY]
        input_max_pixels = options["input_max_pixels"]
        if input_max_pixels is None:
            input_max_pixels = DEFAULT_INPUT_MAX_PIXELS
        else:
            pass
        references = []
        for image in images:
            if options["do_resize"]:
                height_pixels, width_pixels = smart_resize(
                    image.height,
                    image.width,
                    alignment_pixels=RESOLUTION_ALIGNMENT,
                    min_pixels=input_max_pixels,
                    max_pixels=input_max_pixels,
                )
                if image.size != (width_pixels, height_pixels):
                    image = image.resize(
                        (width_pixels, height_pixels), Image.Resampling.LANCZOS
                    )
                else:
                    pass
            elif (
                image.width % RESOLUTION_ALIGNMENT
                or image.height % RESOLUTION_ALIGNMENT
            ):
                raise ValueError(
                    f"Unresized reference dimensions must be divisible by {RESOLUTION_ALIGNMENT}"
                )
            else:
                pass
            references.append(image)
        batch.condition_image = references
        explicit_fields = batch.extra.get("explicit_fields", ())
        if not {"width", "height", "size"}.intersection(explicit_fields):
            target_pixels = batch.width * batch.height
            batch.height, batch.width = smart_resize(
                references[0].height,
                references[0].width,
                alignment_pixels=RESOLUTION_ALIGNMENT,
                min_pixels=target_pixels,
                max_pixels=target_pixels,
            )
        else:
            pass
