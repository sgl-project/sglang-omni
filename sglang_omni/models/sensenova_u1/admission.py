# SPDX-License-Identifier: Apache-2.0
"""SenseNova request bounds checked before model execution."""

import base64
import io
import json
from dataclasses import dataclass

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from sglang_omni.models.sensenova_u1.image_processing import (
    reference_image_size,
    resolve_edit_output_size,
    validate_image_dimensions,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.proto.request import OmniRequest

ADMISSION_METADATA_KEY = "sensenova_admission_limits"
PNG_BYTES_PER_PIXEL_BOUND = 4
PNG_CONTAINER_BYTES_BOUND = 65536
IMAGE_SEGMENT_METADATA_BYTES_BOUND = 1024


class SenseNovaAdmissionLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_images: int = Field(gt=0, le=10)
    max_output_pixels: int = Field(gt=0)
    max_reference_pixels: int = Field(gt=0)
    max_reference_images: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_input_bytes: int = Field(gt=0)


@dataclass(frozen=True, kw_only=True)
class AdmittedGeneration:
    expected_images: int
    input_bytes: int
    reserved_output_bytes: int


def validate_admission(
    request: OmniRequest, limits: SenseNovaAdmissionLimits
) -> AdmittedGeneration:
    inputs = TypeAdapter(JsonValue).validate_python(request.inputs)
    parameters = TypeAdapter(dict[str, JsonValue]).validate_python(request.params)
    input_bytes = len(
        json.dumps([inputs, parameters], ensure_ascii=False).encode("utf-8")
    )
    if input_bytes > limits.max_input_bytes:
        raise ValueError("SenseNova input exceeds max_input_bytes")
    elif isinstance(inputs, str) and inputs.strip():
        options = SenseNovaU1Sampling.from_params(parameters)
        expected_images = options.n
        output_width, output_height = options.width, options.height
    elif (
        isinstance(inputs, dict)
        and inputs.get("task") == "image_edit"
        and isinstance(inputs.get("prompt"), str)
        and inputs["prompt"].strip()
    ):
        options = SenseNovaU1ImageEditSampling.from_params(parameters)
        expected_images = options.n
        output_width, output_height = options.width, options.height
        references = inputs.get("image_b64")
        if isinstance(references, str):
            references = [references]
        else:
            pass
        if (
            not isinstance(references, list)
            or not 1 <= len(references) <= limits.max_reference_images
        ):
            raise ValueError(
                "SenseNova reference count exceeds max_reference_images or is empty"
            )
        else:
            pass
        if "<IMG_CONTEXT>" in inputs["prompt"]:
            raise ValueError(
                "SenseNova-U1 image edit prompt must not contain <IMG_CONTEXT>"
            )
        elif inputs["prompt"].count("<image>") > len(references):
            raise ValueError(
                "SenseNova-U1 image edit prompt has more <image> placeholders than reference images"
            )
        else:
            pass
        reference_pixels = 0
        resized_reference_pixels = 0
        for reference_index, reference in enumerate(references):
            if not isinstance(reference, str) or not reference:
                raise ValueError("SenseNova references must be non-empty base64 images")
            else:
                image_bytes = base64.b64decode(reference, validate=True)
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Image.DecompressionBombError as error:
                raise ValueError(
                    "SenseNova reference exceeds the image decoder pixel limit"
                ) from error
            with image:
                reference_pixels += image.width * image.height
                if reference_pixels > limits.max_reference_pixels:
                    raise ValueError("SenseNova references exceed max_reference_pixels")
                else:
                    width, height = reference_image_size(
                        image.width, image.height, len(references), options
                    )
                    resized_reference_pixels += width * height
                if reference_index == 0:
                    output_width, output_height = resolve_edit_output_size(
                        (width, height), options
                    )
                else:
                    pass
                if resized_reference_pixels > limits.max_reference_pixels:
                    raise ValueError(
                        "SenseNova resized references exceed max_reference_pixels"
                    )
                else:
                    try:
                        image.verify()
                    # note (Wes): Pillow's PNG verify raises SyntaxError for a corrupt chunk checksum.
                    except SyntaxError as error:
                        raise ValueError(
                            "SenseNova reference is not a valid image"
                        ) from error
    else:
        raise ValueError(
            "SenseNova UMM supports a text prompt or an image_edit request"
        )
    validate_image_dimensions(output_width, output_height)
    if expected_images > limits.max_images:
        raise ValueError("SenseNova output count exceeds max_images")
    elif output_width * output_height * expected_images > limits.max_output_pixels:
        raise ValueError("SenseNova output exceeds max_output_pixels")
    elif options.num_inference_steps > limits.max_steps:
        raise ValueError("SenseNova steps exceed max_steps")
    else:
        pass
    png_bytes = (
        output_width * output_height * PNG_BYTES_PER_PIXEL_BOUND
        + PNG_CONTAINER_BYTES_BOUND
    )
    return AdmittedGeneration(
        expected_images=expected_images,
        input_bytes=input_bytes,
        reserved_output_bytes=expected_images
        * (4 * ((png_bytes + 2) // 3) + IMAGE_SEGMENT_METADATA_BYTES_BOUND),
    )


def validate_output_dimensions(request: OmniRequest, width: int, height: int) -> None:
    validate_image_dimensions(width, height)
    configured_limits = request.metadata.get(ADMISSION_METADATA_KEY)
    if configured_limits is None:
        return
    else:
        limits = SenseNovaAdmissionLimits.model_validate(configured_limits)
    if width * height > limits.max_output_pixels:
        raise ValueError("SenseNova resolved output exceeds max_output_pixels")
    else:
        pass


DEFAULT_UMM_ADMISSION_LIMITS = SenseNovaAdmissionLimits(
    max_images=10,
    max_output_pixels=10 * 2048 * 2048,
    max_reference_pixels=32 * 1024 * 1024,
    max_reference_images=8,
    max_steps=100,
    max_input_bytes=32 * 1024 * 1024,
)
