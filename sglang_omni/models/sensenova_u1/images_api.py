# SPDX-License-Identifier: Apache-2.0
"""SenseNova Images API routes, registered by the model package."""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, TypeAdapter

from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import GenerateChunk, GenerateRequest, UMMGenerateChunk
from sglang_omni.models.sensenova_u1.admission import DEFAULT_UMM_ADMISSION_LIMITS
from sglang_omni.models.sensenova_u1.sampling import (
    GenerationSeed,
    GenerationSteps,
    GuidanceScale,
    ImageCount,
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    validate_seed_range,
)
from sglang_omni.serve.openai_api import (
    image_result_with_disconnect_watch,
    parse_image_size,
    read_image_edit_upload,
)
from sglang_omni.serve.openai_errors import generation_error_status_code

logger = logging.getLogger(__name__)


class ImageGenerationRequest(BaseModel):
    """SenseNova T2I request with native multi-output generation (base64 PNG)."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str = Field(min_length=1)
    size: str | None = None
    n: ImageCount | None = None
    response_format: Literal["b64_json"] = "b64_json"
    seed: GenerationSeed | None = None
    num_inference_steps: GenerationSteps | None = None
    guidance_scale: GuidanceScale | None = None
    enable_cache_dit: StrictBool | None = None
    cache_dit_params: dict[str, JsonValue] | None = None


class ImageGenerationData(BaseModel):
    b64_json: str


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageGenerationData]


def build_image_response(encoded_images: list[str]) -> ImageGenerationResponse:
    return ImageGenerationResponse(
        created=int(time.time()),
        data=[ImageGenerationData(b64_json=image) for image in encoded_images],
    )


def umm_images(chunk: GenerateChunk) -> list[str]:
    """Decode the image segments of a SenseNova UMM result in output order."""
    if not isinstance(chunk, UMMGenerateChunk):
        raise RuntimeError("SenseNova-U1 returned no UMM result")
    else:
        return [
            InlineImage.model_validate(segment.data).encoded_image()
            for segment in chunk.segments or []
            if segment.kind == "image"
        ]


def register_image_generations(app: FastAPI) -> None:
    """Expose SenseNova text-to-image generation as a minimal Images API."""

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    async def image_generations(
        image_request: ImageGenerationRequest, request: Request
    ) -> ImageGenerationResponse:
        if (
            image_request.model is not None
            and image_request.model != app.state.model_name
        ):
            raise HTTPException(status_code=400, detail="Unknown image model")
        else:
            pass
        try:
            params = image_request.model_dump(
                exclude={"model", "prompt", "size", "response_format"},
                exclude_none=True,
            )
            if image_request.size is not None:
                width, height = parse_image_size(image_request.size)
                params.update(width=width, height=height)
            else:
                pass
            params = SenseNovaU1Sampling.from_params(params).model_dump(
                exclude_none=True
            )
            if not image_request.prompt.strip():
                raise ValueError("SenseNova-U1 requires a non-empty text prompt")
            else:
                pass
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generate_request = GenerateRequest(
            model=app.state.model_name,
            prompt=image_request.prompt,
            extra_params=params,
            output_modalities=["image"],
            stream=False,
        )
        request_id = str(uuid.uuid4())
        try:
            chunk = await image_result_with_disconnect_watch(
                request, app.state.client, generate_request, request_id
            )
            return build_image_response(umm_images(chunk))
        except Exception as exc:
            logger.exception(f"Image generation failed for request {request_id}")
            raise HTTPException(
                status_code=generation_error_status_code(exc), detail=str(exc)
            ) from exc


def register_image_edits(app: FastAPI) -> None:
    """Expose SenseNova image editing as a minimal Images API."""

    @app.post("/v1/images/edits", response_model=ImageGenerationResponse)
    async def image_edits(
        request: Request,
        image: list[UploadFile] | None = File(default=None),  # noqa: B008
        image_array: list[UploadFile] | None = File(  # noqa: B008
            default=None, alias="image[]"
        ),
        prompt: str = Form(...),
        mask: UploadFile | None = File(default=None),  # noqa: B008
        model: str | None = Form(default=None),
        n: int = Form(default=1),
        response_format: str | None = Form(default=None),
        size: str | None = Form(default=None),
        seed: int | None = Form(default=None),
        num_inference_steps: int | None = Form(default=None),
        guidance_scale: float | None = Form(default=None),
        img_cfg_scale: float | None = Form(default=None),
        input_max_pixels: int | None = Form(default=None),
        do_resize: str | None = Form(default=None),
        enable_cache_dit: str | None = Form(default=None),
        cache_dit_params: str | None = Form(default=None),
    ) -> ImageGenerationResponse:
        if model is not None and model != app.state.model_name:
            raise HTTPException(status_code=400, detail="Unknown image model")
        else:
            pass
        uploads = image or image_array
        if not uploads:
            raise HTTPException(
                status_code=422,
                detail="Field 'image' is required",
            )
        else:
            pass
        if mask is not None:
            raise HTTPException(
                status_code=400,
                detail="SenseNova-U1 image edits do not support masks yet",
            )
        else:
            pass
        resolved_response_format = (response_format or "b64_json").lower()
        if resolved_response_format != "b64_json":
            raise HTTPException(
                status_code=400,
                detail="SenseNova-U1 image edits only support response_format=b64_json",
            )
        else:
            pass
        try:
            n = TypeAdapter(ImageCount).validate_python(n)
            params = {"n": 1, "size_explicit": size is not None}
            submitted_form = await request.form()
            for name, boolean_text in (
                ("enable_cache_dit", enable_cache_dit),
                ("do_resize", do_resize),
            ):
                if boolean_text is None and name not in submitted_form:
                    continue
                else:
                    pass
                normalized_boolean_text = (boolean_text or "").strip().lower()
                if normalized_boolean_text in ("true", "1", "yes", "on"):
                    params[name] = True
                elif normalized_boolean_text in ("false", "0", "no", "off"):
                    params[name] = False
                else:
                    raise ValueError(f"{name} must be a boolean")
            if cache_dit_params is not None:
                parsed_cache_params = json.loads(cache_dit_params)
                if not isinstance(parsed_cache_params, dict):
                    raise ValueError("cache_dit_params must be a JSON object")
                else:
                    pass
                params["cache_dit_params"] = parsed_cache_params
            else:
                pass
            if size is not None:
                width, height = parse_image_size(size)
                params.update(width=width, height=height)
            else:
                pass
            for name, value in (
                ("num_inference_steps", num_inference_steps),
                ("guidance_scale", guidance_scale),
                ("img_cfg_scale", img_cfg_scale),
                ("seed", seed),
                ("input_max_pixels", input_max_pixels),
            ):
                if value is not None:
                    params[name] = value
                else:
                    pass
            edit_sampling = SenseNovaU1ImageEditSampling.from_params(params)
            validate_seed_range(edit_sampling.seed, n)
            params = edit_sampling.model_dump(exclude_none=True)
            if not prompt.strip():
                raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
            else:
                pass
            limits = DEFAULT_UMM_ADMISSION_LIMITS
            if len(uploads) > limits.max_reference_images:
                raise ValueError(
                    "SenseNova reference count exceeds max_reference_images"
                )
            else:
                pass
            encoded_references: list[str] = []
            encoded_reference_bytes = 0
            for upload in uploads:
                reference_image = await read_image_edit_upload(upload)
                encoded_reference_bytes += 4 * ((len(reference_image) + 2) // 3)
                if encoded_reference_bytes > limits.max_input_bytes:
                    raise ValueError("SenseNova input exceeds max_input_bytes")
                else:
                    encoded_references.append(
                        base64.b64encode(reference_image).decode("ascii")
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # note (Wes): Each output is its own request so it gets a full timeout_s and reserves one output at a time.
        encoded_outputs = []
        parent_request_id = str(uuid.uuid4())
        try:
            for output_index in range(n):
                output_params = dict(params)
                output_params["seed"] = edit_sampling.seed + output_index
                generate_request = GenerateRequest(
                    model=app.state.model_name,
                    prompt={
                        "task": "image_edit",
                        "prompt": prompt,
                        "image_b64": encoded_references,
                    },
                    extra_params=output_params,
                    output_modalities=["image"],
                    stream=False,
                )
                request_id = f"{parent_request_id}:{output_index}"
                chunk = await image_result_with_disconnect_watch(
                    request, app.state.client, generate_request, request_id
                )
                encoded_outputs.extend(umm_images(chunk))
        except Exception as exc:
            logger.exception(f"Image edit failed for request {parent_request_id}")
            raise HTTPException(
                status_code=generation_error_status_code(exc), detail=str(exc)
            ) from exc
        return build_image_response(encoded_outputs)


def register_images_api(app: FastAPI) -> None:
    """Register SenseNova's Images API routes on the initialized app."""
    register_image_generations(app)
    register_image_edits(app)
