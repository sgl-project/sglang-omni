# SPDX-License-Identifier: Apache-2.0
"""Native image API envelopes for Omni image-generation pipelines."""

import base64
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)
from starlette.datastructures import UploadFile

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import ImageGenerationParams

logger = logging.getLogger(__name__)


def build_image_request(
    request: ImageGenerationsRequest, source: str | None = None
) -> GenerateRequest:
    """Translate diffusion sampling names without changing the model's defaults."""
    supplied = request.model_dump(exclude_unset=True, exclude_none=True)
    controls = ImageGenerationParams.model_fields.keys() - {"image_h", "image_w"}
    common_fields = {
        "prompt",
        "model",
        "n",
        "response_format",
        "output_format",
        "size",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "user",
    }
    unsupported = supplied.keys() - common_fields - controls
    if unsupported:
        raise ValueError(
            f"Unsupported image parameters: {', '.join(sorted(unsupported))}"
        )
    else:
        pass
    if request.n != 1:
        raise ValueError("This image pipeline supports n=1")
    else:
        pass
    if request.response_format not in (None, "url", "b64_json"):
        raise ValueError("response_format must be url or b64_json")
    else:
        pass
    if request.output_format not in (None, "png"):
        raise ValueError("This image pipeline supports PNG output")
    else:
        pass
    if isinstance(request.seed, list):
        raise ValueError("This image pipeline requires a single integer seed")
    else:
        pass

    options = {key: value for key, value in supplied.items() if key in controls}
    for native, target in (
        ("num_inference_steps", "decoder_steps"),
        ("guidance_scale", "cfg_scale"),
    ):
        if native in supplied:
            if target in options and options[target] != supplied[native]:
                raise ValueError(f"{native} and {target} disagree")
            else:
                pass
            options[target] = supplied[native]
        else:
            pass

    dimensions = {"size", "width", "height"} & supplied.keys()
    if source is not None:
        if not request.prompt.strip():
            raise ValueError("Image editing requires a non-empty instruction")
        else:
            pass
        if dimensions:
            raise ValueError(
                "Edit output follows the source grid; omit size, width and height"
            )
        else:
            pass
        if options.get("mode") == "thinking":
            raise ValueError("Thinking mode does not support editing")
        else:
            pass
    else:
        if (request.width is None) != (request.height is None):
            raise ValueError("width and height must be provided together")
        else:
            pass
        size = supplied.get("size")
        if size is not None:
            width, height = (int(value) for value in size.split("x"))
            if request.width is not None and (width, height) != (
                request.width,
                request.height,
            ):
                raise ValueError("size and width/height disagree")
            else:
                pass
        else:
            width = request.width if request.width is not None else 1024
            height = request.height if request.height is not None else 1024
        options.update(image_w=width, image_h=height)

    params = ImageGenerationParams.model_validate(options)
    return GenerateRequest(
        model=request.model,
        messages=[Message(role="user", content=request.prompt)],
        output_modalities=["image"],
        stream=False,
        sampling=SamplingParams(seed=request.seed),
        metadata={
            "image_generation": params.model_dump(
                exclude_unset=True, exclude_none=True
            ),
            **({"images": [source]} if source is not None else {}),
        },
    )


def register_images(app: FastAPI) -> None:
    async def generate(
        request: ImageGenerationsRequest, source: str | None = None
    ) -> ImageResponse:
        try:
            generation = build_image_request(request, source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        request_id = str(uuid.uuid4())
        client: Client = app.state.client
        try:
            result = await client.completion(generation, request_id=request_id)
        except Exception as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            else:
                logger.exception(f"Error generating image for request {request_id}")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        if result.image is None:
            raise HTTPException(status_code=500, detail="Pipeline returned no image")
        else:
            pass
        if request.response_format == "b64_json":
            image = ImageResponseData(b64_json=result.image)
        else:
            image = ImageResponseData(url=f"data:image/png;base64,{result.image}")
        return ImageResponse(id=request_id, data=[image])

    @app.post("/v1/images/generations", response_model=ImageResponse)
    async def generations(request: ImageGenerationsRequest) -> ImageResponse:
        return await generate(request)

    edit_schema = ImageGenerationsRequest.model_json_schema()
    edit_schema["properties"].update(
        image={"type": "string", "format": "binary"},
        url={"type": "string", "description": "Alternative to an image upload"},
    )
    edit_schema["properties"].pop("size")
    edit_schema["properties"].pop("width")
    edit_schema["properties"].pop("height")

    @app.post(
        "/v1/images/edits",
        response_model=ImageResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"multipart/form-data": {"schema": edit_schema}},
            }
        },
    )
    async def edits(request: Request) -> ImageResponse:
        async with request.form() as form:
            sources = [
                value
                for key, value in form.multi_items()
                if key in ("image", "image[]", "url", "url[]")
            ]
            if len(sources) != 1:
                raise HTTPException(
                    status_code=400, detail="Edit requires exactly one source image"
                )
            else:
                pass
            fields = {
                key: value
                for key, value in form.items()
                if key not in ("image", "image[]", "url", "url[]")
            }
            try:
                params = ImageGenerationsRequest.model_validate(fields)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            source = sources[0]
            if isinstance(source, UploadFile):
                encoded = base64.b64encode(await source.read()).decode("ascii")
                source = f"data:{source.content_type or 'image/png'};base64,{encoded}"
            else:
                pass
        return await generate(params, source)
