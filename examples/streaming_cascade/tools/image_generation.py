# SPDX-License-Identifier: Apache-2.0
"""A background image-generation tool for the cascade's async_tools node, over public text-to-image APIs.

Run it alone to generate one image with a profile's settings:
python -m examples.streaming_cascade.tools.image_generation --profile qwen3-asr-qwen3.8-27b-image-tool --prompt "a cat"
"""

import argparse
import asyncio
import base64
import io
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import httpx
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.streaming_cascade.tasks import ToolOutcome
from sglang_omni.proto.tools import TaskArtifact

logger = logging.getLogger(__name__)

JPEG_QUALITY = 90
REQUEST_TIMEOUT_S = 60
MODELSCOPE_SUCCEEDED = "SUCCEED"
MODELSCOPE_FAILED = "FAILED"


class ImageBackendOptions(BaseModel):
    """The async_tools options of the generate_image tool."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["modelscope", "siliconflow", "openai"]
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    api_key_env: str
    size: str = "1024x1024"
    poll_interval_s: float = Field(default=1.0, gt=0)
    max_side_px: int = Field(default=1536, gt=0)


async def generate_image(
    arguments: Mapping[str, object], options: Mapping[str, object]
) -> ToolOutcome:
    """Generate the image arguments["prompt"] describes and return it as a JPEG artifact."""
    settings = ImageBackendOptions.model_validate(options)
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generate_image needs a non-empty prompt")
    else:
        pass
    api_key = os.environ.get(settings.api_key_env)
    if not api_key:
        raise RuntimeError(f"the image service key {settings.api_key_env} is not set")
    else:
        pass
    base_url = settings.base_url.rstrip("/")
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_S, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        if settings.provider == "modelscope":
            image_bytes = await modelscope_image(client, base_url, settings, prompt)
        elif settings.provider == "siliconflow":
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": settings.model,
                    "prompt": prompt,
                    "image_size": settings.size,
                },
            )
            raise_for_service(response)
            image_bytes = await download(
                client, str(response.json()["images"][0]["url"])
            )
        else:
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": settings.model,
                    "prompt": prompt,
                    "size": settings.size,
                    "response_format": "b64_json",
                },
            )
            raise_for_service(response)
            image = response.json()["data"][0]
            image_bytes = (
                base64.b64decode(image["b64_json"])
                if image.get("b64_json")
                else await download(client, str(image["url"]))
            )
    jpeg = await asyncio.to_thread(fit_jpeg, image_bytes, settings.max_side_px)
    return ToolOutcome(
        summary=f"an image of {prompt}",
        artifact=TaskArtifact(
            type="image", media_type="image/jpeg", data=base64.b64encode(jpeg).decode()
        ),
    )


async def modelscope_image(
    client: httpx.AsyncClient, base_url: str, settings: ImageBackendOptions, prompt: str
) -> bytes:
    """Submit a ModelScope API-Inference task and poll it; a cancelled task stops polling, and the service finishes it on its own."""
    submitted = await client.post(
        f"{base_url}/v1/images/generations",
        json={"model": settings.model, "prompt": prompt, "size": settings.size},
        headers={"X-ModelScope-Async-Mode": "true"},
    )
    raise_for_service(submitted)
    task_id = submitted.json()["task_id"]
    while True:
        await asyncio.sleep(settings.poll_interval_s)
        polled = await client.get(
            f"{base_url}/v1/tasks/{task_id}",
            headers={"X-ModelScope-Task-Type": "image_generation"},
        )
        raise_for_service(polled)
        state = polled.json()
        if state.get("task_status") == MODELSCOPE_SUCCEEDED:
            return await download(client, str(state["output_images"][0]))
        elif state.get("task_status") == MODELSCOPE_FAILED:
            raise RuntimeError(f"the image service failed the task: {str(state)[:300]}")
        else:
            pass


def raise_for_service(response: httpx.Response) -> None:
    """Fail with the service's own message, which the model reads to tell the user what went wrong."""
    if response.status_code == 429:
        raise RuntimeError("the image service quota or rate limit is exhausted")
    elif response.is_error:
        raise RuntimeError(
            f"the image service answered {response.status_code}: {response.text[:300]}"
        )
    else:
        pass


async def download(client: httpx.AsyncClient, url: str) -> bytes:
    # The result URLs are presigned, so the service's bearer token stays off this request.
    response = await client.get(url, headers={"Authorization": ""})
    raise_for_service(response)
    return response.content


def fit_jpeg(image_bytes: bytes, max_side_px: int) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_side_px, max_side_px))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=JPEG_QUALITY)
    return encoded.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="qwen3-asr-qwen3.8-27b-image-tool")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("generated.jpg"))
    parsed = parser.parse_args()
    profile = yaml.safe_load(
        (
            Path("examples/streaming_cascade/profiles") / parsed.profile / "launch.yaml"
        ).read_text()
    )
    options = profile["session_nodes"]["tool"]["tools"]["generate_image"]["options"]
    started = time.monotonic()
    outcome = asyncio.run(generate_image({"prompt": parsed.prompt}, options))
    assert outcome.artifact is not None
    jpeg = base64.b64decode(outcome.artifact.data)
    parsed.output.write_bytes(jpeg)
    print(
        f"{options['provider']} {options['model']}: {len(jpeg)} bytes in {time.monotonic() - started:.1f} s -> {parsed.output}"
    )


if __name__ == "__main__":
    main()
