# SPDX-License-Identifier: Apache-2.0
"""Send one image question through the streaming Chat Completions adapter."""

import argparse
import asyncio
import json
from pathlib import Path

from sglang_omni.models.streaming_cascade.base import ChatImage, ChatMessage
from sglang_omni.models.streaming_cascade.generators import (
    OpenAIChatGenerator,
    OpenAIChatGeneratorConfig,
)
from sglang_omni.preprocessing.image import (
    MAX_UPLOAD_IMAGE_BYTES,
    validate_upload_image,
)


async def probe(
    image_path: Path,
    question: str,
    base_url: str,
    model: str,
    request_output: Path | None,
) -> None:
    with image_path.open("rb") as image_file:
        image_bytes = image_file.read(MAX_UPLOAD_IMAGE_BYTES + 1)
    image = ChatImage(
        media_type=validate_upload_image(image_bytes), image_bytes=image_bytes
    )
    messages = [ChatMessage(role="user", content=question, images=(image,))]
    generator = OpenAIChatGenerator(
        OpenAIChatGeneratorConfig(
            base_url=base_url,
            model=model,
            supports_images=True,
            temperature=0,
            max_tokens=256,
        )
    )
    try:
        if request_output is not None:
            request_output.write_text(
                json.dumps(generator.request_body(messages), indent=2) + "\n"
            )
        else:
            pass
        async for delta in generator.complete(messages):
            print(delta.content or "", end="", flush=True)
        print()
    finally:
        await generator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--request-output", type=Path)
    arguments = parser.parse_args()
    asyncio.run(
        probe(
            arguments.image,
            arguments.question,
            arguments.base_url,
            arguments.model,
            arguments.request_output,
        )
    )


if __name__ == "__main__":
    main()
