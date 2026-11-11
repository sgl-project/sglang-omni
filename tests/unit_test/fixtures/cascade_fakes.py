# SPDX-License-Identifier: Apache-2.0
"""Direct component tests; graph integration uses the real coordinator."""

from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.proto.session import TimedChunk

QUIET = bytes(3200)


async def drain(component, inputs: list) -> list[TimedChunk]:
    """Run the component over the chunks; a coroutine function among them runs at its place."""

    async def stream():
        for chunk in inputs:
            if callable(chunk):
                await chunk()
            else:
                yield chunk

    try:
        return [chunk async for chunk in component.run(stream())]
    finally:
        await component.close()


def build(spec: dict):
    return COMPONENTS.build(COMPONENTS.parse(spec))
