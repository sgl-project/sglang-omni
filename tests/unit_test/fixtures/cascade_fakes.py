# SPDX-License-Identifier: Apache-2.0
"""Direct component tests; graph integration uses the real coordinator."""

import asyncio
from collections.abc import Mapping

from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.tools import ToolImage, ToolResult

QUIET = bytes(3200)
# The gate the `wait_for_gate` tool waits at; a test opens it, or replaces it for a fresh event loop.
GATE = asyncio.Event()
# A transparent 1x1 PNG, the picture the `picture` tool makes.
PNG_DOT = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def echo(arguments: Mapping[str, object]) -> dict[str, object]:
    return {"echo": dict(arguments)}


def picture(arguments: Mapping[str, object]) -> ToolResult:
    return ToolResult(
        response={"drew": arguments.get("prompt")},
        images=[ToolImage(mime_type="image/png", data=PNG_DOT)],
    )


async def wait_for_gate(arguments: Mapping[str, object]) -> dict[str, object]:
    await GATE.wait()
    return {"waited": dict(arguments)}


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


async def wait_forever(arguments: Mapping[str, object]) -> dict[str, object]:
    await asyncio.Event().wait()
    return {"never": dict(arguments)}
