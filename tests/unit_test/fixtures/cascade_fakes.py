# SPDX-License-Identifier: Apache-2.0
"""Direct component tests; graph integration uses the real coordinator."""

import asyncio
import functools
import io
from collections.abc import Mapping

import numpy as np
import pytest

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


@functools.cache
def tiny_mp4(frames: int = 3) -> bytes:
    """A 32x32 MP4 of a few grey frames at 24 fps, the clip a fake video service serves; skips the test without PyAV."""
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(frames):
            image = np.full((32, 32, 3), index * 60, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


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
