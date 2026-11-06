# SPDX-License-Identifier: Apache-2.0
"""Deterministic clips of solid shapes on white, for tests that need known video content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np
import numpy.typing as npt

FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
WHITE_RGB = (255, 255, 255)


@dataclass(frozen=True, kw_only=True)
class Shape:
    kind: Literal["square", "circle"]
    rgb: tuple[int, int, int]
    center_x_pixels: int
    center_y_pixels: int
    half_size_pixels: int


@dataclass(frozen=True, kw_only=True)
class Scene:
    duration_s: float
    shapes: tuple[Shape, ...]


def render_frame(shapes: tuple[Shape, ...]) -> npt.NDArray[np.uint8]:
    frame = np.empty((FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, 3), dtype=np.uint8)
    frame[:] = WHITE_RGB
    rows, columns = np.mgrid[0:FRAME_HEIGHT_PIXELS, 0:FRAME_WIDTH_PIXELS]
    for shape in shapes:
        offset_y = rows - shape.center_y_pixels
        offset_x = columns - shape.center_x_pixels
        if shape.kind == "circle":
            inside = offset_x**2 + offset_y**2 <= shape.half_size_pixels**2
        else:
            inside = (np.abs(offset_x) <= shape.half_size_pixels) & (
                np.abs(offset_y) <= shape.half_size_pixels
            )
        frame[inside] = shape.rgb
    return frame


def write_scene_video(
    path: Path, scenes: list[Scene], *, frames_per_second: int = 10
) -> Path:
    """Encode the scenes back to back as an H.264 MP4 and return its path."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=frames_per_second)
        stream.width = FRAME_WIDTH_PIXELS
        stream.height = FRAME_HEIGHT_PIXELS
        stream.pix_fmt = "yuv420p"
        for scene in scenes:
            frame_rgb = render_frame(scene.shapes)
            for _ in range(round(scene.duration_s * frames_per_second)):
                frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path
