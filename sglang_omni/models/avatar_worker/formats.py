# SPDX-License-Identifier: Apache-2.0
"""The formats an avatar_worker node publishes; their layouts need no model code, so configs and clients import them alone."""

from __future__ import annotations

from sglang_omni.models.avtr.formats import JPEG_FRAME, VIDEO
from sglang_omni.proto.streams import INPUT_TIMELINE, TIMELINE, PluginPort

MOTION = "motion"
# bytes, float32 little-endian [1, 3 + 4 * joints + face] per frame: the root joint's position in the skeleton's units, then each joint's rotation relative to its parent as a unit quaternion (w, x, y, z) in the port's joint order, then the face channels the port names; the port's skeleton (joint names, parents, rest offsets) is what a BVH HIERARCHY carries.
BVH_MOTION = "bvh_motion"
ROOT_FEATURES = 3
QUATERNION_FEATURES = 4


def bvh_features(joints: int, face: int) -> int:
    return ROOT_FEATURES + QUATERNION_FEATURES * joints + face


def media_params(
    *, fps: float, chunk_ms: float, lookahead_ms: float, render_ms: float
) -> dict:
    """The input-timeline parameters every port of the node carries.

    A chunk's first frame hears the agent's audio to the chunk's end and the
    look-ahead past it, so a client aligning a turnless reply stream plays
    the audio that much behind its arrival. A reply's first frame comes once
    a window of its audio arrived, at most a chunk later than the reply's
    start and a render after that; a client holding the audio for it stops
    waiting a little past that.
    """
    return {
        "fps": fps,
        TIMELINE: INPUT_TIMELINE,
        "follows_replies": True,
        "reply_media_wait_ms": round(2 * chunk_ms + lookahead_ms + render_ms),
        "reply_lookahead_ms": round(chunk_ms + lookahead_ms),
    }


def jpeg_frame_port(
    *,
    fps: float,
    width: int,
    height: int,
    chunk_ms: float,
    lookahead_ms: float,
    render_ms: float,
) -> PluginPort:
    """Each frame is stamped with its 1/fps on the Session input timeline, listening frames included, and carries the turn of the reply audio it shows; an empty payload keeps the previous picture."""
    return PluginPort(
        VIDEO,
        JPEG_FRAME,
        {
            **media_params(
                fps=fps,
                chunk_ms=chunk_ms,
                lookahead_ms=lookahead_ms,
                render_ms=render_ms,
            ),
            "width": width,
            "height": height,
        },
    )


def bvh_motion_port(
    *,
    fps: float,
    chunk_ms: float,
    lookahead_ms: float,
    render_ms: float,
    joints: list[str],
    parents: list[int],
    offsets: list[list[float]],
    face: list[str],
    up: str = "y",
) -> PluginPort:
    """Each frame is stamped with its 1/fps on the Session input timeline, listening frames included, and carries the turn of the reply audio it shows; an empty payload keeps the previous pose."""
    return PluginPort(
        MOTION,
        BVH_MOTION,
        {
            **media_params(
                fps=fps,
                chunk_ms=chunk_ms,
                lookahead_ms=lookahead_ms,
                render_ms=render_ms,
            ),
            # Tuples, so the port stays hashable; they serialize as JSON arrays.
            "joints": tuple(joints),
            "parents": tuple(parents),
            "offsets": tuple(tuple(offset) for offset in offsets),
            "face": tuple(face),
            "up": up,
            # Rows carry (w, x, y, z); recordings from before this key stored (z, w, x, y) by mistake.
            "quat_layout": "wxyz",
            "features": bvh_features(len(joints), len(face)),
        },
    )


__all__ = [
    "BVH_MOTION",
    "JPEG_FRAME",
    "MOTION",
    "QUATERNION_FEATURES",
    "ROOT_FEATURES",
    "VIDEO",
    "bvh_features",
    "bvh_motion_port",
    "jpeg_frame_port",
    "media_params",
]
