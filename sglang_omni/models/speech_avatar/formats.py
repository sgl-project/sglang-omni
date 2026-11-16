# SPDX-License-Identifier: Apache-2.0
"""The formats the speech-avatar nodes publish; their layouts need no model code, so configs and clients import them alone."""

from sglang_omni.models.avtr.formats import JPEG_FRAME, VIDEO
from sglang_omni.proto.streams import INPUT_TIMELINE, TIMELINE, PluginPort

MOTION = "motion"
# bytes, float32 little-endian [frames, 271] per chunk, MIBURI's layout: SMPL-X 55-joint axis-angle pose in SMPL-X joint order (165; global_orient 0:3, body_pose 3:66, jaw 66:69, eyes 69:75, left_hand 75:120, right_hand 120:165), expression coefficients (100), root translation in meters (3), root velocity in meters per second (3)
SMPLX_MOTION = "smplx_motion"
SMPLX_NUM_JOINTS = 55
POSE_FEATURES = 3 * SMPLX_NUM_JOINTS
EXPRESSION_FEATURES = 100
ROOT_FEATURES = 3
SMPLX_FEATURES_PER_FRAME = POSE_FEATURES + EXPRESSION_FEATURES + 2 * ROOT_FEATURES
# A reply's first picture comes after a chunk of its audio (plus the look-ahead) arrived and the render lead passed; a client holding the audio waits at most this much past the chunk for it.
REPLY_MEDIA_WAIT_SLACK_MS = 600


def reply_media_wait_ms(chunk_ms: float, lookahead_ms: float) -> int:
    """How long a client holding a reply's audio waits for its first picture: a chunk of the reply's audio, the look-ahead and some slack for the render."""
    return round(chunk_ms + lookahead_ms + REPLY_MEDIA_WAIT_SLACK_MS)


def jpeg_frame_port(
    *, fps: int, width: int, height: int, chunk_ms: float, lookahead_ms: float
) -> PluginPort:
    """Each frame is stamped with its 1/fps on the Session input timeline, listening frames (empty payloads, which keep the picture) included, and carries the turn of the reply audio it shows."""
    return PluginPort(
        VIDEO,
        JPEG_FRAME,
        {
            "fps": fps,
            TIMELINE: INPUT_TIMELINE,
            "follows_replies": True,
            "reply_media_wait_ms": reply_media_wait_ms(chunk_ms, lookahead_ms),
            "reply_lookahead_ms": round(chunk_ms + lookahead_ms),
            "chunk_ms": chunk_ms,
            "width": width,
            "height": height,
        },
    )


def smplx_motion_port(*, fps: int, chunk_ms: float, lookahead_ms: float) -> PluginPort:
    """Each chunk is stamped with its block's start on the Session input timeline, listening blocks included, and carries the turn of the reply audio it shows."""
    return PluginPort(
        MOTION,
        SMPLX_MOTION,
        {
            "fps": fps,
            TIMELINE: INPUT_TIMELINE,
            "follows_replies": True,
            "reply_media_wait_ms": reply_media_wait_ms(chunk_ms, lookahead_ms),
            "reply_lookahead_ms": round(chunk_ms + lookahead_ms),
            "chunk_ms": chunk_ms,
            "features": SMPLX_FEATURES_PER_FRAME,
        },
    )


__all__ = [
    "EXPRESSION_FEATURES",
    "JPEG_FRAME",
    "MOTION",
    "POSE_FEATURES",
    "ROOT_FEATURES",
    "SMPLX_FEATURES_PER_FRAME",
    "SMPLX_MOTION",
    "SMPLX_NUM_JOINTS",
    "VIDEO",
    "jpeg_frame_port",
    "reply_media_wait_ms",
    "smplx_motion_port",
]
