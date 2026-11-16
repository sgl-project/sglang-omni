# SPDX-License-Identifier: Apache-2.0
"""The video format the AVTR-1 node publishes; its layout needs no model code, so configs and clients import it alone."""

from sglang_omni.proto.streams import INPUT_TIMELINE, TIMELINE, PluginPort

VIDEO = "video"
# bytes, one baseline JPEG per frame at the port's width and height; an empty payload keeps the previous picture
JPEG_FRAME = "jpeg_frame"
# The rate the AVTR-1 checkpoint renders at: five frames per 200 ms motion chunk.
VIDEO_FPS = 25
# 360p in the renderer's 16:9 frame: about 25 KB a frame at quality 75, well inside the Session's output byte budget at 25 fps.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
# A reply's first frame comes after 405 ms of its audio or its end, at most one 200 ms chunk later, plus the render; a client holding the audio for it stops waiting a little past that.
REPLY_MEDIA_WAIT_MS = 1000
# A chunk's first frame hears the reply audio 405 ms past it, the checkpoint's window of the chunk, five more frames and 80 samples; a client aligning a turnless reply stream plays the audio that much (plus its playout delay) behind its arrival.
REPLY_LOOKAHEAD_MS = 405


def jpeg_frame_port(width: int, height: int) -> PluginPort:
    """Each frame is stamped with its 40 ms on the Session input timeline, listening frames included, and carries the turn of the reply audio it shows."""
    return PluginPort(
        VIDEO,
        JPEG_FRAME,
        {
            "fps": VIDEO_FPS,
            TIMELINE: INPUT_TIMELINE,
            "follows_replies": True,
            "reply_media_wait_ms": REPLY_MEDIA_WAIT_MS,
            "reply_lookahead_ms": REPLY_LOOKAHEAD_MS,
            "width": width,
            "height": height,
        },
    )


JPEG_FRAME_PORT = jpeg_frame_port(DEFAULT_WIDTH, DEFAULT_HEIGHT)

__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "JPEG_FRAME",
    "JPEG_FRAME_PORT",
    "REPLY_LOOKAHEAD_MS",
    "REPLY_MEDIA_WAIT_MS",
    "VIDEO",
    "VIDEO_FPS",
    "jpeg_frame_port",
]
