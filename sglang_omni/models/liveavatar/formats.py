# SPDX-License-Identifier: Apache-2.0
"""The port the Live Avatar node publishes; its layout needs no model code, so configs and clients import it alone."""

from sglang_omni.models.avtr.formats import JPEG_FRAME, VIDEO
from sglang_omni.proto.streams import INPUT_TIMELINE, TIMELINE, PluginPort

# A block of the model: 3 latent frames, 12 pictures at 25 fps.
FPS = 25
CHUNK_FRAMES = 12
CHUNK_MS = CHUNK_FRAMES * 1000 / FPS
# The worker's default speech window around a block (liveavatar_worker.py --context-ms, --lookahead-ms).
CONTEXT_MS = 1000.0
LOOKAHEAD_MS = 240.0
# The worker's picture for the authors' 720*400 area and a 400x720 portrait reference (examples/fashion_blogger.jpg): 384x704.
WIDTH = 384
HEIGHT = 704
# How far past the frame the node reached a reply starts: its first block's speech has to arrive (a chunk plus the look-ahead, as fast as the TTS streams it) and the block has to pass the worker, about 0.6 s on five H100s with four sequence-parallel DiT ranks (liveavatar_worker.py --stages 1 --sp 4 --gpu-jpeg).
RENDER_LEAD_MS = 1600.0
# Silent blocks a Session starts with: the first block of its second clip (the fifth) also fills the attention sink, and with the authors' TPP layout so does its first.
PRIME_CHUNKS = 5
# A client holding a reply's audio waits this much past the lead for its first picture before it lets the audio go.
REPLY_MEDIA_WAIT_SLACK_MS = 600


def jpeg_frame_port(
    *, width: int, height: int, render_lead_ms: float, chunk_ms: float = CHUNK_MS
) -> PluginPort:
    """Each frame is stamped with its 1/25 s on the Session input timeline, listening frames (empty payloads, which keep the picture) included, and carries the turn of the reply audio it shows; a reply's first picture plays `render_lead_ms` after the frame the node had reached when the reply came."""
    return PluginPort(
        VIDEO,
        JPEG_FRAME,
        {
            "fps": FPS,
            TIMELINE: INPUT_TIMELINE,
            "follows_replies": True,
            "reply_media_wait_ms": round(render_lead_ms + REPLY_MEDIA_WAIT_SLACK_MS),
            "reply_lookahead_ms": round(render_lead_ms),
            "chunk_ms": chunk_ms,
            "width": width,
            "height": height,
        },
    )


__all__ = [
    "CHUNK_FRAMES",
    "CHUNK_MS",
    "CONTEXT_MS",
    "FPS",
    "HEIGHT",
    "JPEG_FRAME",
    "LOOKAHEAD_MS",
    "PRIME_CHUNKS",
    "RENDER_LEAD_MS",
    "VIDEO",
    "WIDTH",
    "jpeg_frame_port",
]
