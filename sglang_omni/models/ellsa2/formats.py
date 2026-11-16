# SPDX-License-Identifier: Apache-2.0
"""The motion format the ELLSA 2 node publishes; its layout needs no model code, so configs and clients import it alone."""

from sglang_omni.proto.streams import INPUT_TIMELINE, TIMELINE, PluginPort

MOTION = "motion"
# bytes, float32 little-endian [frames, 264] per chunk: SMPL-H axis-angle pose in SMPL-H joint order (156; global_orient 0:3, body 3:66, left hand 66:111, right hand 111:156), then the FLAME motion code (108; expression 0:100, head rotation 100:103, jaw 103, eye pose 104:108)
SMPLH_FLAME_MOTION = "smplh_flame_motion"
SMPLH_NUM_JOINTS = 52
POSE_FEATURES = 3 * SMPLH_NUM_JOINTS
FLAME_FEATURES = 108
FEATURES_PER_FRAME = POSE_FEATURES + FLAME_FEATURES
# The release's pose_fps and flame_fps.
MOTION_FPS = 30
# A reply's first motion frame comes one 0.8 s block plus its compute (about 0.1 s) after the reply's audio at most; a client holding the audio for it stops waiting a little past that.
REPLY_MEDIA_WAIT_MS = 1000
# Each chunk is stamped with its block's start on the Session input timeline, listening blocks included, and carries the turn of the reply audio it shows.
SMPLH_FLAME_MOTION_PORT = PluginPort(
    MOTION,
    SMPLH_FLAME_MOTION,
    {
        "fps": MOTION_FPS,
        TIMELINE: INPUT_TIMELINE,
        "follows_replies": True,
        "reply_media_wait_ms": REPLY_MEDIA_WAIT_MS,
    },
)

__all__ = [
    "FEATURES_PER_FRAME",
    "FLAME_FEATURES",
    "MOTION",
    "MOTION_FPS",
    "POSE_FEATURES",
    "REPLY_MEDIA_WAIT_MS",
    "SMPLH_FLAME_MOTION",
    "SMPLH_FLAME_MOTION_PORT",
    "SMPLH_NUM_JOINTS",
]
