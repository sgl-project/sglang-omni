# SPDX-License-Identifier: Apache-2.0
"""The motion format the gesture node publishes; its layout needs no model code, so configs and clients import it alone."""

from sglang_omni.proto.streams import PluginPort

MOTION = "motion"
# bytes, float32 little-endian [frames, 271] per chunk: SMPL-X 55-joint axis-angle pose in SMPL-X joint order (165; global_orient 0:3, body_pose 3:66, jaw 66:69, eyes 69:75 always zero, left_hand 75:120, right_hand 120:165), expression coefficients (100), root translation in meters (3), root velocity in meters per second (3)
SMPLX_MOTION = "smplx_motion"
SMPLX_NUM_JOINTS = 55
POSE_FEATURES = 3 * SMPLX_NUM_JOINTS
EXPRESSION_FEATURES = 100
ROOT_FEATURES = 3
FEATURES_PER_FRAME = POSE_FEATURES + EXPRESSION_FEATURES + 2 * ROOT_FEATURES
# The released motion codecs' frame rate (motion_fps of their configs).
MOTION_FPS = 25
SMPLX_MOTION_PORT = PluginPort(MOTION, SMPLX_MOTION, {"fps": MOTION_FPS})

__all__ = [
    "EXPRESSION_FEATURES",
    "FEATURES_PER_FRAME",
    "MOTION",
    "MOTION_FPS",
    "POSE_FEATURES",
    "ROOT_FEATURES",
    "SMPLX_MOTION",
    "SMPLX_MOTION_PORT",
    "SMPLX_NUM_JOINTS",
]
