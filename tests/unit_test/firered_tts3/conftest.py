# SPDX-License-Identifier: Apache-2.0
"""Portable model fixtures for CPU tests and explicitly marked CUDA cases."""

from tests.unit_test.fixtures.firered_tts3 import (
    device,
    disable_flow_optimization_environment,
    flow_dispatch_dit,
    tiny_dit,
)
from tests.unit_test.fixtures.firered_tts3_vocoder import decoder

__all__ = [
    "decoder",
    "device",
    "disable_flow_optimization_environment",
    "flow_dispatch_dit",
    "tiny_dit",
]
