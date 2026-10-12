# SPDX-License-Identifier: Apache-2.0
"""Runtime accelerator probes for ``accelerator``-marked tests.

Probed in the test body, not at collection time, so the marker still assigns
the test to the accelerator CI job (see tests/README.md).
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.platforms import current_platform
from sglang_omni.utils.device import supports_device_streams


def require_cuda(min_devices: int = 1) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible CUDA devices")


def require_device_streams() -> torch.device:
    """This host's accelerator, skipping when it has no streams or graphs."""
    device = torch.device(current_platform.device_type)
    if not supports_device_streams(device):
        pytest.skip(f"{device.type} has no device streams")
    else:
        pass
    try:
        index = torch.get_device_module(device).current_device()
    except RuntimeError as exc:
        pytest.skip(f"{device.type} cannot initialize in this session: {exc}")
    return torch.device(device.type, index)
