# SPDX-License-Identifier: Apache-2.0
"""Runtime accelerator probes for ``accelerator``-marked tests.

Probed in the test body, not at collection time, so the marker still assigns
the test to the accelerator CI job (see tests/README.md).
"""

from __future__ import annotations

import pytest
import torch


def require_cuda(min_devices: int = 1) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible CUDA devices")
