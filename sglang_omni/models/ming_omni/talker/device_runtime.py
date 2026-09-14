# SPDX-License-Identifier: Apache-2.0
"""Device runtime boundary for Ming-Omni talker worker threads."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class TalkerDeviceRuntime:
    """Provide device stream creation, contexts, and synchronization."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.device_module = (
            None if self.device.type == "cpu" else torch.get_device_module(self.device)
        )

    def create_stream(self):
        if self.device_module is None:
            return None
        return self.device_module.Stream(device=self.device)

    def create_stream_context(self, stream):
        if self.device_module is None:
            return nullcontext()
        return self.device_module.stream(stream)

    def synchronize(self) -> None:
        if self.device_module is None:
            return
        self.device_module.current_stream(self.device).synchronize()
