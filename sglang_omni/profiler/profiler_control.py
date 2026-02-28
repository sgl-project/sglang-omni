# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sglang_omni.pipeline.control_plane import PushSocket
from sglang_omni.proto import ProfilerStartMessage, ProfilerStopMessage

logger = logging.getLogger(__name__)


@dataclass
class ProfilerControlClient:
    """Broadcast profiler control messages to stages (no coordinator object needed)."""

    stage_endpoints: dict[
        str, str
    ]  # stage_name -> stage control_endpoint (PULL bound at stage)

    _socks: dict[str, PushSocket] | None = None

    async def start(self) -> None:
        if self._socks is not None:
            return
        self._socks = {}
        for stage_name, endpoint in self.stage_endpoints.items():
            sock = PushSocket(endpoint)
            await sock.connect()
            self._socks[stage_name] = sock
        logger.info("ProfilerControlClient connected to %d stages", len(self._socks))

    async def close(self) -> None:
        if not self._socks:
            return
        for sock in self._socks.values():
            sock.close()
        self._socks = None

    async def broadcast_start(
        self,
        run_id: str,
        trace_path_template: str,
        config: dict[str, Any] | None = None,
        stages: list[str] | None = None,
    ) -> None:
        await self.start()
        assert self._socks is not None
        targets = stages or list(self._socks.keys())
        msg = ProfilerStartMessage(
            run_id=run_id, trace_path_template=trace_path_template
        )
        for s in targets:
            sock = self._socks.get(s)
            if sock is None:
                continue
            await sock.send(msg)
        logger.info("Broadcast profiler_start run_id=%s to stages=%s", run_id, targets)

    async def broadcast_stop(
        self,
        run_id: str,
        stages: list[str] | None = None,
    ) -> None:
        await self.start()
        assert self._socks is not None
        targets = stages or list(self._socks.keys())
        msg = ProfilerStopMessage(run_id=run_id)
        for s in targets:
            sock = self._socks.get(s)
            if sock is None:
                continue
            await sock.send(msg)
        logger.info("Broadcast profiler_stop run_id=%s to stages=%s", run_id, targets)
