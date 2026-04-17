# SPDX-License-Identifier: Apache-2.0
"""StageGroup — manages the OS processes backing one logical pipeline stage.

For stages with ``tp_size == 1`` (the common case) the group contains a
single process.  For AR stages that require tensor parallelism the group
spawns *tp_size* processes, each with a different ``tp_rank``.

The Coordinator only talks to **rank 0** (the *leader*).  Within a TP
group, all ranks receive the same requests via a leader-broadcast pattern
and make identical scheduling decisions so that NCCL collectives inside the
model forward stay in lockstep.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
from typing import Sequence

from sglang_omni.pipeline.stage_process import StageProcessSpec, stage_process_main

logger = logging.getLogger(__name__)


class StageGroup:
    """Lifecycle manager for all processes of one logical pipeline stage."""

    def __init__(self, stage_name: str, specs: Sequence[StageProcessSpec]):
        if not specs:
            raise ValueError(
                f"StageGroup requires at least one spec (stage={stage_name})"
            )
        self.stage_name = stage_name
        self.specs = list(specs)
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tp_size(self) -> int:
        return len(self.specs)

    @property
    def leader_endpoint(self) -> str:
        """Control-plane recv endpoint for tp_rank 0 (used by Coordinator)."""
        return self.specs[0].recv_endpoint

    @property
    def processes(self) -> list[multiprocessing.Process]:
        return list(self._processes)

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn one OS process per TP rank."""
        for spec in self.specs:
            event = ctx.Event()
            tp_suffix = f"-tp{spec.tp_rank}" if spec.tp_size > 1 else ""
            proc = ctx.Process(
                target=stage_process_main,
                args=(spec, event),
                name=f"stage-{spec.stage_name}{tp_suffix}",
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)
            self._ready_events.append(event)

        logger.info(
            "StageGroup %s: spawned %d process(es) (pids=%s)",
            self.stage_name,
            len(self._processes),
            [p.pid for p in self._processes],
        )

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    async def wait_ready(self, timeout: float) -> None:
        """Block until every TP rank signals ready or *timeout* expires."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        for i, event in enumerate(self._ready_events):
            proc = self._processes[i]
            tp_label = f"{self.stage_name}:tp{self.specs[i].tp_rank}"

            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Stage {tp_label} did not become ready "
                        f"within {timeout:.0f}s"
                    )
                if not proc.is_alive():
                    raise RuntimeError(
                        f"Stage {tp_label} process died during startup "
                        f"(exit code {proc.exitcode})"
                    )
                await loop.run_in_executor(None, event.wait, min(remaining, 1.0))

            logger.info("Stage %s ready", tp_label)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def any_dead(self) -> bool:
        """Return True if any process in the group has exited unexpectedly."""
        return any(not p.is_alive() and p.exitcode != 0 for p in self._processes)

    def dead_summary(self) -> str:
        """Human-readable summary of dead processes (for error messages)."""
        parts = []
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                parts.append(
                    f"{self.stage_name}:tp{self.specs[i].tp_rank} "
                    f"(pid={p.pid}, exit={p.exitcode})"
                )
        return ", ".join(parts) if parts else "(none)"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self, join_timeout: float = 30.0) -> None:
        """Graceful shutdown: join → terminate → kill."""
        for p in self._processes:
            p.join(timeout=join_timeout)
            if p.is_alive():
                logger.warning("Terminating stuck process %s (pid=%s)", p.name, p.pid)
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        self._processes.clear()
        self._ready_events.clear()
