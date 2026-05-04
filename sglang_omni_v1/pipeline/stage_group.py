# SPDX-License-Identifier: Apache-2.0
"""StageGroup — manages the OS processes backing one logical pipeline stage.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager
from typing import Sequence

from sglang_omni_v1.pipeline.stage_process import (
    StageProcessSpec,
    get_stage_process_env,
    stage_process_main,
)

logger = logging.getLogger(__name__)


@contextmanager
def _patched_spawn_env(spec: StageProcessSpec):
    updates = get_stage_process_env(spec)
    if not updates:
        yield
        return

    backup = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    @property
    def tp_size(self) -> int:
        return len(self.specs)

    @property
    def leader_spec(self) -> StageProcessSpec:
        for spec in self.specs:
            if spec.role in {"single", "leader"}:
                return spec
        raise RuntimeError(f"StageGroup {self.stage_name} has no leader-owned spec")

    @property
    def leader_endpoint(self) -> str:
        """Control-plane recv endpoint for tp_rank 0 (used by Coordinator)."""
        return self.leader_spec.recv_endpoint

    @property
    def processes(self) -> list[multiprocessing.Process]:
        return list(self._processes)

    def spawn(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn one OS process per TP rank."""
        for spec in self.specs:
            event = ctx.Event()
            if spec.role == "single":
                proc_name = f"stage-{spec.stage_name}"
            elif spec.role == "leader":
                proc_name = f"stage-{spec.stage_name}-leader"
            else:
                proc_name = f"stage-{spec.stage_name}-tp{spec.tp_rank}-follower"
            proc = ctx.Process(
                target=stage_process_main,
                args=(spec, event),
                name=proc_name,
                daemon=True,
            )
            with _patched_spawn_env(spec):
                proc.start()
            self._processes.append(proc)
            self._ready_events.append(event)

        logger.info(
            "StageGroup %s: spawned %d process(es) (pids=%s)",
            self.stage_name,
            len(self._processes),
            [p.pid for p in self._processes],
        )

    async def wait_ready(self, timeout: float) -> None:
        """Block until every TP rank signals ready or *timeout* expires."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        for i, event in enumerate(self._ready_events):
            proc = self._processes[i]
            spec = self.specs[i]
            tp_label = f"{self.stage_name}:{spec.role}:tp{spec.tp_rank}"

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

    def any_dead(self) -> bool:
        """Return True if any process in the group has exited unexpectedly."""
        return any(not p.is_alive() and p.exitcode != 0 for p in self._processes)

    def dead_summary(self) -> str:
        """Human-readable summary of dead processes (for error messages)."""
        parts = []
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                parts.append(
                    f"{self.stage_name}:{self.specs[i].role}:tp{self.specs[i].tp_rank} "
                    f"(pid={p.pid}, exit={p.exitcode})"
                )
        return ", ".join(parts) if parts else "(none)"

    async def shutdown(self, join_timeout: float = 30.0) -> None:

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
