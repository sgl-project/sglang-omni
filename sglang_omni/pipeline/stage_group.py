# SPDX-License-Identifier: Apache-2.0
"""StageGroup manages OS processes backing one topology process group."""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager
from typing import Sequence

from sglang_omni.pipeline.stage_process import (
    StageProcessSpec,
    StageWorkerProcessSpec,
    get_stage_process_env,
    stage_process_main,
)

logger = logging.getLogger(__name__)


def _get_worker_process_env(spec: StageWorkerProcessSpec) -> dict[str, str]:
    """Return the spawn-time env overrides for *spec*.

    Hard invariant: a TP stage (``tp_size > 1``) must own its OS process
    exclusively — its CUDA env remap and NCCL settings depend on being the
    sole tenant. Mixing a TP stage with any other stage in the same
    process group is a placement bug, not a fallback case.
    """
    tp_stages = [s for s in spec.stage_specs if s.tp_size > 1]
    if not tp_stages:
        return {}
    if len(tp_stages) > 1 or len(spec.stage_specs) > 1:
        raise AssertionError(
            f"Process {spec.process_name!r} mixes a TP stage with other "
            "stages; TP stages must own their OS process exclusively. "
            f"stage_specs={[s.stage_name for s in spec.stage_specs]}"
        )
    return get_stage_process_env(tp_stages[0])


@contextmanager
def _patched_spawn_env(spec: StageWorkerProcessSpec):
    updates = _get_worker_process_env(spec)
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
    """Lifecycle manager for one or more OS processes in a topology group."""

    def __init__(
        self,
        group_name: str,
        process_specs: Sequence[StageWorkerProcessSpec],
    ):
        if not process_specs:
            raise ValueError(
                f"StageGroup requires at least one process spec (group={group_name})"
            )
        self.group_name = group_name
        self.process_specs = list(process_specs)
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []

    @property
    def process_count(self) -> int:
        return len(self.process_specs)

    @property
    def specs(self) -> list[StageProcessSpec]:
        return [
            stage_spec
            for process_spec in self.process_specs
            for stage_spec in process_spec.stage_specs
        ]

    @property
    def leader_spec(self) -> StageProcessSpec:
        for spec in self.specs:
            if spec.role in {"single", "leader"}:
                return spec
        raise RuntimeError(f"StageGroup {self.group_name} has no leader-owned spec")

    @property
    def leader_endpoint(self) -> str:
        """Control-plane recv endpoint for tp_rank 0 (used by Coordinator)."""
        return self.leader_spec.recv_endpoint

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        return {
            spec.stage_name: spec.recv_endpoint
            for spec in self.specs
            if spec.owns_external_io
        }

    @property
    def processes(self) -> list[multiprocessing.Process]:
        return list(self._processes)

    def spawn(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn the OS process(es) owned by this group."""
        for spec in self.process_specs:
            event = ctx.Event()
            proc_name = _process_name(spec)
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
            self.group_name,
            len(self._processes),
            [p.pid for p in self._processes],
        )

    async def wait_ready(self, timeout: float) -> None:
        """Block until every TP rank signals ready or *timeout* expires."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        for i, event in enumerate(self._ready_events):
            proc = self._processes[i]
            spec = self.process_specs[i]
            process_label = spec.process_name

            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Process {process_label} did not become ready "
                        f"within {timeout:.0f}s"
                    )
                if not proc.is_alive():
                    raise RuntimeError(
                        f"Process {process_label} died during startup "
                        f"(exit code {proc.exitcode})"
                    )
                await loop.run_in_executor(None, event.wait, min(remaining, 1.0))

            logger.info("Process %s ready", process_label)

    def any_dead(self) -> bool:
        """Return True if any process in the group exited while runner is active."""
        return any(not p.is_alive() for p in self._processes)

    def dead_summary(self) -> str:
        """Human-readable summary of dead processes (for error messages)."""
        parts = []
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                process_spec = self.process_specs[i]
                parts.append(
                    f"{process_spec.process_name} " f"(pid={p.pid}, exit={p.exitcode})"
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


def _process_name(spec: StageWorkerProcessSpec) -> str:
    if len(spec.stage_specs) > 1:
        return f"process-{spec.process_name}"
    stage_spec = spec.stage_specs[0]
    if stage_spec.role == "single":
        return f"stage-{stage_spec.stage_name}"
    if stage_spec.role == "leader":
        return f"stage-{stage_spec.stage_name}-leader"
    return f"stage-{stage_spec.stage_name}-tp{stage_spec.tp_rank}-follower"
