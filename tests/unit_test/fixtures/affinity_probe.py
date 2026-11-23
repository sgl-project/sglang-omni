# SPDX-License-Identifier: Apache-2.0
"""Spawn target that reports the CPU mask of every thread in its process."""

from __future__ import annotations

import os
import threading
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

from sglang_omni.pipeline.stage_workers import StageWorkerProcessSpec

SPAWNED_ENV = "SGLANG_OMNI_TEST_AFFINITY_PROBE_CHILD"

# note (Richard Wang): a thread started while the child imports its target,
# before the target runs, as native thread pools are. A spawned child imports
# this module before multiprocessing records its parent, so the spawning test
# marks the child through the environment it inherits.
import_thread = None
if os.environ.get(SPAWNED_ENV) == "1":
    import_thread = threading.Thread(target=threading.Event().wait, daemon=True)
    import_thread.start()
else:
    pass


def report_thread_cpus(
    spec: StageWorkerProcessSpec,
    ready_event: Event,
    startup_error_channel: Queue[tuple[list[int] | None, list[list[int]]]],
) -> None:
    startup_error_channel.put(
        (
            (
                None
                if import_thread is None
                else sorted(os.sched_getaffinity(import_thread.native_id))
            ),
            [
                sorted(os.sched_getaffinity(int(tid)))
                for tid in os.listdir("/proc/self/task")
            ],
        )
    )
