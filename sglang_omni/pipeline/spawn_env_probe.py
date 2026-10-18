# SPDX-License-Identifier: Apache-2.0
"""Resolve the platform and stage spawn envs in a short-lived process.

A pipeline's server process stays free of torch and SGLang. When a stage
process needs its env set before it starts, the platform policy that sets it
runs here, in a child process that exits once it has answered.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing.spawn
import pickle
import subprocess
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageWorkerProcessSpec
else:
    pass


@dataclasses.dataclass(frozen=True)
class ProbeReply:
    platform_spec: str
    device_count: int | None
    worker_envs: dict[str, dict[str, str]]


REPLY_PREFIX = "sglang-omni-spawn-env-probe "
# Like a multiprocessing spawn, the child takes the parent's import path before
# it unpickles the specs.
BOOTSTRAP = """
import pickle
import sys

request = pickle.load(sys.stdin.buffer)
sys.path[:] = request["sys_path"]
from sglang_omni.pipeline.spawn_env_probe import answer

answer(pickle.loads(request["specs"]))
"""


def without_spawn_only_fields(value: object) -> object:
    """A dataclass copy whose fields only a process spawn can carry are reset.

    Queues and events refuse to pickle outside a spawn, and no platform policy
    reads them, so they go back to their defaults.
    """
    changes = {}
    for field in dataclasses.fields(value):
        try:
            pickle.dumps(getattr(value, field.name))
        except (RuntimeError, TypeError, pickle.PicklingError):
            if field.default_factory is not dataclasses.MISSING:
                changes[field.name] = field.default_factory()
            else:
                changes[field.name] = field.default
        else:
            pass
    return dataclasses.replace(value, **changes)


def probe(specs: Sequence[StageWorkerProcessSpec], *, timeout: float) -> ProbeReply:
    """The platform, the visible device count and the spawn envs, from a child."""
    payload = [
        dataclasses.replace(
            without_spawn_only_fields(spec),
            stage_specs=[without_spawn_only_fields(s) for s in spec.stage_specs],
        )
        for spec in specs
    ]
    request = {"sys_path": list(sys.path), "specs": pickle.dumps(payload)}
    result = subprocess.run(
        [multiprocessing.spawn.get_executable(), "-c", BOOTSTRAP],
        input=pickle.dumps(request),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    replies = [
        line[len(REPLY_PREFIX) :]
        for line in result.stdout.decode(errors="replace").splitlines()
        if line.startswith(REPLY_PREFIX)
    ]
    if result.returncode != 0 or not replies:
        detail = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(
            f"Resolving the platform for stage spawn envs failed with exit "
            f"code {result.returncode}: {detail}"
        )
    else:
        pass
    reply = json.loads(replies[-1])
    return ProbeReply(reply["platform"], reply["device_count"], reply["envs"])


def answer(specs: Sequence[StageWorkerProcessSpec]) -> None:
    """Resolve the platform, then print it, the device count and the envs."""
    from sglang_omni import platforms
    from sglang_omni.pipeline.runtime_config import visible_device_count
    from sglang_omni.pipeline.stage_workers import get_worker_process_env

    platform = platforms.get_current_platform()
    reply = {
        "platform": platforms.get_platform_spec(platform),
        "device_count": visible_device_count(),
        "envs": {spec.process_name: get_worker_process_env(spec) for spec in specs},
    }
    print(REPLY_PREFIX + json.dumps(reply), flush=True)
