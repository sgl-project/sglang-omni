# SPDX-License-Identifier: Apache-2.0
"""Read the baseline file, and merge one environment's measurements into it under a lock."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path

from scripts.xpu.baseline.gate import merged_serve_extra
from scripts.xpu.baseline.models import BaselineFile, Environment, Record


def load_baseline(path: Path) -> BaselineFile:
    if path.exists():
        return json.loads(path.read_text())
    else:
        return {"environments": {}}


def write_entry(
    path: Path,
    env_key: str,
    environment: Environment,
    extras: dict[str, tuple[str, ...]],
    measured: dict[str, Record],
) -> None:
    """Merge under a sidecar lock and publish by rename, so concurrent recordings keep both and a reader never sees half a file."""
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        on_disk = load_baseline(path)
        previous = on_disk["environments"].get(env_key, {})
        on_disk["environments"][env_key] = {
            **environment,
            "serve_extra": merged_serve_extra(
                previous.get("serve_extra", {}), extras, set(measured)
            ),
            "models": {**previous.get("models", {}), **measured},
        }
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        handle, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(handle, "w") as staging:
                staging.write(json.dumps(on_disk, indent=2, sort_keys=True) + "\n")
            os.chmod(staged, mode)
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise
