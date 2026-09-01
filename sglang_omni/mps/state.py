# SPDX-License-Identifier: Apache-2.0
"""Filesystem layouts and locking primitives for CUDA MPS runtimes."""

from __future__ import annotations

import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX unit-test hosts
    fcntl = None

# AF_UNIX sun_path is 108 bytes including the terminator on Linux.
_SUN_PATH_LIMIT = 107


def validate_control_socket(control_socket: Path) -> None:
    socket_bytes = len(str(control_socket).encode())
    if socket_bytes > _SUN_PATH_LIMIT:
        # Note (Jiaxin Deng): over the limit the daemon starts, fails to bind,
        # and exits reporting only "Cannot find MPS control daemon process".
        raise ValueError(
            f"MPS control socket path is {socket_bytes} bytes, over the "
            f"{_SUN_PATH_LIMIT}-byte AF_UNIX sun_path limit: "
            f"{control_socket}. Use a shorter state root."
        )


_GPU_DIR_PATTERN = re.compile(r"(GPU|MIG)-[0-9a-fA-F-]+")


@dataclass(frozen=True)
class MpsGpuPaths:
    """Layout of the shared per-physical-GPU MPS state directory.

    One daemon serves every pipeline that colocates work on this GPU, so the
    directory is keyed by the immutable device UUID, not by run or ordinal.
    """

    state_root: Path
    gpu_uuid: str

    def __post_init__(self) -> None:
        if not _GPU_DIR_PATTERN.fullmatch(self.gpu_uuid):
            raise ValueError(f"unexpected GPU uuid {self.gpu_uuid!r}")

    @property
    def state_dir(self) -> Path:
        return self.state_root / self.gpu_uuid

    @property
    def pipe_dir(self) -> Path:
        return self.state_dir / "pipe"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "log"

    @property
    def owners_dir(self) -> Path:
        return self.state_dir / "owners"

    @property
    def control_socket(self) -> Path:
        return self.pipe_dir / "control"


def _ensure_private_state_root(root: Path) -> None:
    """Create a private state root, or validate an existing caller path."""

    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode):
            raise ValueError(f"MPS state root must not be a symlink: {root}")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"MPS state root is not a directory: {root}")
        if root_stat.st_uid != os.getuid():
            raise ValueError(
                f"MPS state root {root} is owned by uid {root_stat.st_uid}, "
                f"not current uid {os.getuid()}"
            )
        mode = stat.S_IMODE(root_stat.st_mode)
        if mode != 0o700:
            raise ValueError(
                f"MPS state root {root} has mode {mode:#05o}; expected 0o700"
            )
    else:
        # mkdir honors umask. Tightening a directory created by this call is
        # safe; caller-provided paths are never mutated.
        root.chmod(0o700)


@contextmanager
def state_root_lock(root: Path, lock_name: str = ".lock"):
    """Serialize daemon create/join/leave for one GPU across processes.

    No-op where flock is unavailable (non-POSIX unit-test hosts).
    """
    _ensure_private_state_root(root)
    if fcntl is None:
        yield
        return
    with open(root / lock_name, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
