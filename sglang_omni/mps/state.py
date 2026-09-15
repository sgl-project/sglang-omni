# SPDX-License-Identifier: Apache-2.0
"""Private directory validation for CUDA MPS runtimes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

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


def ensure_private_state_root(root: Path) -> None:
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
