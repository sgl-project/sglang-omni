# SPDX-License-Identifier: Apache-2.0
"""Filesystem exchange for one weight-IPC run."""

from __future__ import annotations

import os
import pickle
import stat
import tempfile
import time
from pathlib import Path

from sglang_omni.distributed.weight_ipc.types import SCHEMA_VERSION, WeightIpcBundle

BUNDLE_NAME = "bundle.pkl"
READY_NAME = "READY"
MANIFEST_NAME = "MANIFEST"


class WeightIpcStore:
    def __init__(self, store_dir: str | os.PathLike[str]) -> None:
        self.root = Path(store_dir)

    def prepare(self) -> None:
        try:
            root_stat = self.root.lstat()
        except FileNotFoundError:
            self.root.mkdir(parents=True, mode=0o700)
        else:
            self._validate_root_stat(root_stat)
        os.chmod(self.root, 0o700)
        self._validate_secure_root()

    def prepare_for_write(self) -> None:
        """Prepare a private store and remove any previous publication."""
        self.prepare()
        self.cleanup()

    @property
    def bundle_path(self) -> Path:
        return self.root / BUNDLE_NAME

    @property
    def ready_path(self) -> Path:
        return self.root / READY_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def write_bundle(self, bundle: WeightIpcBundle) -> None:
        if bundle.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={bundle.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        self.prepare()
        self.ready_path.unlink(missing_ok=True)
        payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_bytes(self.bundle_path, payload)
        self._atomic_write_text(
            self.manifest_path,
            (
                f"schema_version={bundle.schema_version}\n"
                f"n_tensors={len(bundle.tensors)}\n"
                f"name_digest={bundle.name_digest}\n"
                f"model_path={bundle.model_path}\n"
                f"leader_pid={bundle.leader_pid}\n"
            ),
        )
        # Note (guozhihao): READY last so followers only observe a durable bundle.
        self._atomic_write_text(self.ready_path, "1\n")

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                self._validate_secure_root()
                self._validate_private_file(self.ready_path)
                self._validate_private_file(self.bundle_path)
            except FileNotFoundError:
                pass
            else:
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"weight IPC READY not found under {self.root} within {timeout_s}s"
        )

    def load_bundle(self) -> WeightIpcBundle:
        self._validate_secure_root()
        self._validate_private_file(self.bundle_path)
        bundle = pickle.loads(self.bundle_path.read_bytes())
        if not isinstance(bundle, WeightIpcBundle):
            raise TypeError(f"unexpected bundle type: {type(bundle)!r}")
        if bundle.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: got {bundle.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        return bundle

    def cleanup(self) -> None:
        try:
            self._validate_secure_root()
        except FileNotFoundError:
            return
        for name in (READY_NAME, BUNDLE_NAME, MANIFEST_NAME):
            path = self.root / name
            if path.exists() or path.is_symlink():
                path.unlink()

    def _validate_secure_root(self) -> None:
        self._validate_root_stat(self.root.lstat())

    def _validate_root_stat(self, root_stat: os.stat_result) -> None:
        if not stat.S_ISDIR(root_stat.st_mode):
            raise PermissionError(
                f"weight IPC store must be a real directory, got {self.root}"
            )
        if root_stat.st_uid != os.geteuid():
            raise PermissionError(
                f"weight IPC store {self.root} is owned by uid={root_stat.st_uid}, "
                f"expected uid={os.geteuid()}"
            )
        mode = stat.S_IMODE(root_stat.st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"weight IPC store {self.root} must not grant group/world "
                f"permissions, got mode={mode:#o}"
            )

    def _validate_private_file(self, path: Path) -> None:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise PermissionError(
                f"weight IPC store entry must be a regular file: {path}"
            )
        if file_stat.st_uid != os.geteuid():
            raise PermissionError(
                f"weight IPC store entry {path} is owned by uid={file_stat.st_uid}, "
                f"expected uid={os.geteuid()}"
            )
        mode = stat.S_IMODE(file_stat.st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"weight IPC store entry {path} must not grant group/world "
                f"permissions, got mode={mode:#o}"
            )

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        self._atomic_write_bytes(path, payload.encode("utf-8"))

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
