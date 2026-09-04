# SPDX-License-Identifier: Apache-2.0
"""Routable-worker snapshot: the CP -> DP handoff for the multi-process router.

The control plane serializes its registry view to one JSON file after every
state change; data-plane processes poll it cheaply (stat + seq check) and
rebuild their read-only worker view from it. Writes are atomic (temp file +
os.replace) so readers never observe a torn file; a reader that does find an
invalid file keeps its last good snapshot and retries on the next poll.
"""

from __future__ import annotations

import os
import time

from pydantic import BaseModel, Field

SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotWorker(BaseModel):
    url: str
    worker_id: str
    incarnation: str = ""
    model: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    routable: bool = True
    state: str = "healthy"
    disabled: bool = False
    # Note (Jiaxin Deng): reserved for a power-of-two-choices load hint; not
    # populated or read anywhere yet.
    load_hint: float | None = None


class WorkerSnapshot(BaseModel):
    version: int = SNAPSHOT_SCHEMA_VERSION
    seq: int
    cp_epoch: str
    generated_at: float
    workers: list[SnapshotWorker]


class SnapshotWriter:
    """Single-writer (CP-owned) publisher with a monotonic per-epoch seq."""

    def __init__(self, path: str, cp_epoch: str) -> None:
        self._path = path
        self._cp_epoch = cp_epoch
        self._seq = 0

    @property
    def seq(self) -> int:
        return self._seq

    def publish(self, workers: list[SnapshotWorker]) -> WorkerSnapshot:
        self._seq += 1
        snapshot = WorkerSnapshot(
            seq=self._seq,
            cp_epoch=self._cp_epoch,
            generated_at=time.time(),
            workers=workers,
        )
        tmp_path = f"{self._path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json())
            os.replace(tmp_path, self._path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return snapshot

    def unlink(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


class SnapshotReader:
    """DP-side poller.

    Contract: torn or invalid files never replace the last good snapshot;
    within one cp_epoch only strictly newer seqs are applied; a new cp_epoch
    (CP restart) is always applied even though its seq restarts from 1.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._last_fingerprint: tuple[int, int, int] | None = None
        self._snapshot: WorkerSnapshot | None = None

    @property
    def snapshot(self) -> WorkerSnapshot | None:
        return self._snapshot

    def age_secs(self, now: float | None = None) -> float | None:
        if self._snapshot is None:
            return None
        current = now if now is not None else time.time()
        return current - self._snapshot.generated_at

    def maybe_reload(self) -> bool:
        try:
            stat_result = os.stat(self._path)
        except OSError:
            return False
        # Note (Jiaxin Deng): mtime alone is tick-granular and can be shared
        # by rapid publishes; os.replace gives every publish a fresh inode, so
        # (ino, size, mtime) does change.
        fingerprint = (
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
        )
        if fingerprint == self._last_fingerprint:
            return False
        try:
            with open(self._path, encoding="utf-8") as f:
                snapshot = WorkerSnapshot.model_validate_json(f.read())
        except (OSError, ValueError):
            # Note (Jiaxin Deng): the fingerprint is not advanced, so a torn
            # file is retried on the next poll.
            return False
        if snapshot.version != SNAPSHOT_SCHEMA_VERSION:
            return False
        self._last_fingerprint = fingerprint
        if (
            self._snapshot is not None
            and snapshot.cp_epoch == self._snapshot.cp_epoch
            and snapshot.seq <= self._snapshot.seq
        ):
            return False
        self._snapshot = snapshot
        return True
