# SPDX-License-Identifier: Apache-2.0
"""Hand parameter handles from one PD half to the other at startup.

:mod:`sglang_omni.model_runner.weight_sharing` can export and adopt handles,
but the two halves are separate processes with no channel between them at load
time. The KV plane does not supply one: ``prepare_kv_receive`` and
``send_kv_pages`` are per-transfer and run long after both halves are up.

This uses the directory the run already has. ``create_ipc_runtime_dir`` makes
one private directory per pipeline instance before any stage is spawned, every
stage is handed endpoints inside it, and it is removed when the run ends. A
file there is therefore visible to both halves, private to the run, and cleaned
up without new ownership rules.

Publishing is a write to a temporary name followed by ``os.replace``, so a
reader never observes a partial file. Reading returns ``None`` when the peer
has not published rather than waiting for it: ``_construct_scheduler`` builds
each stage inside ``gpu_startup_lock(gpu_id)``, so two halves on one device
load one at a time, and a reader that waited would hold that lock against the
very half it is waiting for. A half that finds nothing publishes its own
handles instead, so whichever loads second is the one that adopts.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_SUBDIR = "pd-weights"
_POLL_INTERVAL_S = 0.2


class RendezvousUnavailable(RuntimeError):
    """The run's IPC directory could not be derived from an endpoint."""


class PublisherUnavailable(RuntimeError):
    """The recorded publisher generation is no longer alive."""


@dataclass(frozen=True)
class PublishedWeightManifest:
    generation: str
    publisher_pid: int
    publisher_start_identity: str | None
    gpu_id: int
    parameters: dict[str, Any]
    weight_bytes: int


def _process_start_identity(pid: int) -> str | None:
    try:
        # Linux /proc stat field 22 is stable for one process generation even
        # when a PID is later reused.
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, IndexError, OSError):
        return None


def assert_publisher_alive(manifest: PublishedWeightManifest) -> None:
    try:
        os.kill(manifest.publisher_pid, 0)
    except ProcessLookupError as exc:
        raise PublisherUnavailable(
            f"weight publisher process {manifest.publisher_pid} is unavailable"
        ) from exc
    except PermissionError:
        pass
    expected = manifest.publisher_start_identity
    if (
        expected is not None
        and _process_start_identity(manifest.publisher_pid) != expected
    ):
        raise PublisherUnavailable(
            f"weight publisher process {manifest.publisher_pid} generation changed"
        )


def rendezvous_dir_from_endpoint(endpoint: str) -> Path:
    """Return the run's directory, given any ``ipc://`` endpoint from this run.

    ``allocate_endpoints`` puts every socket directly in the run directory, so
    the parent of an endpoint path is that directory. Deriving it here keeps
    the halves from needing a new argument threaded through stage startup.
    """
    if not endpoint.startswith("ipc://"):
        raise RendezvousUnavailable(
            f"expected an ipc:// endpoint to locate the run directory, got {endpoint!r}"
        )
    return Path(endpoint[len("ipc://") :]).parent


def publish_parameter_handles(
    handles: dict[str, Any],
    *,
    rendezvous_dir: Path,
    stage_name: str,
    gpu_id: int,
    weight_bytes: int = 0,
    publisher_pid: int | None = None,
) -> Path:
    """Write *handles* where the peer half can read them. Returns the path.

    ``weight_bytes`` travels with them so the adopting half can check that it
    has room to materialize its own copy before it loads one. On one card the
    publisher is already holding weights and its KV pool by then, and the
    adopter still has to load before it can swap.

    The device is recorded alongside them. A CUDA IPC handle names memory on
    one GPU, so a half on another card must not adopt these, and stating the
    device here lets the reader check that rather than assume it.
    """
    directory = Path(rendezvous_dir) / _SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{stage_name}.pkl"
    staging = directory / f"{stage_name}.pkl.{os.getpid()}"
    pid = os.getpid() if publisher_pid is None else int(publisher_pid)
    manifest = PublishedWeightManifest(
        generation=uuid4().hex,
        publisher_pid=pid,
        publisher_start_identity=_process_start_identity(pid),
        gpu_id=int(gpu_id),
        parameters=handles,
        weight_bytes=int(weight_bytes),
    )
    staging.write_bytes(pickle.dumps(manifest))
    os.replace(staging, final)
    logger.info(
        "published %d parameter handles for %s at %s",
        len(handles),
        stage_name,
        final,
    )
    return final


def read_parameter_handles(
    *,
    rendezvous_dir: Path,
    stage_name: str,
    gpu_id: int,
) -> PublishedWeightManifest | None:
    """Return the handles *stage_name* published for *gpu_id*, or None.

    Returns None when the peer has not published, and when it published for a
    different device.

    This does not wait. ``_construct_scheduler`` builds a stage inside
    ``gpu_startup_lock(gpu_id)``, so two halves on one device load one at a
    time and the second one to load finds the first one's file already there.
    Waiting here would instead hold that lock against the half being waited
    for, which needs the same lock to load at all.
    """
    path = Path(rendezvous_dir) / _SUBDIR / f"{stage_name}.pkl"
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        logger.info(
            "%s has not published parameter handles; this half publishes its own",
            stage_name,
        )
        return None
    published = pickle.loads(payload)
    if not isinstance(published, PublishedWeightManifest):
        raise PublisherUnavailable(
            f"{stage_name} published an unversioned weight manifest"
        )
    if published.gpu_id != int(gpu_id):
        logger.info(
            "%s published handles for GPU %s, not GPU %s; this half keeps its own",
            stage_name,
            published.gpu_id,
            gpu_id,
        )
        return None
    assert_publisher_alive(published)
    logger.info(
        "adopted %d parameter handles from %s generation=%s",
        len(published.parameters),
        stage_name,
        published.generation,
    )
    return published


def read_published_weight_bytes(
    *,
    rendezvous_dir: Path,
    stage_name: str,
) -> int:
    """Return the byte count the publisher recorded, or 0 if it recorded none."""
    path = Path(rendezvous_dir) / _SUBDIR / f"{stage_name}.pkl"
    try:
        published = pickle.loads(path.read_bytes())
    except (FileNotFoundError, KeyError, EOFError):
        return 0
    if isinstance(published, PublishedWeightManifest):
        return int(published.weight_bytes)
    return int(published.get("weight_bytes", 0))


def wait_for_parameter_handles(
    *,
    rendezvous_dir: Path,
    stage_name: str,
    gpu_id: int,
    timeout_s: float,
) -> PublishedWeightManifest | None:
    """Block until *stage_name* publishes, then return its handles or None.

    The wait is for the file to appear, not for handles this half can use.
    A peer on another device publishes handles that name memory on that
    device, and no amount of waiting changes that, so this returns as soon as
    it can decide. Treating a device mismatch as "not yet" made a cross-GPU
    pair wait out the whole timeout and fail to start.

    Only safe to call before taking ``gpu_startup_lock``. Inside the lock this
    would hold it against the very half being waited for, which is why
    :func:`read_parameter_handles` does not wait.

    Returning None lets the caller load its own weights rather than fail the
    stage, which is the right trade both when the peer is absent and when it
    is on another card.
    """
    path = Path(rendezvous_dir) / _SUBDIR / f"{stage_name}.pkl"
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            logger.warning(
                "%s published no parameter handles within %.0fs; "
                "this half loads its own",
                stage_name,
                timeout_s,
            )
            return None
        time.sleep(_POLL_INTERVAL_S)
    return read_parameter_handles(
        rendezvous_dir=rendezvous_dir, stage_name=stage_name, gpu_id=gpu_id
    )
