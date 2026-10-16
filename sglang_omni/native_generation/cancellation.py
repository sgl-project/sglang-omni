# SPDX-License-Identifier: Apache-2.0
"""Cooperative cancellation for native multimodal_gen generation run by Omni.

Only the native workers that an Omni native generation stage starts can
find this plugin. Inside ``activation_environment`` the stage writes a
minimal distribution that declares the ``sglang.multimodal_gen.plugins``
entry point into its private handshake directory, puts that directory on
the import path and names it in ACTIVATION_ENV. Native starts its workers
with the spawn context, which hands both to each child, so native plugin
discovery finds the plugin in those workers alone. Every other process in
the environment, the stage host included, sees no plugin.

The stage arms a request by creating ARMED_MARKER in the request's output
directory and cancels it by creating CANCELLED_MARKER beside it. Inside the
native worker, each pipeline stage call and each step of a progress loop
checks the cancelled marker of an armed request and raises
GenerationCancelled, which native error handling turns into a failed output
for that request only.

Keep module scope limited to the standard library: native loads plugins in
each worker before its runtime modules are imported.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# The entry point name in the sglang.multimodal_gen.plugins group, and the
# distribution that declares it for armed workers.
PLUGIN_GROUP = "sglang.multimodal_gen.plugins"
PLUGIN_NAME = "omni_generation_cancellation"
PLUGIN_DIST = "sglang_omni_native_cancellation"
ACTIVATION_ENV = "SGLANG_OMNI_NATIVE_CANCELLATION_DIR"
ARMED_MARKER = ".sglang-omni-armed"
CANCELLED_MARKER = ".sglang-omni-cancelled"

# Hook targets, pinned to SGLang 0.5.21.
STAGE_CALL_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.PipelineStage.__call__"
)
PROGRESS_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
    "PipelineStage.progress_bar"
)
WORKER_TARGET = (
    "sglang.multimodal_gen.runtime.managers.gpu_worker.run_scheduler_process"
)


class GenerationCancelled(RuntimeError):
    """Raised inside native generation for a request that Omni cancelled."""


class CancellationState:
    """Per process switch, enabled only after the worker handshake."""

    enabled = False


def handshake_path(directory: str, rank: int) -> str:
    return os.path.join(directory, f"active-rank{rank}")


def write_entry_point(directory: str) -> None:
    """Declare the plugin entry point in a distribution under ``directory``."""
    dist_info = os.path.join(directory, f"{PLUGIN_DIST}-0.dist-info")
    os.mkdir(dist_info)
    with open(os.path.join(dist_info, "METADATA"), "w") as handle:
        handle.write(f"Metadata-Version: 2.1\nName: {PLUGIN_DIST}\nVersion: 0\n")
    with open(os.path.join(dist_info, "entry_points.txt"), "w") as handle:
        handle.write(f"[{PLUGIN_GROUP}]\n{PLUGIN_NAME} = {__name__}:register\n")


@contextmanager
def activation_environment(directory: str | None) -> Generator[None, None, None]:
    """Arm the native workers started inside this block, then restore.

    The directory goes last on the import path, so it shadows no module, and
    leaves it on exit. Workers keep the copy of the path they started with.
    """
    if directory is None:
        yield
        return
    else:
        pass
    write_entry_point(directory)
    previous = os.environ.get(ACTIVATION_ENV)
    os.environ[ACTIVATION_ENV] = directory
    sys.path.append(directory)
    try:
        yield
    finally:
        if directory in sys.path:
            sys.path.remove(directory)
        else:
            pass
        if previous is None:
            os.environ.pop(ACTIVATION_ENV, None)
        else:
            os.environ[ACTIVATION_ENV] = previous


def cancelled_marker(batch: object) -> str | None:
    """The cancelled marker path of an armed request, else None."""
    directory = getattr(batch, "output_path", None)
    if not isinstance(directory, str):
        return None
    elif not os.path.exists(os.path.join(directory, ARMED_MARKER)):
        return None
    else:
        return os.path.join(directory, CANCELLED_MARKER)


def raise_if_cancelled(marker: str) -> None:
    if os.path.exists(marker):
        raise GenerationCancelled("Generation cancelled by its Omni stage")
    else:
        pass


def checked_steps(iterable: Iterable[object], marker: str) -> Iterator[object]:
    for item in iterable:
        raise_if_cancelled(marker)
        yield item


def before_stage_call(
    stage: object, batch: object = None, *args: object, **kwargs: object
) -> None:
    """Stop a cancelled request at a pipeline stage boundary."""
    if CancellationState.enabled:
        marker = cancelled_marker(batch)
        if marker is not None:
            raise_if_cancelled(marker)
        else:
            pass
    else:
        pass


def around_progress_bar(
    original: Callable[..., object], stage: object, *args: object, **kwargs: object
) -> object:
    """Check an armed request's cancelled marker before every loop step."""
    marker = (
        cancelled_marker(kwargs.get("batch")) if CancellationState.enabled else None
    )
    if marker is None:
        return original(stage, *args, **kwargs)
    else:
        pass
    raise_if_cancelled(marker)
    iterable = args[0] if args else kwargs.get("iterable")
    if iterable is not None:
        total = args[1] if len(args) > 1 else kwargs.get("total")
        if total is None and hasattr(iterable, "__len__"):
            kwargs["total"] = len(iterable)
        else:
            pass
        steps = checked_steps(iterable, marker)
        if args:
            args = (steps, *args[1:])
        else:
            kwargs["iterable"] = steps
        return original(stage, *args, **kwargs)
    else:
        pass
    bar = original(stage, *args, **kwargs)
    update = bar.update

    def checked_update(n: float = 1) -> object:
        raise_if_cancelled(marker)
        return update(n)

    try:
        bar.update = checked_update
    except AttributeError:
        pass
    return bar


def announce_worker(*args: object, **kwargs: object) -> None:
    """Enable the hooks and write the handshake once they are in place.

    Never raises, so a mismatch leaves native startup untouched and the stage
    falls back to waiting for aborted work.
    """
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
            PipelineStage,
        )
        from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

        directory = os.environ.get(ACTIVATION_ENV)
        rank = kwargs.get("rank")
        patched = (
            HookRegistry._patched
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        applied = {STAGE_CALL_TARGET, PROGRESS_TARGET} <= patched
        parameters = inspect.signature(PipelineStage.progress_bar).parameters
        # Independent checks on several ranks could disagree at a collective.
        single_rank = getattr(kwargs.get("server_args"), "num_gpus", None) == 1
        if directory and isinstance(rank, int) and applied and "batch" in parameters:
            CancellationState.enabled = single_rank
        else:
            pass
        if CancellationState.enabled:
            with open(handshake_path(directory, rank), "w"):
                pass
        else:
            logger.warning(
                "Omni cooperative cancellation is off in this native worker "
                "(hooks applied=%s, single rank=%s)",
                applied,
                single_rank,
            )
    except Exception:
        CancellationState.enabled = False
        logger.exception("Omni cooperative cancellation could not start")


def register() -> None:
    """Entry point callback that native plugin discovery runs in armed workers."""
    if not os.environ.get(ACTIVATION_ENV):
        return
    else:
        pass
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(STAGE_CALL_TARGET, before_stage_call, HookType.BEFORE)
    HookRegistry.register(PROGRESS_TARGET, around_progress_bar, HookType.AROUND)
    HookRegistry.register(WORKER_TARGET, announce_worker, HookType.BEFORE)
