# SPDX-License-Identifier: Apache-2.0
"""Accelerator naming for the Ming-Omni-TTS graph paths.

Ming captures two graphs of its own -- the DiTAR tail step in the engine stage
and the streaming AudioVAE transition in the vocoder stage. Both reach streams,
allocator counters and the graph type through the device module the tensors
already live on, so the same capture serves CUDA and XPU without a branch per
call site.
"""

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager, nullcontext
from types import ModuleType
from typing import Any

import torch

# Note (siju): Every accelerator names its graph type after itself and none of
# them share a base class, so there is nothing to isinstance against. Importing
# the type instead would fail at module load on a build without that backend,
# which is exactly the case this has to survive.
_GRAPH_TYPE_NAMES = ("CUDAGraph", "XPUGraph")


def accelerator_module(device: torch.device) -> ModuleType | None:
    """The torch device module backing an accelerator, or None for CPU.

    The CPU path is Ming's internal-verification path: it has no streams, no
    graphs and nothing to synchronize, and callers skip that work on None.
    """
    if device.type == "cpu":
        return None
    return torch.get_device_module(device)


def accelerator_graph_class(device_module: ModuleType) -> type:
    """The graph type this accelerator captures into."""
    for name in _GRAPH_TYPE_NAMES:
        graph_class = getattr(device_module, name, None)
        if graph_class is not None:
            return graph_class
    raise RuntimeError(
        f"{device_module.__name__} exposes no graph type, so Ming-Omni-TTS "
        "cannot capture graphs on this device"
    )


def graph_capture(
    device_module: ModuleType,
    graph: Any,
    *,
    stream: Any | None = None,
    thread_local_errors: bool = False,
) -> AbstractContextManager[Any]:
    """Open a capture on device_module, honoring what its context accepts.

    CUDA scopes a capture failure to the whole process by default, so a caller
    that wants the failure kept to the capturing thread asks for it here. XPU's
    graph context declares no such parameter, and passing one is a TypeError
    rather than a no-op, so the request is dropped where it does not exist.
    """
    kwargs: dict[str, Any] = {}
    if stream is not None:
        kwargs["stream"] = stream
    if thread_local_errors and _accepts_capture_error_mode(device_module):
        kwargs["capture_error_mode"] = "thread_local"
    return device_module.graph(graph, **kwargs)


def _accepts_capture_error_mode(device_module: ModuleType) -> bool:
    # Note (siju): Read off the signature rather than a per-backend table so a
    # backend that gains the parameter later starts getting it, instead of
    # silently keeping the process-global scope a stale table would pin it to.
    parameters = inspect.signature(device_module.graph).parameters
    return "capture_error_mode" in parameters


def graph_capture_attention() -> AbstractContextManager[Any]:
    """Pin SDPA to the backends this platform can capture, if it names any.

    Ming's two captures both run scaled-dot-product attention, and a platform
    whose default SDPA selection is not capturable needs the choice made before
    capture opens. A platform that names nothing keeps its own dispatch.
    """
    from sglang_omni.platforms import current_platform

    backends = current_platform.get_graph_capture_sdpa_backends()
    if not backends:
        return nullcontext()

    from torch.nn.attention import sdpa_kernel

    return sdpa_kernel(list(backends))


__all__ = [
    "accelerator_graph_class",
    "accelerator_module",
    "graph_capture",
    "graph_capture_attention",
]
