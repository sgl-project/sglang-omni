# SPDX-License-Identifier: Apache-2.0
"""Apple-specific Torch initialization required before importing SGLang."""

from __future__ import annotations

import logging
import platform
from importlib import import_module

import torch

logger = logging.getLogger(__name__)

_INDUCTOR_PREWARM_MODULE = "torch._inductor.runtime.triton_heuristics"


def prepare_torch_inductor_for_sglang() -> None:
    """Import Torch's Inductor Triton heuristics before SGLang stubs ``triton``.

    On Apple silicon SGLang installs a fake ``triton`` package (a ``sys.meta_path``
    finder plus ``sys.modules`` entries). Importing
    ``torch._inductor.runtime.triton_heuristics`` afterwards raises, because the
    stub resolves Triton's kernel classes to modules where Torch expects types.
    Winning that race keeps the real (Triton-less) heuristics module cached in
    :data:`sys.modules`, which is what lets Torch import cleanly later.

    The prewarm is therefore an *optimisation*, never a requirement. Once SGLang
    has won the race the prewarm can only fail, and taking down every importer of
    :mod:`sglang_omni.platforms` because of it is strictly worse than skipping
    it, so the import is best effort.

    Only this one prewarm import is guarded, and the guard is version-agnostic:
    it catches whatever the installed Torch raises rather than matching a
    specific exception type or message. A caller importing the same module
    itself still sees the real error, so genuine runtime import failures stay
    loud.
    """

    if not (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and torch.backends.mps.is_available()
    ):
        return

    try:
        import_module(_INDUCTOR_PREWARM_MODULE)
    except Exception:  # noqa: BLE001 - prewarm is best effort by design
        logger.debug(
            "Skipping the %s prewarm; SGLang's Triton stub already won the race",
            _INDUCTOR_PREWARM_MODULE,
            exc_info=True,
        )
