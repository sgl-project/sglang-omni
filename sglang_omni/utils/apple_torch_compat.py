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
    """Import Torch's Inductor Triton heuristics before SGLang stubs ``triton``."""

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
