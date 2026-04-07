# SPDX-License-Identifier: Apache-2.0
"""Framework-level torch.compile support for omni models.

See: https://github.com/sgl-project/sglang-omni/issues/172
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)

DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"
FALLBACK_COMPILE_MODE = "max-autotune"


@runtime_checkable
class CompilableModel(Protocol):
    def get_compile_targets(self) -> dict[str, Callable]: ...


def _set_inductor_config() -> None:
    """Match SGLang's CudaGraphRunner compile settings."""
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
    torch._dynamo.config.accumulated_cache_size_limit = 1024


def apply_compile_targets(
    *models: Any,
    compile_mode: str = DEFAULT_COMPILE_MODE,
) -> list[str]:
    """Compile all registered targets on one or more models.

    For targets named "X", calls set_compiled_X_fn() if available,
    otherwise stores as _compiled_X on the model.
    """
    _set_inductor_config()
    all_compiled = []

    for model in models:
        if not isinstance(model, CompilableModel):
            continue

        targets = model.get_compile_targets()
        if not targets:
            continue

        for name, fn in targets.items():
            compiled_fn = _compile_function(fn, name, compile_mode)
            if compiled_fn is None:
                continue

            setter_name = f"set_compiled_{name}_fn"
            if hasattr(model, setter_name):
                getattr(model, setter_name)(compiled_fn)
            else:
                setattr(model, f"_compiled_{name}", compiled_fn)
            all_compiled.append(name)

    if all_compiled:
        logger.info(
            "Compiled %d target(s) with mode '%s': %s",
            len(all_compiled),
            compile_mode,
            ", ".join(all_compiled),
        )

    return all_compiled


def _compile_function(
    fn: Callable,
    name: str,
    compile_mode: str,
) -> Callable | None:
    """Compile a single function with fullgraph=True, with fallback."""
    try:
        return torch.compile(fn, mode=compile_mode, fullgraph=True)
    except Exception:
        logger.warning(
            "torch.compile mode '%s' failed for '%s'; trying '%s'.",
            compile_mode,
            name,
            FALLBACK_COMPILE_MODE,
        )
    try:
        return torch.compile(fn, mode=FALLBACK_COMPILE_MODE, fullgraph=True)
    except Exception:
        logger.error(
            "torch.compile failed for '%s' with all modes. Running eager.",
            name,
        )
        return None
