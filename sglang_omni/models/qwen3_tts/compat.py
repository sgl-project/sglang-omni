# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for upstream qwen-tts."""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable

import torch

_APPLY_LOCK = threading.Lock()
_PATCHED_FLAG = "_sglang_omni_qwen_tts_compat_patched"
# Note (Akazaakane): the factories qwen-tts 0.1.1 imports. It splats one
# mask_kwargs dict into both, so shimming create_causal_mask alone just moves
# the failure to the next line.
_MASK_FACTORY_NAMES = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
)


def _compute_default_rope_parameters(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def _make_mask_factory_compat(
    original: Callable[..., Any], name: str
) -> Callable[..., Any]:
    def mask_factory_compat(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            kwargs.setdefault("inputs_embeds", kwargs.pop("input_embeds"))
        kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    mask_factory_compat.__name__ = getattr(original, "__name__", name)
    mask_factory_compat.__doc__ = getattr(original, "__doc__", None)
    setattr(mask_factory_compat, _PATCHED_FLAG, True)
    return mask_factory_compat


def _patch_mask_factories() -> None:
    """Accept the qwen-tts call shape for the Transformers mask factories."""
    from transformers import masking_utils

    for name in _MASK_FACTORY_NAMES:
        original = getattr(masking_utils, name, None)
        if original is None or getattr(original, _PATCHED_FLAG, False):
            continue

        try:
            parameters = inspect.signature(original).parameters
        except (TypeError, ValueError):
            continue

        if "inputs_embeds" not in parameters or "input_embeds" in parameters:
            continue

        setattr(masking_utils, name, _make_mask_factory_compat(original, name))


def apply_qwen_tts_transformers_compatibility_patches() -> None:
    """Patch Transformers APIs expected by qwen-tts."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    with _APPLY_LOCK:
        ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)
        _patch_mask_factories()

        current = generic.check_model_inputs
        if getattr(current, _PATCHED_FLAG, False):
            return

        try:
            signature = inspect.signature(current)
        except (TypeError, ValueError):
            return

        params = list(signature.parameters.values())
        needs_func_arg = (
            len(params) == 1
            and params[0].default is inspect.Parameter.empty
            and params[0].kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if not needs_func_arg:
            return

        original = current

        def check_model_inputs_compat(
            func: Callable[..., Any] | None = None,
        ) -> Callable[..., Any]:
            if func is None:

                def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
                    return original(inner)

                return decorator
            return original(func)

        check_model_inputs_compat.__name__ = getattr(
            original, "__name__", "check_model_inputs"
        )
        check_model_inputs_compat.__doc__ = getattr(original, "__doc__", None)
        setattr(check_model_inputs_compat, _PATCHED_FLAG, True)
        generic.check_model_inputs = check_model_inputs_compat
