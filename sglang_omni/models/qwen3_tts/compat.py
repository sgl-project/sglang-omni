# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for upstream qwen-tts."""
 
from __future__ import annotations
 
import inspect
import threading
from typing import Any, Callable
 
import torch
 
_APPLY_LOCK = threading.Lock()
_PATCHED_FLAG = "_sglang_omni_qwen_tts_compat_patched"
 
 
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
 
 
def _patch_create_causal_mask() -> None:
    """Shim `create_causal_mask` to accept qwen-tts's `input_embeds=` /
    `cache_position=` call signature.
 
    qwen-tts 0.1.1 calls `create_causal_mask(..., input_embeds=..., cache_position=...)`.
    Under transformers==5.12.1 the parameter is spelled `inputs_embeds` and
    `cache_position` has been dropped from the signature entirely, so every
    request 500s. This normalises the call: `input_embeds` is renamed to
    `inputs_embeds`, and `cache_position` is absorbed (dropped) unless the
    installed transformers version still accepts it.
    """
    from transformers import masking_utils
 
    original = masking_utils.create_causal_mask
    if getattr(original, _PATCHED_FLAG, False):
        return
 
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return
 
    accepted_params = set(signature.parameters)
 
    def create_causal_mask_compat(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            value = kwargs.pop("input_embeds")
            if "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = value
 
        if "cache_position" in kwargs and "cache_position" not in accepted_params:
            kwargs.pop("cache_position")
 
        return original(*args, **kwargs)
 
    create_causal_mask_compat.__name__ = getattr(
        original, "__name__", "create_causal_mask"
    )
    create_causal_mask_compat.__doc__ = getattr(original, "__doc__", None)
    setattr(create_causal_mask_compat, _PATCHED_FLAG, True)
 
    masking_utils.create_causal_mask = create_causal_mask_compat
 
    # `create_causal_mask` is commonly re-imported by name into modeling
    # modules (e.g. `from transformers.masking_utils import
    # create_causal_mask`), so patching the source module alone won't reach
    # call sites that already bound the original reference. Patch the other
    # common rebinding point too, best-effort.
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        pass
    else:
        if getattr(modeling_utils, "create_causal_mask", None) is original:
            modeling_utils.create_causal_mask = create_causal_mask_compat
 
 
def apply_qwen_tts_transformers_compatibility_patches() -> None:
    """Patch Transformers APIs expected by qwen-tts."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic
 
    with _APPLY_LOCK:
        ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)
 
        _patch_create_causal_mask()
 
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