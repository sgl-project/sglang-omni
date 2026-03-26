# SPDX-License-Identifier: Apache-2.0
"""Graph-safe hidden state capture for multi-layer extraction.

Uses pre-allocated GPU buffers and register_forward_hook to capture
intermediate hidden states from transformer layers. The in-place copy_
operations within hooks are recorded as CUDA kernels during graph capture,
so hidden states are correctly updated during both eager and graph replay.

This replaces the previous Python-wrapper approach that intercepted tuple
returns from the text model's forward. That approach was incompatible with
CUDA graph replay because Python code does not execute during replay, only
captured CUDA kernels replay. It also had an integer/string mismatch bug
where layers_to_capture=[0, 24] never triggered the model's "embed" check.

Design reference: vllm-omni separates graph-captured operations from
host-side result processing via make_omni_output and dedicated GPU buffers
(PRs #523, #669). We adopt the same principle: hooks write to fixed buffers
inside the graph, and the output processor reads from them after replay.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GraphSafeHiddenCapture:
    """Graph-compatible hidden state capture using pre-allocated buffers.

    Hooks registered via register_forward_hook fire during eager execution
    and during CUDA graph capture. The copy_ operations they perform are
    recorded as CUDA kernels. During graph replay, these kernels execute
    at the same memory addresses, correctly updating the buffers with new
    intermediate hidden states.

    Requirements for CUDA graph safety:
    - Buffers must be allocated before graph capture and not freed/moved.
    - Intermediate tensors (hook outputs) use graph-pool addresses that
      remain stable across replays.
    - Only in-place operations (copy_) are used; no Python-side state
      changes are relied upon during replay.
    """

    def __init__(
        self,
        model: nn.Module,
        capture_layers: list[int],
        max_num_tokens: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capture_layers = capture_layers
        self.max_num_tokens = max_num_tokens
        self._num_tokens = 0
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        text_model = _find_text_model(model)

        self.buffers: dict[str | int, torch.Tensor] = {}
        for layer_id in capture_layers:
            key = "embed" if layer_id == 0 else layer_id
            self.buffers[key] = torch.zeros(
                max_num_tokens,
                hidden_size,
                device=device,
                dtype=dtype,
            )

        self._install_hooks(text_model)

        logger.info(
            "Installed graph-safe hidden capture on %s for layers %s "
            "(max_tokens=%d, hidden=%d)",
            type(text_model).__name__,
            capture_layers,
            max_num_tokens,
            hidden_size,
        )

    def _install_hooks(self, text_model: nn.Module) -> None:
        for layer_id in self.capture_layers:
            if layer_id == 0:
                target = getattr(text_model, "embed_tokens", None)
                if target is None:
                    get_fn = getattr(text_model, "get_input_embeddings", None)
                    if callable(get_fn):
                        target = get_fn()
                if target is not None:
                    buf = self.buffers["embed"]
                    handle = target.register_forward_hook(
                        _make_embed_hook(buf)
                    )
                    self._handles.append(handle)
                    logger.debug("Hook installed on embed_tokens for layer 0")
            else:
                layers = getattr(text_model, "layers", None)
                if layers is not None and (layer_id - 1) < len(layers):
                    key = layer_id
                    buf = self.buffers[key]
                    handle = layers[layer_id - 1].register_forward_hook(
                        _make_layer_hook(buf)
                    )
                    self._handles.append(handle)
                    logger.debug(
                        "Hook installed on layers[%d] for capture layer %d",
                        layer_id - 1,
                        layer_id,
                    )

    def set_num_tokens(self, n: int) -> None:
        """Record the number of actual (non-padded) tokens for slicing."""
        self._num_tokens = n

    def get_hidden_states(self, num_tokens: Optional[int] = None) -> dict:
        """Get captured hidden states, cloned and sliced to actual token count."""
        n = num_tokens if num_tokens is not None else self._num_tokens
        result = {}
        for key, buf in self.buffers.items():
            n_clamped = min(n, buf.shape[0])
            result[key] = buf[:n_clamped].clone()
        return result

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _make_embed_hook(buffer: torch.Tensor):
    """Hook for embedding layer output (returns tensor directly)."""

    def hook(module, input, output):
        hidden = output
        n = hidden.shape[0]
        if n <= buffer.shape[0]:
            buffer[:n].copy_(hidden[:n])

    return hook


def _make_layer_hook(buffer: torch.Tensor):
    """Hook for transformer layer output (returns (hidden_states, residual))."""

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        n = hidden.shape[0]
        if n <= buffer.shape[0]:
            buffer[:n].copy_(hidden[:n])

    return hook


def _find_text_model(model: nn.Module) -> nn.Module:
    """Navigate model hierarchy to find the text backbone."""
    if hasattr(model, "thinker"):
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    raise AttributeError(
        f"Cannot find text model on {type(model).__name__}. "
        "Expected .thinker.model or .model attribute."
    )


def install_hidden_capture_hooks(
    model: nn.Module,
    capture_layers: list[int],
) -> None:
    """Legacy entry point -- still works for eager-only (no graph) execution.

    Kept for backward compatibility. New code should use
    GraphSafeHiddenCapture for CUDA graph compatibility.
    """
    text_model = _find_text_model(model)
    text_model.layers_to_capture = list(capture_layers)
    model._captured_aux_hidden_states = None

    import functools
    from typing import Any

    original_forward = text_model.forward

    @functools.wraps(original_forward)
    def _capturing_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_forward(*args, **kwargs)
        if isinstance(result, tuple):
            hidden_states, aux_hidden_states = result
            model._captured_aux_hidden_states = aux_hidden_states
            return hidden_states
        else:
            model._captured_aux_hidden_states = None
            return result

    text_model.forward = _capturing_forward
    logger.info(
        "Installed legacy hidden capture hooks on %s for layers %s",
        type(text_model).__name__,
        capture_layers,
    )
