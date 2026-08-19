# SPDX-License-Identifier: Apache-2.0
"""Graph-safe hidden-state capture for multi-layer extraction."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _check_rows(num_rows: int, max_tokens: int) -> None:
    if num_rows > max_tokens:
        raise RuntimeError(
            "Static aux hidden capture received "
            f"{num_rows} rows but capacity is {max_tokens}"
        )


class StaticAuxHiddenCapture:
    def __init__(
        self,
        *,
        buffers: list[torch.Tensor],
        hook_handles: list[Any],
        max_tokens: int,
    ) -> None:
        self.buffers = tuple(buffers)
        self.max_tokens = max_tokens
        self._hook_handles = tuple(hook_handles)

    def views(self, num_rows: int) -> list[torch.Tensor]:
        # The caller owns forward freshness: graph replay refreshes these
        # buffers without executing Python-side row bookkeeping.
        _check_rows(num_rows, self.max_tokens)
        return [buffer[:num_rows] for buffer in self.buffers]


def _layer_input_capture_hook(buffer: torch.Tensor, max_tokens: int):
    buffer_dtype = buffer.dtype

    def _capture(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(args) > 1:
            hidden_states = args[1]
        else:
            raise RuntimeError("Decoder layer call is missing hidden_states")

        if "residual" in kwargs:
            residual = kwargs["residual"]
        elif len(args) > 3:
            residual = args[3]
        else:
            residual = None

        num_rows = hidden_states.shape[0]
        _check_rows(num_rows, max_tokens)
        layer_input = hidden_states if residual is None else hidden_states + residual
        if layer_input.dtype != buffer_dtype:
            raise RuntimeError(
                f"Static aux hidden capture expected {buffer_dtype} layer input "
                f"but got {layer_input.dtype}"
            )
        buffer[:num_rows].copy_(layer_input)

    return _capture


def install_hidden_capture_hooks(
    model: nn.Module,
    capture_layers: list[int],
    *,
    max_tokens: int,
) -> None:
    """Capture decoder-layer inputs into stable, non-persistent buffers.

    Layer 0 captures the embedding output (input to the first decoder layer);
    layer N captures the input to layer N (= output of layer N-1). max_tokens
    bounds the token rows of one eager or graph-backed forward.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if not capture_layers:
        raise ValueError("capture_layers must not be empty")
    if len(set(capture_layers)) != len(capture_layers):
        raise ValueError(f"capture_layers contains duplicates: {capture_layers}")

    if hasattr(model, "thinker"):
        text_model = model.thinker.model
    elif hasattr(model, "model"):
        text_model = model.model
    else:
        raise AttributeError(
            f"Cannot find text model on {type(model).__name__}. "
            "Expected .thinker.model or .model attribute."
        )

    if getattr(model, "_omni_aux_hidden_capture", None) is not None:
        raise RuntimeError("Static aux hidden capture is already installed")

    start_layer = int(getattr(text_model, "start_layer", 0))
    end_layer = int(getattr(text_model, "end_layer", len(text_model.layers)))
    nonlocal_layers = [
        layer_id
        for layer_id in capture_layers
        if layer_id < start_layer or layer_id >= end_layer
    ]
    if nonlocal_layers:
        raise ValueError(
            f"Capture layers {nonlocal_layers} are outside local layer range "
            f"[{start_layer}, {end_layer})"
        )

    embedding_weight = text_model.get_input_embeddings().weight
    hidden_size = int(embedding_weight.shape[1])
    buffers: list[torch.Tensor] = []
    for layer_id in capture_layers:
        name = f"_omni_aux_hidden_layer_{layer_id}"
        if hasattr(text_model, name):
            raise RuntimeError(f"Capture buffer {name} is already registered")
        text_model.register_buffer(
            name,
            embedding_weight.new_empty((max_tokens, hidden_size)),
            persistent=False,
        )
        buffers.append(getattr(text_model, name))

    hook_handles: list[Any] = []
    capture = StaticAuxHiddenCapture(
        buffers=buffers,
        hook_handles=hook_handles,
        max_tokens=max_tokens,
    )
    for layer_id, buffer in zip(capture_layers, buffers):
        hook_handles.append(
            text_model.layers[layer_id].register_forward_pre_hook(
                _layer_input_capture_hook(buffer, max_tokens),
                with_kwargs=True,
            )
        )
    capture._hook_handles = tuple(hook_handles)

    # Note (wenyao): the inner model must keep returning a single tensor, so the
    # upstream tuple-return capture path stays disabled and the hooks above own
    # every copy, which also puts them inside CUDA graph capture.
    text_model.layers_to_capture = []
    model._omni_aux_hidden_capture = capture
    logger.info(
        "Installed static hidden capture on %s for layers %s with %d-token capacity",
        type(text_model).__name__,
        capture_layers,
        max_tokens,
    )
