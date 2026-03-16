# SPDX-License-Identifier: Apache-2.0
"""Hook-based hidden state capture for multi-layer extraction.

SGLang's VL wrapper (Qwen3VLForConditionalGeneration) doesn't support
the aux_hidden_states tuple returned by the text model when layers_to_capture
is set. This module wraps the text model's forward to intercept that tuple,
store aux_hidden_states on a side-channel, and return only the plain
hidden_states so the VL wrapper's logits_processor works correctly.

The OutputProcessor then reads from the side-channel to build per-layer
hidden state dicts.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def install_hidden_capture_hooks(
    model: nn.Module,
    capture_layers: list[int],
) -> None:
    """Install forward wrapper on the text model to capture aux hidden states.

    Args:
        model: Top-level SGLang model (e.g. Qwen3OmniMoeForConditionalGeneration)
        capture_layers: Layer indices to capture (e.g. [0, 24]).
            Layer 0 captures embed output (input to first transformer layer).
            Layer N captures input to layer N (= output of layer N-1).
    """
    # Navigate to the text model that has layers_to_capture
    # Qwen3OmniMoeForConditionalGeneration -> .thinker -> .model (Qwen3MoeLLMModel)
    # Qwen3OmniTalker -> .model (text model)
    if hasattr(model, "thinker"):
        text_model = model.thinker.model
    elif hasattr(model, "model"):
        text_model = model.model
    else:
        raise AttributeError(
            f"Cannot find text model on {type(model).__name__}. "
            "Expected .thinker.model or .model attribute."
        )

    # Set layers_to_capture on the text model
    text_model.layers_to_capture = list(capture_layers)

    # Storage for captured aux hidden states (overwritten each forward pass)
    model._captured_aux_hidden_states = None

    # Wrap the text model's forward to intercept tuple returns
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
        "Installed hidden capture hooks on %s for layers %s",
        type(text_model).__name__,
        capture_layers,
    )
