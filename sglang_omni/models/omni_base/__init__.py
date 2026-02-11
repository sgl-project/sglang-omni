# SPDX-License-Identifier: Apache-2.0
"""Shared omni-model abstractions and helpers."""

from sglang_omni.models.omni_base.merge import (
    create_standard_merge_fn,
    default_build_thinker_inputs,
)
from sglang_omni.models.omni_base.stages import (
    create_aggregate_executor,
    create_decode_executor,
    create_encoder_executor,
    create_thinker_executor,
    event_to_dict,
)
from sglang_omni.models.omni_base.types import OmniEvent, PromptInputs, ThinkerOutput

__all__ = [
    # IO types
    "OmniEvent",
    "PromptInputs",
    "ThinkerOutput",
    # Merge utilities
    "create_standard_merge_fn",
    "default_build_thinker_inputs",
    # Stage factories (the real value!)
    "create_aggregate_executor",
    "create_decode_executor",
    "create_encoder_executor",
    "create_thinker_executor",
    "event_to_dict",
]
