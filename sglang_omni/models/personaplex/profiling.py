# SPDX-License-Identifier: Apache-2.0
"""Optional torch-profiler scopes for PersonaPlex inference components."""

from contextlib import nullcontext
from typing import Literal

from torch.autograd.profiler import record_function

from sglang_omni.platforms import current_platform


def component_scope(
    component_name: Literal[
        "temporal_transformer",
        "text_logits",
        "depformer",
        "mimi_encode",
        "mimi_decode",
        "h2d",
        "d2h",
        "embeddings",
    ],
) -> record_function | nullcontext[None]:
    """Record host launch ranges without synchronizing device execution."""
    if current_platform.get_torch_profiler().is_active():
        return record_function(f"personaplex.{component_name}")
    else:
        return nullcontext()
