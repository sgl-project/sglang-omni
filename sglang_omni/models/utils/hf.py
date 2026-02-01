# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch.nn as nn
from transformers import AutoConfig
from accelerate import init_empty_weights as no_init_weights


@lru_cache(maxsize=8)
def load_hf_config(
    model_path: str,
    *,
    trust_remote_code: bool = True,
) -> Any:
    """Load the HF config from a local model path."""
    return AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )


def instantiate_module(module_cls: type[nn.Module], config: Any) -> nn.Module:
    """Instantiate a module without allocating its parameters."""
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(config)
        return module_cls(config)
