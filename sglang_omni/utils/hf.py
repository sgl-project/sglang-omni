# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch.nn as nn
from transformers import AutoConfig

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from transformers.utils.hub import cached_file


@lru_cache(maxsize=8)
def load_hf_config(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> Any:
    """Load the HF config, preferring the local cache."""
    try:
        config_path = cached_file(
            model_id, "config.json", local_files_only=local_files_only
        )
        cfg = AutoConfig.from_pretrained(
            str(Path(config_path).parent),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return cfg


def instantiate_module(module_cls: type[nn.Module], config: Any) -> nn.Module:
    """Instantiate a module without allocating its parameters."""
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(config)
        return module_cls(config)
