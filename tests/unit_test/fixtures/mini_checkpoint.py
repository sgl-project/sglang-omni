# SPDX-License-Identifier: Apache-2.0
"""A checkpoint directory small enough for the resolution pipeline.

The pipeline returns before the cuda graph handler on a dummy model path, so a
test that needs a fully resolved record points the builder at a directory
holding this config.
"""

from __future__ import annotations

import json
from pathlib import Path

MINI_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}


def write_mini_llama_checkpoint(directory: Path) -> str:
    (directory / "config.json").write_text(json.dumps(MINI_LLAMA_CONFIG))
    return str(directory)
