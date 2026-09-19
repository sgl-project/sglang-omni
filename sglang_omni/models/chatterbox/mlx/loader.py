# SPDX-License-Identifier: Apache-2.0
"""Weight loading for the Chatterbox-Turbo MLX T3 model."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from safetensors import safe_open

_CONV1D_WEIGHT_SUFFIXES = ("c_proj", "c_fc")


def _split_fused_qkv(name: str, tensor: Any) -> dict[str, mx.array]:
    """Split a fused Conv1D c_attn weight/bias into q/k/v projections."""
    prefix = name[: name.index("attn.c_attn")]
    is_weight = name.endswith(".weight")
    fused = mx.array(tensor.t().contiguous().numpy() if is_weight else tensor.numpy())
    q, k, v = mx.split(fused, 3, axis=0)
    suffix = "weight" if is_weight else "bias"
    return {
        f"{prefix}attn.q_proj.{suffix}": q,
        f"{prefix}attn.k_proj.{suffix}": k,
        f"{prefix}attn.v_proj.{suffix}": v,
    }


def load_t3_weights(model: Any, checkpoint_dir: str) -> Any:
    """Load the T3 checkpoint into the MLX model.

    The checkpoint stores the GPT-2 backbone under a tfmr. prefix with
    HF-style Conv1D weights (transposed relative to Linear); tfmr.wte is
    unused and skipped, and the fused c_attn is split into q/k/v so the
    SGLang MLX contract is satisfied.
    """
    path = os.path.join(checkpoint_dir, "t3_turbo_v1.safetensors")
    weights: dict[str, mx.array] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.startswith("tfmr.wte."):
                continue
            new_name = name
            if new_name.startswith("tfmr."):
                new_name = new_name[len("tfmr.") :]
            new_name = new_name.replace("cond_enc.spkr_enc", "cond_enc")
            tensor = f.get_tensor(name)

            if "attn.c_attn" in new_name:
                weights.update(_split_fused_qkv(new_name, tensor))
                continue
            if new_name.endswith(".weight") and any(
                suffix in new_name for suffix in _CONV1D_WEIGHT_SUFFIXES
            ):
                tensor = tensor.t().contiguous()
            new_name = new_name.replace("attn.c_proj", "attn.o_proj")
            weights[new_name] = mx.array(tensor.numpy())
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model
