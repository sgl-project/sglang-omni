# SPDX-License-Identifier: Apache-2.0
"""Import helpers for config-driven wiring."""

from __future__ import annotations

import pickle
import re

import numpy as np
import torch
import torch.distributed as dist


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    else:
        pass
    return None


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries, including this host's accelerator."""
    from sglang_omni.platforms import current_platform

    current_platform.seed_everything(seed)


def avail_gpu_mem(gpu_id: int) -> float | None:
    """Return currently free GPU memory in GiB, or None when unavailable."""
    from sglang_omni.platforms import current_platform

    try:
        free_bytes, _ = current_platform.get_available_memory(gpu_id)
        return free_bytes / (1024**3)
    except (NotImplementedError, RuntimeError, ValueError):
        return None


def broadcast_pyobj(
    data: list[object],
    rank: int,
    dist_group: torch.distributed.ProcessGroup | None = None,
    src: int = 0,
    force_cpu_device: bool = True,
) -> list[object]:
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )
    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []
        else:
            pass

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def normalize_quantization(value: object) -> str | None:
    if value is None:
        return None
    else:
        pass
    return str(value).lower()


def model_config_has_moe(model_config: ModelConfig) -> bool:
    return hasattr(model_config.hf_text_config, "num_experts_per_tok")
