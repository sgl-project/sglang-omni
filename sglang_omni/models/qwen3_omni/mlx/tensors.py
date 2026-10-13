"""CPU pipeline tensor boundaries and explicit multimodal positions."""

import mlx.core as mx
import numpy as np
import torch
from sglang.srt.managers.schedule_batch import Req


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.detach().cpu().float().numpy())


def mlx_to_torch(tensor: mx.array) -> torch.Tensor:
    mx.eval(tensor)
    return torch.from_numpy(np.array(tensor.astype(mx.float32), copy=True))


def request_positions(request: Req, token_count: int) -> tuple[mx.array, int]:
    multimodal = request.multimodal_inputs
    if multimodal is None or multimodal.mrope_positions is None:
        return (
            mx.broadcast_to(mx.arange(token_count)[None, None, :], (3, 1, token_count)),
            0,
        )
    else:
        pass
    positions = multimodal.mrope_positions
    if tuple(positions.shape) != (3, token_count):
        raise ValueError(
            f"M-RoPE positions must cover all {token_count} tokens, got {positions.shape}"
        )
    else:
        pass
    delta = multimodal.mrope_position_delta
    if delta is None:
        raise ValueError("Multimodal positions require their decode delta")
    else:
        pass
    return torch_to_mlx(positions).astype(mx.int32)[:, None, :], int(
        torch.as_tensor(delta).reshape(-1)[0]
    )
