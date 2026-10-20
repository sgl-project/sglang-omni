"""Generation math."""

import contextlib
import operator
import time
from typing import Literal

import torch.utils.checkpoint
from packaging import version

# Modified for SGLang; see this directory's README.md for upstream source.


class SenseNovaStageTimer:
    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled: bool = enabled
        self.device: torch.device = device
        self.timings_ms: dict[str, float] = {}
        self.started_at: float = time.perf_counter()
        self.last_mark: float = self.started_at

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        else:
            pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        now = time.perf_counter()
        self.timings_ms[name] = round((now - self.last_mark) * 1000, 3)
        self.last_mark: float = now

    def finish(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        else:
            pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        self.timings_ms["total"] = round(
            (time.perf_counter() - self.started_at) * 1000, 3
        )
        return self.timings_ms


def version_cmp(
    v1: str, v2: str, op: Literal["eq", "ne", "lt", "le", "gt", "ge"] = "eq"
) -> bool:
    comparison = getattr(operator, op)
    return comparison(version.parse(v1), version.parse(v2))


def optimized_scale(
    positive_flat: torch.Tensor, negative_flat: torch.Tensor
) -> torch.Tensor:
    device_type = positive_flat.device.type
    if device_type == "mps":
        device_type = "cpu"
    else:
        pass
    with torch.autocast(device_type=device_type, enabled=False):
        positive_flat = positive_flat.float()
        negative_flat = negative_flat.float()

        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

        st_star = dot_product / squared_norm

    return st_star


def randn_with_seed(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int | list[int],
) -> torch.Tensor:
    if isinstance(seed, list):
        if len(shape) == 0 or len(seed) != shape[0]:
            raise ValueError(
                f"expected one seed per batch item, got {len(seed)} seeds for shape {shape}"
            )
        else:
            pass
        return torch.cat(
            [
                randn_with_seed(
                    (1, *shape[1:]), device=device, dtype=dtype, seed=item_seed
                )
                for item_seed in seed
            ],
            dim=0,
        )
    else:
        pass
    try:
        generator = torch.Generator(device)
    except (RuntimeError, TypeError):
        device = torch.device(device)
        if device.type == "cpu":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                return torch.randn(shape, device=device, dtype=dtype)
        else:
            pass

        device_module = torch.get_device_module(device)
        with torch.random.fork_rng(devices=[device], device_type=device.type):
            device_context = (
                device_module.device(device)
                if hasattr(device_module, "device")
                else contextlib.nullcontext()
            )
            with device_context:
                device_module.manual_seed(seed)
            return torch.randn(shape, device=device, dtype=dtype)
    generator.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute patch coordinates (x, y)"""
    device = grid_hw.device
    B = grid_hw.shape[0]

    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)

    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = (
        patch_id_within_image
        - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[
            patch_to_sample
        ]
    )

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y
