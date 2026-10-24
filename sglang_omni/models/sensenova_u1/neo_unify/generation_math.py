"""Seeded image noise."""

import contextlib

import torch


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
