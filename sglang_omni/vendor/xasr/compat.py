"""Inference-only replacements for upstream icefall/k2 training dependencies."""

from contextlib import contextmanager

import torch


@contextmanager
def torch_autocast(enabled=False):
    with torch.autocast(device_type="cuda", enabled=enabled):
        yield


class _K2Inference:
    @staticmethod
    def swoosh_l_forward(x):
        return torch.nn.functional.softplus(x - 4) - 0.08 * x - 0.035

    @staticmethod
    def swoosh_r_forward(x):
        return torch.nn.functional.softplus(x - 1) - 0.08 * x - 0.313261687

    swoosh_l = swoosh_l_forward
    swoosh_r = swoosh_r_forward


k2 = _K2Inference()
